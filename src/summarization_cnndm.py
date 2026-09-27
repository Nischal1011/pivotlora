import os, sys, csv, time, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from datasets import load_dataset
from transformers import (
    AutoTokenizer, GPT2LMHeadModel, TrainingArguments, Trainer,
    TrainerCallback, set_seed,
)
from transformers.pytorch_utils import Conv1D
from sklearn.decomposition import PCA
import hashlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bert"))
from train_classification import (
    LoRALayer,
    compute_pca_eigen,
    compute_pissa_init,
)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

MODEL_NAME       = "gpt2"
RANK             = 8
ALPHA            = 16
BATCH_SIZE       = 4
LEARNING_RATE    = 2e-4
EPOCHS           = 15
MAX_INPUT_LEN    = 512
MAX_TARGET_LEN   = 64
SUMMARY_MARKER   = "\n\nTL;DR: "

LR_RATIO           = 16
PIVOT_EPOCH   = 3
PIVOT_K_FRACTION    = 0.5
PIVOT_WARMUP_STEPS = 10

SEED_LIST  = [42, 43, 44]
SHOTS      = [64, 128, 256, 512]
METHODS    = ["lora", "pivotlora"]

RESULTS_CSV = "results_cnndm.csv"
CSV_FIELDS  = ["task", "shot", "method", "seed", "rouge1", "rouge2", "rougeL"]


# ══════════════════════════════════════════════════════════════
# Conv1D LoRA (identical to gen_experiment.py)
# ══════════════════════════════════════════════════════════════

class Conv1DLoRALayer(nn.Module):
    def __init__(self, original_conv1d, rank, alpha,
                 init_A=None, init_B=None, track_A_init=False,
                 b_lr_scales=None):
        super().__init__()
        self.original = original_conv1d
        self.rank     = rank
        self.scaling  = alpha / rank
        d_in  = original_conv1d.weight.shape[0]
        d_out = original_conv1d.weight.shape[1]
        device = original_conv1d.weight.device

        if init_A is not None:
            scale = np.sqrt(5.0) / np.sqrt(d_in)
            self.lora_A = nn.Parameter(
                init_A.to(device=device, dtype=torch.float32) * scale)
        else:
            self.lora_A = nn.Parameter(torch.empty(d_in, rank, device=device))
            nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))

        self._split_b = (b_lr_scales is not None)
        if self._split_b:
            rows = [nn.Parameter((init_B[ri:ri+1, :] if init_B is not None
                                  else torch.zeros(1, d_out)).to(device=device, dtype=torch.float32))
                    for ri in range(rank)]
            self.lora_B_rows = nn.ParameterList(rows)
            self.lora_B = None
            self.register_buffer("b_lr_scales",
                torch.tensor(b_lr_scales, dtype=torch.float32, device=device))
        else:
            self.lora_B = nn.Parameter(
                (init_B.to(device=device, dtype=torch.float32).clone()
                 if init_B is not None
                 else torch.zeros(rank, d_out, device=device)))
            self.lora_B_rows = None
            self.b_lr_scales = None

        if track_A_init:
            self.register_buffer("A_init_snapshot", self.lora_A.data.clone().detach())
        else:
            self.A_init_snapshot = None

        original_conv1d.weight.requires_grad = False
        if original_conv1d.bias is not None:
            original_conv1d.bias.requires_grad = False

    @property
    def lora_B_matrix(self):
        return torch.cat(list(self.lora_B_rows), dim=0) if self._split_b else self.lora_B

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A @ self.lora_B_matrix) * self.scaling


# ══════════════════════════════════════════════════════════════
# CNN/DailyMail data loading
# ══════════════════════════════════════════════════════════════

def load_cnndm_splits(n_shot, seed):
    """
    CNN/DailyMail with summary-only label masking.
    Format: "{article}\n\nTL;DR: {highlights}<EOS>"
    Identical protocol to XSum, different dataset.
    """
    print(f"\n[DATA] Loading CNN/DailyMail n_shot={n_shot} seed={seed}...")
    set_seed(seed)
    np.random.seed(seed)

    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    marker_ids = tokenizer.encode(SUMMARY_MARKER, add_special_tokens=False)
    marker_len = len(marker_ids)

    def tokenize_fn(examples):
        all_input_ids, all_attention_mask, all_labels = [], [], []
        for article, highlights in zip(examples["article"], examples["highlights"]):
            summary_ids = tokenizer.encode(
                highlights + tokenizer.eos_token, add_special_tokens=False)
            doc_budget = max(10, MAX_INPUT_LEN - marker_len - len(summary_ids))
            doc_ids = tokenizer.encode(article, add_special_tokens=False,
                                       max_length=doc_budget, truncation=True)
            input_ids = (doc_ids + marker_ids + summary_ids)[:MAX_INPUT_LEN]

            n_masked = len(doc_ids) + marker_len
            labels = ([-100] * min(n_masked, len(input_ids))
                      + input_ids[min(n_masked, len(input_ids)):])

            pad_len = MAX_INPUT_LEN - len(input_ids)
            all_input_ids.append(input_ids + [tokenizer.pad_token_id] * pad_len)
            all_attention_mask.append([1] * len(input_ids) + [0] * pad_len)
            all_labels.append(labels + [-100] * pad_len)

        return {"input_ids": all_input_ids,
                "attention_mask": all_attention_mask,
                "labels": all_labels}

    dataset = dataset.map(tokenize_fn, batched=True,
                          remove_columns=dataset["train"].column_names)
    dataset.set_format(type="torch",
                       columns=["input_ids", "attention_mask", "labels"])

    train_full = dataset["train"]
    val_ds = dataset["validation"]

    rng = np.random.default_rng(seed)
    idxs = np.arange(len(train_full))
    rng.shuffle(idxs)
    train_ds = train_full.select(idxs[:n_shot].tolist())

    sample = train_ds[0]
    n_active = (sample["labels"] != -100).sum().item()
    n_nonpad = (sample["attention_mask"] == 1).sum().item()
    print(f"[DATA] Train={len(train_ds)} Val={len(val_ds)} | "
          f"label check: {n_active}/{n_nonpad} active "
          f"({100*n_active/max(n_nonpad,1):.0f}% = summary only)")
    return train_ds, val_ds, tokenizer


# ══════════════════════════════════════════════════════════════
# Subspace computation (identical to gen_experiment.py)
# ══════════════════════════════════════════════════════════════

def get_gpt2_subspaces(model, dataset, rank, seed):
    print(f"[SUBSPACE] Computing per-target pca_eigen...")
    model.eval().to(DEVICE)
    acts_cproj = defaultdict(list)
    acts_mlpfc = defaultdict(list)

    def make_hook(store, idx):
        def hook(module, inputs):
            store[idx].append(inputs[0].detach().mean(dim=1).cpu())
        return hook

    handles = []
    for i, block in enumerate(model.transformer.h):
        handles.append(block.attn.c_proj.register_forward_pre_hook(
            make_hook(acts_cproj, i)))
        handles.append(block.mlp.c_fc.register_forward_pre_hook(
            make_hook(acts_mlpfc, i)))

    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            model(**{k: v.to(DEVICE) for k, v in batch.items() if k != "labels"})
    for h in handles:
        h.remove()

    subspaces = defaultdict(dict)
    eig_dict = defaultdict(dict)
    for target, acts in [("c_proj", acts_cproj), ("c_fc", acts_mlpfc)]:
        for idx in acts:
            X = torch.cat(acts[idx], dim=0).numpy()
            Q, eigs = compute_pca_eigen(X, rank)
            subspaces[idx][target] = Q.to(DEVICE)
            eig_dict[idx][target] = eigs
    print(f"[SUBSPACE] Done.")
    return dict(subspaces), dict(eig_dict)


# ══════════════════════════════════════════════════════════════
# Injection (identical to gen_experiment.py)
# ══════════════════════════════════════════════════════════════

def inject_gpt2(model, rank, block_subspaces=None,
                eigenvalues_dict=None, track_A_init=False):
    target_attrs = [("attn", "c_proj"), ("mlp", "c_fc")]
    count = 0
    for i, block in enumerate(model.transformer.h):
        for parent_name, target_name in target_attrs:
            parent = getattr(block, parent_name)
            original = getattr(parent, target_name)

            curr_A = None
            if block_subspaces and i in block_subspaces:
                layer_sub = block_subspaces[i]
                curr_A = layer_sub.get(target_name) if isinstance(layer_sub, dict) else layer_sub

            if curr_A is not None:
                seed_int = int.from_bytes(
                    hashlib.sha256(f"layer_{i}_{target_name}".encode()).digest()[:4], "little")
                g = torch.Generator(device=DEVICE); g.manual_seed(seed_int)
                rot, _ = torch.linalg.qr(torch.randn(rank, rank, generator=g, device=DEVICE))
                curr_A = curr_A @ rot

            b_scales = None
            if eigenvalues_dict and i in eigenvalues_dict:
                layer_eigs = eigenvalues_dict[i]
                eigs = layer_eigs.get(target_name) if isinstance(layer_eigs, dict) else layer_eigs
                if eigs is not None:
                    b_scales = (eigs / (float(eigs.mean()) + 1e-8)).astype(np.float32)

            adapter = Conv1DLoRALayer(original, rank, ALPHA,
                                      init_A=curr_A, track_A_init=track_A_init,
                                      b_lr_scales=b_scales)
            setattr(parent, target_name, adapter)
            count += 1

    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, Conv1DLoRALayer):
            m.lora_A.requires_grad = True
            if m._split_b:
                for row in m.lora_B_rows: row.requires_grad = True
            else: m.lora_B.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INJECT] {count} adapters | trainable: {trainable:,}")
    return model


# ══════════════════════════════════════════════════════════════
# Optimizers
# ══════════════════════════════════════════════════════════════

def build_eigen_optimizer(model):
    A_params = [p for n, p in model.named_parameters()
                if p.requires_grad and "lora_A" in n]
    param_groups = [{"params": A_params, "lr": LEARNING_RATE, "weight_decay": 0.01}]
    for _, m in model.named_modules():
        if not isinstance(m, Conv1DLoRALayer): continue
        if m._split_b:
            for ri, row in enumerate(m.lora_B_rows):
                param_groups.append({"params": [row],
                    "lr": m.b_lr_scales[ri].item() * LR_RATIO * LEARNING_RATE,
                    "weight_decay": 0.01, "_is_B_group": True})
        else:
            param_groups.append({"params": [m.lora_B],
                "lr": LR_RATIO * LEARNING_RATE,
                "weight_decay": 0.01, "_is_B_group": True})
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)


def build_uniform_optimizer(model):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)


# ══════════════════════════════════════════════════════════════
# PivotLoRA callback
# ══════════════════════════════════════════════════════════════

class GPT2PivotCallback(TrainerCallback):
    def __init__(self, pivot_epoch, k_fraction, warmup_steps, lr_ratio, base_lr):
        self.pivot_epoch = pivot_epoch; self.k_fraction = k_fraction
        self.warmup_steps = warmup_steps; self.lr_ratio = lr_ratio
        self.base_lr = base_lr; self.pivot_done = False
        self.warmup_remaining = 0; self._optimizer = None

    def on_epoch_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        if self.pivot_done or state.epoch is None: return
        if state.epoch + 1e-4 < self.pivot_epoch or state.epoch > self.pivot_epoch + 1.0: return
        opt = optimizer or self._optimizer
        k = max(1, int(RANK * self.k_fraction)); pivoted = 0
        for name, m in model.named_modules():
            if not isinstance(m, Conv1DLoRALayer): continue
            if m.A_init_snapshot is None: continue
            with torch.no_grad():
                delta_A = m.lora_A.data - m.A_init_snapshot
                if torch.norm(delta_A).item() < 1e-8: continue
                try: U, S, _ = torch.linalg.svd(delta_A, full_matrices=False)
                except: continue
                A_current = m.lora_A.data.clone()
                Q_aug, _ = torch.linalg.qr(torch.cat([A_current, U[:, :k]], dim=1))
                A_new = Q_aug[:, :RANK]
                B_new = (A_new.T @ A_current) @ m.lora_B_matrix
                m.lora_A.data.copy_(A_new)
                if m._split_b:
                    for ri in range(m.rank): m.lora_B_rows[ri].data.copy_(B_new[ri:ri+1, :])
                else: m.lora_B.data.copy_(B_new)
                if opt:
                    for group in opt.param_groups:
                        for p in group["params"]:
                            if p is m.lora_A:
                                st = opt.state.get(p, {})
                                if "exp_avg" in st: st["exp_avg"].zero_(); st["exp_avg_sq"].zero_()
                pivoted += 1
        print(f"[PivotLoRA] Pivoted {pivoted} layers.")
        self.pivot_done = True; self.warmup_remaining = self.warmup_steps
        if opt:
            for group in opt.param_groups:
                if group.get("_is_B_group", False): group["lr"] = self.base_lr

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        if not self.pivot_done or self.warmup_remaining <= 0: return
        opt = optimizer or self._optimizer
        if not opt: return
        frac = (self.warmup_steps - self.warmup_remaining + 1) / self.warmup_steps
        lr = self.base_lr + frac * (self.lr_ratio - 1) * self.base_lr
        for group in opt.param_groups:
            if group.get("_is_B_group", False): group["lr"] = lr
        self.warmup_remaining -= 1


# ══════════════════════════════════════════════════════════════
# ROUGE evaluation
# ══════════════════════════════════════════════════════════════

def evaluate_rouge(model, tokenizer, num_samples=500, batch_size=16):
    from rouge_score import rouge_scorer as rs_lib
    scorer = rs_lib.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    model.eval()
    dataset_raw = load_dataset("abisee/cnn_dailymail", "3.0.0", split="validation")
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(dataset_raw),
                      size=min(num_samples, len(dataset_raw)), replace=False)
    examples = [dataset_raw[int(i)] for i in idxs]

    r1, r2, rL = [], [], []
    for start in range(0, len(examples), batch_size):
        batch_ex = examples[start:start + batch_size]
        prompts, refs = [], []
        for ex in batch_ex:
            doc_toks = tokenizer.encode(
                ex["article"],
                max_length=MAX_INPUT_LEN - MAX_TARGET_LEN - 20,
                truncation=True)
            prompts.append(
                tokenizer.decode(doc_toks, skip_special_tokens=True)
                + SUMMARY_MARKER)
            refs.append(ex["highlights"])

        inputs = tokenizer(
            prompts, return_tensors="pt", truncation=True,
            max_length=MAX_INPUT_LEN - MAX_TARGET_LEN,
            padding=True).to(DEVICE)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_TARGET_LEN,
                do_sample=False, num_beams=1,
                pad_token_id=tokenizer.eos_token_id)

        prompt_len = inputs["input_ids"].shape[1]
        for gen_ids, ref in zip(out, refs):
            gen_text = tokenizer.decode(
                gen_ids[prompt_len:], skip_special_tokens=True).strip()
            s = scorer.score(ref, gen_text)
            r1.append(s['rouge1'].fmeasure)
            r2.append(s['rouge2'].fmeasure)
            rL.append(s['rougeL'].fmeasure)

    return {'rouge1': float(np.mean(r1)),
            'rouge2': float(np.mean(r2)),
            'rougeL': float(np.mean(rL))}


# ══════════════════════════════════════════════════════════════
# Experiment runner
# ══════════════════════════════════════════════════════════════

def run_experiment(shot, seed, method):
    print(f"\n{'='*80}")
    print(f"Task=CNN/DM | Shot={shot} | Seed={seed} | Method={method}")
    print(f"{'='*80}")

    train_ds, val_ds, tokenizer = load_cnndm_splits(shot, seed)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.config.pad_token_id = tokenizer.pad_token_id
    cbs = []

    if method == "lora":
        inject_gpt2(model, RANK)
        optimizer = build_uniform_optimizer(model)

    elif method == "pivotlora":
        subspaces, eig_dict = get_gpt2_subspaces(model, train_ds, RANK, seed)
        inject_gpt2(model, RANK, block_subspaces=subspaces,
                     eigenvalues_dict=eig_dict, track_A_init=True)
        optimizer = build_eigen_optimizer(model)
        pivot_cb = GPT2PivotCallback(PIVOT_EPOCH, PIVOT_K_FRACTION,
                                       PIVOT_WARMUP_STEPS, LR_RATIO, LEARNING_RATE)
        pivot_cb._optimizer = optimizer
        cbs.append(pivot_cb)

    else:
        raise ValueError(f"Unknown method: {method}")

    _ta_kwargs = dict(
        output_dir=f"tmp/cnndm_{method}_{seed}",
        learning_rate=LEARNING_RATE, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        save_strategy="no", logging_steps=10,
        report_to="none", seed=seed,
        dataloader_num_workers=0,
        fp16=True, tf32=False, disable_tqdm=True)
    try: training_args = TrainingArguments(eval_strategy="no", **_ta_kwargs)
    except TypeError: training_args = TrainingArguments(evaluation_strategy="no", **_ta_kwargs)

    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_ds, callbacks=cbs,
                      optimizers=(optimizer, None))

    t0 = time.perf_counter()
    trainer.train()
    wall = time.perf_counter() - t0

    print(f"[EVAL] ROUGE on 500 val samples...")
    rouge = evaluate_rouge(model, tokenizer, num_samples=500, batch_size=16)
    print(f"  [RESULT] R1={rouge['rouge1']:.4f} R2={rouge['rouge2']:.4f} "
          f"RL={rouge['rougeL']:.4f} | {wall:.1f}s")
    return rouge, wall


def run_study(methods=None, shots=None, seeds=None):
    methods = methods or METHODS
    shots = shots or SHOTS
    seeds = seeds or SEED_LIST

    completed = {}
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                key = ("cnndm", int(row["shot"]), row["method"], int(row["seed"]))
                try:
                    completed[key] = {k: float(row[k])
                                      for k in ("rouge1", "rouge2", "rougeL")}
                except (ValueError, KeyError):
                    pass
        print(f"[RESUME] {len(completed)} completed")

    need_header = not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0
    fh = open(RESULTS_CSV, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if need_header: writer.writeheader()

    all_results = {}
    for shot in shots:
        for method in methods:
            scores_list = []
            for seed in seeds:
                key = ("cnndm", shot, method, seed)
                if key in completed:
                    scores_list.append(completed[key])
                    print(f"  [SKIP] cnndm-{shot}-{method}-s{seed}: "
                          f"RL={completed[key]['rougeL']:.4f}")
                    continue
                try:
                    rouge, wall = run_experiment(shot, seed, method)
                    scores_list.append(rouge)
                    writer.writerow({"task": "cnndm", "shot": shot,
                                     "method": method, "seed": seed,
                                     **{k: f"{v:.6f}" for k, v in rouge.items()}})
                    fh.flush()
                except Exception as e:
                    print(f"  CRASH: {e}")
                    import traceback; traceback.print_exc()

            if scores_list:
                rL = np.mean([s['rougeL'] for s in scores_list])
                all_results[(shot, method)] = scores_list
                print(f">>> cnndm {shot}-shot {method}: RL={rL:.4f}")
    fh.close()

    # Summary
    print(f"\n{'='*70}")
    print("CNN/DailyMail RESULTS (GPT-2 | ROUGE-L)")
    print(f"{'='*70}")
    print(f"  {'Method':<18}", end="")
    for s in shots: print(f"{'%d-shot':>14}" % s, end="")
    print()
    for method in methods:
        print(f"  {method:<18}", end="")
        for shot in shots:
            if (shot, method) in all_results:
                sl = all_results[(shot, method)]
                rL = np.mean([s['rougeL'] for s in sl])
                print(f"  {rL:.4f}      ", end="")
            else:
                print(f"  {'N/A':>12}", end="")
        print()
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--shots", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    if args.smoke:
        EPOCHS = 2
        run_study(methods=["lora", "pivotlora"], shots=[64], seeds=[42])
    else:
        seeds = SEED_LIST[:args.seeds] if args.seeds else SEED_LIST
        run_study(methods=args.methods, shots=args.shots, seeds=seeds)