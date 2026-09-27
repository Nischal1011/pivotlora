import os, sys, csv, time, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, TrainerCallback, set_seed,
)
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import hashlib

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = 1
print(f"Running on: {DEVICE}")
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"GPUs available: {n_gpus}")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")

# Decoder backbone. Both reported backbones run this same script with
# identical settings -- only the checkpoint changes -- so `--model` selects it
# at the command line. The aliases below expand to the checkpoints used in the
# paper; any other Hugging Face id may be passed verbatim.
MODEL_ALIASES = {
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
}
DEFAULT_MODEL = MODEL_ALIASES["llama3.1-8b"]

MODEL_NAME = DEFAULT_MODEL
RANK = 8
ALPHA = 16
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
EPOCHS = 15
MAX_SEQ_LEN = {"sst2": 128, "agnews": 128, "boolq": 192}

LR_RATIO = 16
PIVOT_EPOCH = 3
PIVOT_K_FRACTION = 0.5
PIVOT_WARMUP_STEPS = 10

SEED_LIST = [42, 43, 44]
SHOTS = [64, 256]
METHODS = ["lora", "pivotlora", "pissa", "loram"]

TASK_INFO = {
    "sst2": {"dataset": ("nyu-mll/glue", "sst2"), "text_fields": ["sentence"],
             "num_labels": 2, "metric": "accuracy"},
    "agnews": {"dataset": ("fancyzhx/ag_news",), "text_fields": ["text"],
               "num_labels": 4, "metric": "accuracy"},
    "boolq": {"dataset": ("aps/super_glue", "boolq"), "text_fields": ["question", "passage"],
              "num_labels": 2, "metric": "accuracy"},
}

CSV_FIELDS = ["task", "shot", "method", "seed", "score"]
TARGET_MODULES = ["q_proj", "v_proj"]


def results_csv_for(model_name):
    """One results file per backbone.

    run_study() resumes by reading the existing CSV and skipping
    (task, shot, method, seed) rows it already contains. That key does not
    include the model, so writing two backbones to one file would make the
    second run skip every cell the first had finished. Deriving the filename
    from the checkpoint keeps the two studies separate.
    """
    slug = model_name.rstrip("/").split("/")[-1].replace(".", "").lower()
    return f"results_decoder_{slug}.csv"


RESULTS_CSV = results_csv_for(MODEL_NAME)


class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank, alpha,
                 init_A=None, init_B=None, track_A_init=False,
                 b_lr_scales=None):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.scaling = alpha / rank
        d_in = original_layer.in_features
        d_out = original_layer.out_features
        device = original_layer.weight.device

        if init_A is not None:
            scale = np.sqrt(5.0) / np.sqrt(d_in)
            self.lora_A = nn.Parameter(
                init_A.to(device=device, dtype=torch.float32) * scale)
        else:
            self.lora_A = nn.Parameter(torch.empty(d_in, rank, device=device))
            nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))

        self._split_b = (b_lr_scales is not None)
        if self._split_b:
            if init_B is not None:
                rows = [nn.Parameter(init_B[ri:ri+1, :].to(device=device, dtype=torch.float32))
                        for ri in range(rank)]
            else:
                rows = [nn.Parameter(torch.zeros(1, d_out, device=device))
                        for _ in range(rank)]
            self.lora_B_rows = nn.ParameterList(rows)
            self.lora_B = None
            self.register_buffer("b_lr_scales",
                torch.tensor(b_lr_scales, dtype=torch.float32, device=device))
        else:
            if init_B is not None:
                self.lora_B = nn.Parameter(
                    init_B.to(device=device, dtype=torch.float32).clone())
            else:
                self.lora_B = nn.Parameter(torch.zeros(rank, d_out, device=device))
            self.lora_B_rows = None
            self.b_lr_scales = None

        if track_A_init:
            self.register_buffer("A_init_snapshot", self.lora_A.data.clone().detach())
        else:
            self.A_init_snapshot = None

        original_layer.weight.requires_grad = False
        if original_layer.bias is not None:
            original_layer.bias.requires_grad = False

    @property
    def lora_B_matrix(self):
        if self._split_b:
            return torch.cat(list(self.lora_B_rows), dim=0)
        return self.lora_B

    def forward(self, x):
        return self.original_layer(x) + (x @ self.lora_A @ self.lora_B_matrix) * self.scaling


def load_data(task_name, n_shot, seed):
    set_seed(seed)
    np.random.seed(seed)
    info = TASK_INFO[task_name]
    dataset = load_dataset(*info["dataset"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    seq_len = MAX_SEQ_LEN.get(task_name, 128)

    def tokenize(examples):
        fields = info["text_fields"]
        texts = examples[fields[0]] if len(fields) == 1 else \
                [f"{a} {b}" for a, b in zip(examples[fields[0]], examples[fields[1]])]
        return tokenizer(texts, truncation=True, max_length=seq_len, padding="max_length")

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_full = dataset["train"]
    val_ds = dataset.get("validation") or dataset.get("validation_matched") or dataset.get("test")

    rng = np.random.default_rng(seed)
    labels_arr = np.array(train_full["labels"])
    train_indices = []
    for label in range(info["num_labels"]):
        lbl_idx = np.where(labels_arr == label)[0]
        rng.shuffle(lbl_idx)
        train_indices.extend(lbl_idx[:n_shot // info["num_labels"]].tolist())
    rng.shuffle(train_indices)
    train_ds = train_full.select(train_indices)
    print(f"[DATA] {task_name} {n_shot}-shot: Train={len(train_ds)} Val={len(val_ds)}")
    return train_ds, val_ds, info, tokenizer


def compute_pca_eigen(X_np, rank):
    N, D = X_np.shape
    n_comp = min(rank, N - 1, D)
    pca = PCA(n_components=n_comp)
    pca.fit(X_np)
    Q = pca.components_.T
    if n_comp < rank:
        rng = np.random.default_rng(42)
        Q_orth, _ = np.linalg.qr(Q)
        for _ in range(rank - n_comp):
            v = rng.standard_normal(D)
            for j in range(Q_orth.shape[1]):
                v -= np.dot(v, Q_orth[:, j]) * Q_orth[:, j]
            v /= np.linalg.norm(v) + 1e-12
            Q_orth = np.concatenate([Q_orth, v[:, None]], axis=1)
        Q = Q_orth[:, :rank]
        eigenvalues = np.concatenate([
            pca.explained_variance_,
            np.full(rank - n_comp, pca.explained_variance_.min() * 0.1 + 1e-8)])
    else:
        Q, _ = np.linalg.qr(Q)
        Q = Q[:, :rank]
        eigenvalues = pca.explained_variance_[:rank]
    return torch.tensor(Q, dtype=torch.float32), eigenvalues


def compute_pissa_init(weight, rank):
    W = weight.detach().float().cpu()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return Vh[:rank, :].T, (U[:, :rank] * S[:rank].unsqueeze(0)).T


def compute_loram_init(weight, rank):
    W = weight.detach().float().cpu()
    d_out, d_in = W.shape
    v = torch.randn(d_in); v = v / v.norm()
    for _ in range(10):
        u = W @ v; u = u / u.norm().clamp(min=1e-8)
        v = W.T @ u; v = v / v.norm().clamp(min=1e-8)
    sigma_max = abs((u @ W @ v).item())
    rng = np.random.default_rng(42)
    Q, _ = np.linalg.qr(rng.standard_normal((d_in, rank)).astype(np.float32))
    return torch.tensor(Q[:, :rank] * np.log(1.0 + sigma_max), dtype=torch.float32), None


def get_llama_layers(model):
    return model.model.layers


def get_input_device(model):
    return next(get_llama_layers(model)[0].parameters()).device


def get_subspaces(model, dataset, rank, seed):
    print(f"[SUBSPACE] Computing mean-pooled pca_eigen...")
    model.eval()
    layers = get_llama_layers(model)
    acts_q = defaultdict(list)
    acts_v = defaultdict(list)

    def make_hook(store, idx):
        def hook(module, inputs):
            x = inputs[0].detach()
            store[idx].append(x.mean(dim=1).cpu())
        return hook

    handles = []
    for i, layer in enumerate(layers):
        handles.append(layer.self_attn.q_proj.register_forward_pre_hook(make_hook(acts_q, i)))
        handles.append(layer.self_attn.v_proj.register_forward_pre_hook(make_hook(acts_v, i)))

    input_device = get_input_device(model)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            model(**{k: v.to(input_device) for k, v in batch.items() if k != "labels"})
    for h in handles:
        h.remove()

    subspaces = defaultdict(dict)
    eig_dict = defaultdict(dict)
    for target_name, acts_store in [("q_proj", acts_q), ("v_proj", acts_v)]:
        for idx, act_list in acts_store.items():
            X = torch.cat(act_list, dim=0).float().numpy()
            Q, eigs = compute_pca_eigen(X, rank)
            subspaces[idx][target_name] = Q
            eig_dict[idx][target_name] = eigs
    print(f"[SUBSPACE] Done.")
    return dict(subspaces), dict(eig_dict)


def inject_llama(model, rank, method, block_subspaces=None,
                 eigenvalues_dict=None, track_A_init=False,
                 b_inits=None, subtract_residual=False):
    layers = get_llama_layers(model)
    count = 0
    for i, layer in enumerate(layers):
        for target_name in TARGET_MODULES:
            original = getattr(layer.self_attn, target_name)
            layer_device = original.weight.device
            curr_A, curr_B = None, None

            if block_subspaces and i in block_subspaces:
                layer_sub = block_subspaces[i]
                curr_A = layer_sub.get(target_name) if isinstance(layer_sub, dict) else layer_sub

            if b_inits and i in b_inits and target_name in b_inits[i]:
                curr_A, curr_B = b_inits[i][target_name]

            if curr_A is not None and curr_B is None:
                rot_seed = f"layer_{i}_{target_name}"
                seed_int = int.from_bytes(hashlib.sha256(rot_seed.encode()).digest()[:4], "little")
                g = torch.Generator(device=layer_device); g.manual_seed(seed_int)
                rot, _ = torch.linalg.qr(torch.randn(rank, rank, generator=g, device=layer_device))
                curr_A = curr_A.to(layer_device) @ rot

            b_scales = None
            if eigenvalues_dict and i in eigenvalues_dict:
                layer_eigs = eigenvalues_dict[i]
                eigs = layer_eigs.get(target_name) if isinstance(layer_eigs, dict) else layer_eigs
                if eigs is not None:
                    b_scales = (eigs / (float(eigs.mean()) + 1e-8)).astype(np.float32)

            adapter = LoRALayer(original, rank, ALPHA, init_A=curr_A, init_B=curr_B,
                                track_A_init=track_A_init, b_lr_scales=b_scales)

            if subtract_residual and curr_A is not None and curr_B is not None:
                with torch.no_grad():
                    delta_W = (adapter.lora_A.data @ adapter.lora_B_matrix).T * adapter.scaling
                    original.weight.data -= delta_W

            setattr(layer.self_attn, target_name, adapter)
            count += 1

    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALayer):
            m.lora_A.requires_grad = True
            if m._split_b:
                for row in m.lora_B_rows: row.requires_grad = True
            else:
                m.lora_B.requires_grad = True
    if hasattr(model, "score"):
        for p in model.score.parameters(): p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INJECT] {count} adapters | trainable: {trainable:,}")
    return model


class PivotCallback(TrainerCallback):
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
            if not isinstance(m, LoRALayer) or m.A_init_snapshot is None: continue
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


def build_optimizer(model, use_eigen=False):
    if use_eigen:
        A_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_A" in n]
        head_params = [p for n, p in model.named_parameters() if p.requires_grad and "score" in n]
        param_groups = [{"params": A_params, "lr": LEARNING_RATE, "weight_decay": 0.01}]
        for _, m in model.named_modules():
            if not isinstance(m, LoRALayer): continue
            if m._split_b:
                for ri, row in enumerate(m.lora_B_rows):
                    param_groups.append({"params": [row],
                        "lr": m.b_lr_scales[ri].item() * LR_RATIO * LEARNING_RATE,
                        "weight_decay": 0.01, "_is_B_group": True})
            else:
                param_groups.append({"params": [m.lora_B],
                    "lr": LR_RATIO * LEARNING_RATE, "weight_decay": 0.01, "_is_B_group": True})
        if head_params:
            param_groups.append({"params": head_params, "lr": LEARNING_RATE, "weight_decay": 0.01})
        return torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    else:
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)


def run_experiment(task, shot, seed, method):
    print(f"\n{'='*80}")
    print(f"Task={task.upper()} | Shot={shot} | Seed={seed} | Method={method}")
    print(f"{'='*80}")

    train_ds, val_ds, info, tokenizer = load_data(task, shot, seed)

    if NUM_GPUS >= 2:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=info["num_labels"],
            torch_dtype=torch.bfloat16, device_map="auto")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=info["num_labels"],
            torch_dtype=torch.bfloat16).to(DEVICE)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    cbs = []

    if method == "lora":
        inject_llama(model, RANK, method)
        optimizer = build_optimizer(model, use_eigen=False)

    elif method == "pivotlora":
        subspaces, eig_dict = get_subspaces(model, train_ds, RANK, seed)
        inject_llama(model, RANK, method, block_subspaces=subspaces,
                     eigenvalues_dict=eig_dict, track_A_init=True)
        optimizer = build_optimizer(model, use_eigen=True)
        pivot_cb = PivotCallback(PIVOT_EPOCH, PIVOT_K_FRACTION,
                                 PIVOT_WARMUP_STEPS, LR_RATIO, LEARNING_RATE)
        pivot_cb._optimizer = optimizer
        cbs.append(pivot_cb)

    elif method == "pissa":
        b_inits = {}
        for i, layer in enumerate(get_llama_layers(model)):
            b_inits[i] = {}
            for t in TARGET_MODULES:
                b_inits[i][t] = compute_pissa_init(getattr(layer.self_attn, t).weight, RANK)
        inject_llama(model, RANK, method, b_inits=b_inits, subtract_residual=True)
        optimizer = build_optimizer(model, use_eigen=False)

    elif method == "loram":
        b_inits = {}
        for i, layer in enumerate(get_llama_layers(model)):
            b_inits[i] = {}
            for t in TARGET_MODULES:
                b_inits[i][t] = compute_loram_init(getattr(layer.self_attn, t).weight, RANK)
        inject_llama(model, RANK, method, b_inits=b_inits)
        optimizer = build_optimizer(model, use_eigen=False)

    else:
        raise ValueError(f"Unknown method: {method}")

    _ta_kwargs = dict(
        output_dir=f"tmp/llama_{task}_{method}_{seed}",
        learning_rate=LEARNING_RATE, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        save_strategy="no", logging_steps=5, report_to="none",
        seed=seed, dataloader_num_workers=2,
        fp16=False, bf16=True, tf32=True, disable_tqdm=True)
    try: training_args = TrainingArguments(eval_strategy="epoch", **_ta_kwargs)
    except TypeError: training_args = TrainingArguments(evaluation_strategy="epoch", **_ta_kwargs)

    def compute_metrics(p):
        return {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))}

    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_ds, eval_dataset=val_ds,
                      compute_metrics=compute_metrics, callbacks=cbs,
                      optimizers=(optimizer, None))

    t0 = time.perf_counter()
    trainer.train()
    wall = time.perf_counter() - t0

    best = max((log.get("eval_accuracy", 0) for log in trainer.state.log_history), default=0)
    print(f"  [RESULT] {method}: {best:.4f} | {wall:.1f}s")
    return best, wall


def run_study(methods=None, shots=None, seeds=None, tasks=None):
    methods = methods or METHODS; shots = shots or SHOTS
    seeds = seeds or SEED_LIST; tasks = tasks or ["sst2", "agnews"]

    completed = {}
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                key = (row["task"], int(row["shot"]), row["method"], int(row["seed"]))
                try: completed[key] = float(row["score"])
                except ValueError: pass
        print(f"[RESUME] {len(completed)} completed rows")

    need_header = not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0
    csv_fh = open(RESULTS_CSV, "a", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
    if need_header: writer.writeheader()
    all_results = {}

    for task in tasks:
        for shot in shots:
            for method in methods:
                scores = []
                for seed in seeds:
                    key = (task, shot, method, seed)
                    if key in completed:
                        scores.append(completed[key])
                        print(f"  [SKIP] {task}-{shot}-{method}-s{seed}: {completed[key]:.4f}")
                        continue
                    try:
                        score, wall = run_experiment(task, shot, seed, method)
                        scores.append(score)
                        writer.writerow({"task": task, "shot": shot,
                                         "method": method, "seed": seed,
                                         "score": f"{score:.6f}"})
                        csv_fh.flush()
                    except Exception as e:
                        print(f"  CRASH: {e}")
                        import traceback; traceback.print_exc()
                if scores:
                    mu, std = np.mean(scores), np.std(scores)
                    all_results[(task, shot, method)] = (mu, std)
                    print(f">>> {task} {shot}-shot {method}: {mu:.4f}+/-{std:.4f}")
    csv_fh.close()

    print(f"\n{'='*80}")
    for task in tasks:
        print(f"\n  {task.upper()}:")
        for method in methods:
            for shot in shots:
                if (task, shot, method) in all_results:
                    mu, std = all_results[(task, shot, method)]
                    print(f"    {method:<15} {shot}-shot: {mu:.4f}+/-{std:.4f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decoder-scale few-shot classification: LoRA, PiSSA, LoRAM, "
                    "and PivotLoRA. The same protocol is used for every backbone; "
                    "only --model changes between the reported studies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--shots", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Backbone checkpoint. Accepts the aliases "
             + ", ".join(f"{k} ({v})" for k, v in MODEL_ALIASES.items())
             + ", or any Hugging Face model id.",
    )
    parser.add_argument(
        "--results-csv", default=None,
        help="Override the results file. Defaults to one file per backbone, "
             "which is what keeps the resume logic from mixing two studies.",
    )
    args = parser.parse_args()

    # Rebind the module-level globals the training functions read.
    MODEL_NAME = MODEL_ALIASES.get(args.model, args.model)
    RESULTS_CSV = args.results_csv or results_csv_for(MODEL_NAME)

    print(f"Backbone     : {MODEL_NAME}")
    print(f"Results file : {RESULTS_CSV}")

    NUM_GPUS = args.gpus
    if args.smoke:
        EPOCHS = 2
        run_study(methods=["lora", "pivotlora"], shots=[64], seeds=[42], tasks=["sst2"])
    else:
        seeds = SEED_LIST[:args.seeds] if args.seeds else SEED_LIST
        run_study(methods=args.methods, shots=args.shots, seeds=seeds, tasks=args.tasks)