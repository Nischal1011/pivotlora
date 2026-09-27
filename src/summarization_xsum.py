"""
xsum_experiment.py
==================
GPT-2 + XSum abstractive summarization.
Reproduces Table 7 (generation negative control) in the paper.

The PivotLoRA algorithm is IDENTICAL to the BERT version.
Only the LoRA layer differs: Conv1DLoRALayer wraps GPT-2's Conv1D
instead of BERT's nn.Linear. Everything else (PCA init, pivot,
eigenvalue-scaled LR) is the same.

Methods: LoRA, EigenLoRA, PivotLoRA, PiSSA, EVA.
Shots: 64, 128, 256, 512.
Seeds: 42, 43, 44.
Metric: ROUGE-1/2/L on 500 validation samples.

Requires: train_classification.py in the same directory (for LoRALayer, PivotCallback imports).
Time: ~2 hours on single T4/5090.

Usage:
    python xsum_experiment.py                                  # full
    python xsum_experiment.py --methods lora pivotlora          # subset
    python xsum_experiment.py --shots 64 256                    # subset
    python xsum_experiment.py --smoke                           # quick test
"""

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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bert')))
from train_classification import (
    LoRALayer,
    PivotCallback,
    compute_pca_eigen,
    compute_pissa_init,
)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

# ── Config: IDENTICAL to BERT except where noted ─────────────────────────────
MODEL_NAME       = "gpt2"       # 124M, comparable to BERT-base (110M)
RANK             = 8            # same as BERT
ALPHA            = 16           # same as BERT
BATCH_SIZE       = 4            # smaller due to longer sequences
LEARNING_RATE    = 2e-4         # same as BERT
EPOCHS           = 15           # same as BERT
MAX_INPUT_LEN    = 512
MAX_TARGET_LEN   = 64
SUMMARY_MARKER   = "\n\nTL;DR: "

# PivotLoRA hyperparameters: IDENTICAL to BERT
LR_RATIO           = 16         # same as BERT (not reduced for generation)
PIVOT_EPOCH   = 3          # same as BERT
PIVOT_K_FRACTION    = 0.5        # same as BERT
PIVOT_WARMUP_STEPS = 10         # same as BERT
# Optimizer reset: A only, not B (same as BERT, validated by reset ablation)

SEED_LIST  = [42, 43, 44]
SHOTS      = [64, 128, 256, 512]
METHODS    = ["lora", "eigenlora", "pivotlora", "pissa", "eva"]

RESULTS_CSV = "results_xsum.csv"
CSV_FIELDS  = ["task", "shot", "method", "seed", "rouge1", "rouge2", "rougeL"]


# ══════════════════════════════════════════════════════════════════════════════
# Conv1D-safe LoRA layer
# ══════════════════════════════════════════════════════════════════════════════

class Conv1DLoRALayer(nn.Module):
    """
    LoRA adapter for GPT-2's Conv1D layers.
    Keeps original Conv1D intact. LoRA is a parallel low-rank path.

    Conv1D:  y = x @ W + b     where W is (d_in, d_out)
    LoRA:    y += (x @ A @ B) * scaling
    A: (d_in, rank), B: (rank, d_out) -- same layout as LoRALayer.
    """
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

        # lora_A: (d_in, rank)
        if init_A is not None:
            scale = np.sqrt(5.0) / np.sqrt(d_in)
            self.lora_A = nn.Parameter(
                init_A.to(device=device, dtype=torch.float32) * scale)
        else:
            self.lora_A = nn.Parameter(torch.empty(d_in, rank, device=device))
            nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))

        # lora_B: per-rank split (eigenvalue-scaled LR) or unified
        self._split_b = (b_lr_scales is not None)
        if self._split_b:
            rows = []
            for ri in range(rank):
                if init_B is not None:
                    rows.append(nn.Parameter(
                        init_B[ri:ri+1, :].to(device=device, dtype=torch.float32)))
                else:
                    rows.append(nn.Parameter(torch.zeros(1, d_out, device=device)))
            self.lora_B_rows = nn.ParameterList(rows)
            self.lora_B      = None
            self.register_buffer(
                "b_lr_scales",
                torch.tensor(b_lr_scales, dtype=torch.float32, device=device))
        else:
            if init_B is not None:
                self.lora_B = nn.Parameter(
                    init_B.to(device=device, dtype=torch.float32).clone())
            else:
                self.lora_B = nn.Parameter(torch.zeros(rank, d_out, device=device))
            self.lora_B_rows = None
            self.b_lr_scales = None

        # PivotLoRA snapshot (same as BERT)
        if track_A_init:
            self.register_buffer("A_init_snapshot",
                                 self.lora_A.data.clone().detach())
        else:
            self.A_init_snapshot = None

        # Freeze original
        original_conv1d.weight.requires_grad = False
        if original_conv1d.bias is not None:
            original_conv1d.bias.requires_grad = False

    @property
    def lora_B_matrix(self):
        if self._split_b:
            return torch.cat(list(self.lora_B_rows), dim=0)
        return self.lora_B

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A @ self.lora_B_matrix) * self.scaling


# ══════════════════════════════════════════════════════════════════════════════
# Data loading with summary-only label masking
# ══════════════════════════════════════════════════════════════════════════════

def load_xsum_splits(n_shot, seed):
    """
    XSum with summary-only label masking.
    Format: "{document}\n\nTL;DR: {summary}<EOS>"
    Labels: -100 for document+marker, actual ids for summary+EOS only.
    """
    print(f"\n[DATA] Loading XSum n_shot={n_shot} seed={seed}...")
    set_seed(seed)
    np.random.seed(seed)

    dataset   = load_dataset("EdinburghNLP/xsum")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    marker_ids = tokenizer.encode(SUMMARY_MARKER, add_special_tokens=False)
    marker_len = len(marker_ids)

    def tokenize_fn(examples):
        all_input_ids, all_attention_mask, all_labels = [], [], []
        for doc, summary in zip(examples["document"], examples["summary"]):
            summary_ids = tokenizer.encode(
                summary + tokenizer.eos_token, add_special_tokens=False)
            doc_budget  = max(10, MAX_INPUT_LEN - marker_len - len(summary_ids))
            doc_ids     = tokenizer.encode(doc, add_special_tokens=False,
                                           max_length=doc_budget, truncation=True)
            input_ids   = (doc_ids + marker_ids + summary_ids)[:MAX_INPUT_LEN]

            # Mask document + marker, keep summary + EOS
            n_masked = len(doc_ids) + marker_len
            labels   = ([-100] * min(n_masked, len(input_ids))
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
    val_ds = dataset.get("validation") or dataset.get("test") or train_full

    rng  = np.random.default_rng(seed)
    idxs = np.arange(len(train_full))
    rng.shuffle(idxs)
    train_ds = train_full.select(idxs[:n_shot].tolist())

    # Sanity check
    sample_labels = train_ds[0]["labels"]
    n_active = (sample_labels != -100).sum().item()
    n_nonpad = (train_ds[0]["attention_mask"] == 1).sum().item()
    print(f"[DATA] Train={len(train_ds)} Val={len(val_ds)} | "
          f"label check: {n_active}/{n_nonpad} active "
          f"({100*n_active/max(n_nonpad,1):.0f}% = summary only)")

    return train_ds, val_ds, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Per-target subspace computation (FIX: hook mismatch)
# ══════════════════════════════════════════════════════════════════════════════

def get_gpt2_subspaces(model, dataset, rank, seed):
    """
    Separate PCA per adapter target.
    c_proj hook  -> init for c_proj adapters
    mlp.c_fc hook -> init for mlp.c_fc adapters

    This avoids the hook mismatch where both adapters would get the same
    PCA from c_proj activations, even though mlp.c_fc sees a completely
    different activation distribution (post-LayerNorm, not attention output).
    """
    print(f"[SUBSPACE-GPT2] Computing per-target pca_eigen...")
    model.eval().to(DEVICE)

    acts_cproj = defaultdict(list)
    acts_mlpfc = defaultdict(list)

    def make_hook(store, idx):
        def hook(module, inputs):
            store[idx].append(inputs[0].detach().mean(dim=1).cpu())
        return hook

    handles = []
    for i, block in enumerate(model.transformer.h):
        handles.append(
            block.attn.c_proj.register_forward_pre_hook(
                make_hook(acts_cproj, i)))
        handles.append(
            block.mlp.c_fc.register_forward_pre_hook(
                make_hook(acts_mlpfc, i)))

    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            model(**{k: v.to(DEVICE) for k, v in batch.items() if k != "labels"})
    for h in handles:
        h.remove()

    # Returns nested dicts: subspaces[layer_idx][target_name] = Q
    subspaces = defaultdict(dict)
    eig_dict  = defaultdict(dict)

    for idx in acts_cproj:
        X = torch.cat(acts_cproj[idx], dim=0).numpy()
        Q, eigs = compute_pca_eigen(X, rank)
        subspaces[idx]["c_proj"] = Q.to(DEVICE)
        eig_dict[idx]["c_proj"]  = eigs
        print(f"  [c_proj]  Layer {idx:>2}: eig {eigs.min():.2f}..{eigs.max():.2f} "
              f"max/mean={eigs.max()/eigs.mean():.2f}x")

    for idx in acts_mlpfc:
        X = torch.cat(acts_mlpfc[idx], dim=0).numpy()
        Q, eigs = compute_pca_eigen(X, rank)
        subspaces[idx]["c_fc"] = Q.to(DEVICE)
        eig_dict[idx]["c_fc"]  = eigs
        print(f"  [mlp.c_fc] Layer {idx:>2}: eig {eigs.min():.2f}..{eigs.max():.2f} "
              f"max/mean={eigs.max()/eigs.mean():.2f}x")

    print(f"[SUBSPACE-GPT2] Done. {len(subspaces)} layers x 2 targets.")
    return dict(subspaces), dict(eig_dict)


def compute_gpt2_pissa(model, rank):
    """
    PiSSA init for Conv1D layers.
    Conv1D weight is (d_in, d_out). Do NOT transpose before SVD.
    compute_pissa_init expects (d_out, d_in), so we transpose here.
    Returns A_init (d_in, rank) and B_init (rank, d_out).
    """
    b_inits = {}
    for i, block in enumerate(model.transformer.h):
        b_inits[i] = {}
        for parent_name, target_name in [("attn", "c_proj"), ("mlp", "c_fc")]:
            original = getattr(getattr(block, parent_name), target_name)
            # compute_pissa_init expects W as (d_out, d_in) like nn.Linear
            # Conv1D weight is (d_in, d_out), so transpose
            W = original.weight.T.contiguous()  # (d_out, d_in)
            A_init, B_init = compute_pissa_init(W, rank)
            # A_init: (d_in, rank), B_init: (rank, d_out) -- correct for our layout
            b_inits[i][target_name] = (A_init, B_init)
    return b_inits


def compute_gpt2_eva(model, dataset, rank):
    """EVA: all-token PCA of c_proj pre-activations (uniform LR, no eigenvalues)."""
    print("[EVA-GPT2] Computing all-token activation SVD...")
    model.eval().to(DEVICE)
    activations = defaultdict(list)

    def make_hook(idx):
        def hook(module, inputs):
            x = inputs[0].detach()
            B, T, D = x.shape
            activations[idx].append(x.reshape(-1, D).cpu())
        return hook

    handles = [
        block.attn.c_proj.register_forward_pre_hook(make_hook(i))
        for i, block in enumerate(model.transformer.h)
    ]
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            model(**{k: v.to(DEVICE) for k, v in batch.items() if k != "labels"})
    for h in handles:
        h.remove()

    subspaces = {}
    for idx, acts in activations.items():
        X      = torch.cat(acts, dim=0).numpy()
        n_comp = min(rank, X.shape[0] - 1, X.shape[1])
        pca    = PCA(n_components=n_comp).fit(X)
        Q      = pca.components_.T
        if n_comp < rank:
            rng = np.random.default_rng(42 + idx)
            Q, _ = np.linalg.qr(Q)
            for _ in range(rank - n_comp):
                v = rng.standard_normal(X.shape[1])
                for j in range(Q.shape[1]):
                    v -= np.dot(v, Q[:, j]) * Q[:, j]
                v /= np.linalg.norm(v) + 1e-12
                Q = np.concatenate([Q, v[:, None]], axis=1)
        else:
            Q, _ = np.linalg.qr(Q)
        Q = Q[:, :rank]
        subspaces[idx] = torch.tensor(Q, dtype=torch.float32).to(DEVICE)
        expl = pca.explained_variance_ratio_[:n_comp].sum() * 100
        print(f"  [EVA] Layer {idx:>2}: explained_var={expl:.1f}%")

    print("[EVA-GPT2] Done.")
    return subspaces


# ══════════════════════════════════════════════════════════════════════════════
# Injection (handles per-target subspaces)
# ══════════════════════════════════════════════════════════════════════════════

def inject_gpt2(model, rank, block_subspaces=None,
                eigenvalues_dict=None, track_A_init=False,
                b_inits=None, subtract_residual=False):
    """
    Inject Conv1DLoRALayer into attn.c_proj and mlp.c_fc.

    block_subspaces can be:
      - dict {layer_idx: tensor} (flat, EVA style -- same init for both targets)
      - dict {layer_idx: {target_name: tensor}} (per-target, PCA style)

    lm_head is NOT unfrozen (tied to wte, 38.6M params).
    """
    print(f"[INJECT-GPT2] rank={rank} alpha={ALPHA} lr_ratio={LR_RATIO}")
    target_attrs = [("attn", "c_proj"), ("mlp", "c_fc")]
    count = 0

    for i, block in enumerate(model.transformer.h):
        for parent_name, target_name in target_attrs:
            parent   = getattr(block, parent_name)
            original = getattr(parent, target_name)

            # Look up subspace: per-target dict or flat
            curr_A = None
            if block_subspaces and i in block_subspaces:
                layer_sub = block_subspaces[i]
                if isinstance(layer_sub, dict):
                    curr_A = layer_sub.get(target_name)
                else:
                    curr_A = layer_sub  # flat (EVA)

            curr_B = None
            if b_inits and i in b_inits and target_name in b_inits[i]:
                curr_A, curr_B = b_inits[i][target_name]

            # Random rotation for direction-only inits (same as BERT)
            if curr_A is not None and curr_B is None:
                rot_seed = f"layer_{i}_{target_name}"
                seed_int = int.from_bytes(
                    hashlib.sha256(rot_seed.encode()).digest()[:4], "little")
                g = torch.Generator(device=DEVICE)
                g.manual_seed(seed_int)
                rot, _ = torch.linalg.qr(
                    torch.randn(rank, rank, generator=g, device=DEVICE))
                curr_A = curr_A @ rot

            # Eigenvalue-scaled B LR (same formula as BERT, no clipping)
            b_scales = None
            if eigenvalues_dict and i in eigenvalues_dict:
                layer_eigs = eigenvalues_dict[i]
                if isinstance(layer_eigs, dict):
                    eigs = layer_eigs.get(target_name)
                else:
                    eigs = layer_eigs
                if eigs is not None:
                    sigma_mean = float(eigs.mean()) + 1e-8
                    b_scales = (eigs / sigma_mean).astype(np.float32)

            if isinstance(original, Conv1D):
                adapter = Conv1DLoRALayer(original, rank, ALPHA,
                                          init_A=curr_A, init_B=curr_B,
                                          track_A_init=track_A_init,
                                          b_lr_scales=b_scales)
                # PiSSA residual: Conv1D weight is (d_in, d_out), delta = A@B = (d_in, d_out)
                if subtract_residual and curr_A is not None and curr_B is not None:
                    with torch.no_grad():
                        delta = (adapter.lora_A.data @ adapter.lora_B_matrix) * adapter.scaling
                        original.weight.data -= delta
                setattr(parent, target_name, adapter)
            else:
                adapter = LoRALayer(original, rank, ALPHA,
                                      init_A=curr_A, init_B=curr_B,
                                      track_A_init=track_A_init,
                                      b_lr_scales=b_scales)
                if subtract_residual and curr_A is not None and curr_B is not None:
                    with torch.no_grad():
                        delta = (adapter.lora_A.data @ adapter.lora_B_matrix).T * adapter.scaling
                        original.weight.data -= delta
                setattr(parent, target_name, adapter)
            count += 1

    # Freeze everything, unfreeze adapters only (NOT lm_head)
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, (Conv1DLoRALayer, LoRALayer)):
            m.lora_A.requires_grad = True
            if m._split_b:
                for row in m.lora_B_rows:
                    row.requires_grad = True
            else:
                m.lora_B.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lm_trainable = sum(p.numel() for p in model.lm_head.parameters() if p.requires_grad)
    print(f"[INJECT-GPT2] {count} adapters | trainable: {trainable:,} "
          f"(lm_head: {lm_trainable:,}, should be 0)")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Optimizer (same structure as BERT)
# ══════════════════════════════════════════════════════════════════════════════

def build_eigen_optimizer(model):
    """
    Eigenvalue-scaled per-rank B LR, same as BERT.
    A: base LR. B rows: eigenvalue_scale * LR_RATIO * base LR.
    No clipping, no generation-specific adjustments.
    """
    A_params = [p for n, p in model.named_parameters()
                if p.requires_grad and "lora_A" in n]

    param_groups = [
        {"params": A_params, "lr": LEARNING_RATE, "weight_decay": 0.01},
    ]

    for _, m in model.named_modules():
        if not isinstance(m, (Conv1DLoRALayer, LoRALayer)):
            continue
        if m._split_b:
            for rank_i, row in enumerate(m.lora_B_rows):
                scale_i = m.b_lr_scales[rank_i].item()
                param_groups.append({
                    "params": [row],
                    "lr": scale_i * LR_RATIO * LEARNING_RATE,
                    "weight_decay": 0.01,
                    "_is_B_group": True,
                })
        else:
            param_groups.append({
                "params": [m.lora_B],
                "lr": LR_RATIO * LEARNING_RATE,
                "weight_decay": 0.01,
                "_is_B_group": True,
            })

    n_b = sum(1 for g in param_groups if g.get("_is_B_group"))
    print(f"  [OPT] EigenLoRA: A lr={LEARNING_RATE:.2e} | "
          f"{n_b} B groups, lr_ratio={LR_RATIO}")
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)


def build_uniform_optimizer(model):
    """Uniform LR for random, pissa, eva."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  [OPT] Uniform: {sum(p.numel() for p in trainable):,} params "
          f"lr={LEARNING_RATE:.2e}")
    return torch.optim.AdamW(trainable, lr=LEARNING_RATE,
                              betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)


# ══════════════════════════════════════════════════════════════════════════════
# PivotLoRA callback for GPT-2 (same algorithm as BERT)
# ══════════════════════════════════════════════════════════════════════════════

class GPT2PivotCallback(PivotCallback):
    """
    Identical pivot algorithm to BERT's PivotCallback.
    Only difference: also recognizes Conv1DLoRALayer (not just LoRALayer).
    Reset behavior: A optimizer state reset, B kept (same as BERT).
    """
    def on_epoch_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        if self.pivot_done or state.epoch is None:
            return
        epoch_f = state.epoch
        if epoch_f + 1e-4 < self.pivot_epoch:
            return
        if epoch_f > self.pivot_epoch + 1.0:
            return

        opt = optimizer or self._optimizer
        print(f"\n[PivotLoRA-GPT2] PIVOT at epoch {epoch_f:.3f} "
              f"(target={self.pivot_epoch})")
        from train_classification import RANK as _RANK
        k       = max(1, int(_RANK * self.k_fraction))
        pivoted = 0

        for name, m in model.named_modules():
            if not isinstance(m, (LoRALayer, Conv1DLoRALayer)):
                continue
            if m.A_init_snapshot is None:
                continue

            with torch.no_grad():
                delta_A    = m.lora_A.data - m.A_init_snapshot
                delta_norm = torch.norm(delta_A).item()
                if delta_norm < 1e-8:
                    print(f"  {name}: delta_norm={delta_norm:.2e} -> skip")
                    continue
                try:
                    U, S, _ = torch.linalg.svd(delta_A, full_matrices=False)
                except Exception as e:
                    print(f"  {name}: SVD failed ({e}) -> skip")
                    continue

                V_k       = U[:, :k]
                A_current = m.lora_A.data.clone()
                Q_aug, _  = torch.linalg.qr(
                    torch.cat([A_current, V_k], dim=1))
                A_new = Q_aug[:, :RANK]

                # B-preserving re-projection (identical to BERT)
                proj  = A_new.T @ A_current
                B_old = m.lora_B_matrix
                B_new = proj @ B_old

                m.lora_A.data.copy_(A_new)
                if m._split_b:
                    for ri in range(m.rank):
                        m.lora_B_rows[ri].data.copy_(B_new[ri:ri+1, :])
                else:
                    m.lora_B.data.copy_(B_new)

                # Reset A optimizer state only (same as BERT, NOT B)
                if opt is not None:
                    for group in opt.param_groups:
                        for p in group["params"]:
                            if p is m.lora_A:
                                st = opt.state.get(p, {})
                                if "exp_avg" in st:
                                    st["exp_avg"].zero_()
                                    st["exp_avg_sq"].zero_()
                                    if "step" in st:
                                        st["step"] = (
                                            torch.zeros_like(st["step"])
                                            if isinstance(st["step"], torch.Tensor)
                                            else 0)

                print(f"  {name}: delta_norm={delta_norm:.3f} "
                      f"S[:3]={[f'{s:.3f}' for s in S[:3].tolist()]} k={k}")
                pivoted += 1

        print(f"[PivotLoRA-GPT2] Pivot done: {pivoted} layers. "
              f"B warmup: {self.warmup_steps} steps.")
        self.pivot_done       = True
        self.warmup_remaining = self.warmup_steps
        if opt is not None:
            self._set_b_lr(opt, self.base_lr)


# ══════════════════════════════════════════════════════════════════════════════
# ROUGE evaluation (batched)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_rouge(model, tokenizer, num_samples=500, batch_size=16):
    """Batched greedy generation + ROUGE scoring."""
    from rouge_score import rouge_scorer as rs_lib
    scorer = rs_lib.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    model.eval()
    dataset_raw = load_dataset("EdinburghNLP/xsum", split="validation")
    rng  = np.random.default_rng(42)
    idxs = rng.choice(len(dataset_raw),
                      size=min(num_samples, len(dataset_raw)), replace=False)
    examples = [dataset_raw[int(i)] for i in idxs]

    r1, r2, rL = [], [], []

    for start in range(0, len(examples), batch_size):
        batch_ex = examples[start:start + batch_size]
        prompts, refs = [], []
        for ex in batch_ex:
            doc_toks = tokenizer.encode(
                ex["document"],
                max_length=MAX_INPUT_LEN - MAX_TARGET_LEN - 20,
                truncation=True)
            prompts.append(
                tokenizer.decode(doc_toks, skip_special_tokens=True)
                + SUMMARY_MARKER)
            refs.append(ex["summary"])

        inputs = tokenizer(
            prompts, return_tensors="pt", truncation=True,
            max_length=MAX_INPUT_LEN - MAX_TARGET_LEN,
            padding=True).to(DEVICE)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_TARGET_LEN,
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


# ══════════════════════════════════════════════════════════════════════════════
# Experiment runner
# ══════════════════════════════════════════════════════════════════════════════

def run_gen_experiment(shot, seed, method):
    print(f"\n{'='*80}")
    print(f"Task=XSUM | Shot={shot} | Seed={seed} | Method={method}")
    print(f"{'='*80}")

    train_ds, val_ds, tokenizer = load_xsum_splits(shot, seed)

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.config.pad_token_id = tokenizer.pad_token_id

    tied = (model.lm_head.weight.data_ptr()
            == model.transformer.wte.weight.data_ptr())
    print(f"[INFO] lm_head tied to wte: {tied}")

    cbs = []

    if method == "lora":
        inject_gpt2(model, RANK)
        optimizer = build_uniform_optimizer(model)

    elif method == "eigenlora":
        subspaces, eig_dict = get_gpt2_subspaces(model, train_ds, RANK, seed)
        inject_gpt2(model, RANK, block_subspaces=subspaces,
                     eigenvalues_dict=eig_dict)
        optimizer = build_eigen_optimizer(model)

    elif method == "pivotlora":
        subspaces, eig_dict = get_gpt2_subspaces(model, train_ds, RANK, seed)
        inject_gpt2(model, RANK, block_subspaces=subspaces,
                     eigenvalues_dict=eig_dict, track_A_init=True)
        optimizer = build_eigen_optimizer(model)

        pivot_cb = GPT2PivotCallback(
            pivot_epoch=PIVOT_EPOCH,
            k_fraction=PIVOT_K_FRACTION,
            warmup_steps=PIVOT_WARMUP_STEPS,
            lr_ratio=LR_RATIO,
            base_lr=LEARNING_RATE,
        )
        pivot_cb._optimizer = optimizer
        cbs.append(pivot_cb)

    elif method == "pissa":
        b_inits = compute_gpt2_pissa(model, RANK)
        inject_gpt2(model, RANK, b_inits=b_inits, subtract_residual=True)
        optimizer = build_uniform_optimizer(model)

    elif method == "eva":
        subspaces = compute_gpt2_eva(model, train_ds, RANK)
        inject_gpt2(model, RANK, block_subspaces=subspaces)
        optimizer = build_uniform_optimizer(model)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Sanity check
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable > 5_000_000:
        print(f"  [WARN] {trainable:,} trainable params -- check lm_head is frozen!")

    _ta_kwargs = dict(
        output_dir=f"tmp/xsum_{method}_{seed}",
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        seed=seed,
        dataloader_num_workers=0,
        fp16=True,
        tf32=False,
        disable_tqdm=True,
    )
    try:
        training_args = TrainingArguments(eval_strategy="no", **_ta_kwargs)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy="no", **_ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        callbacks=cbs,
        optimizers=(optimizer, None),
    )

    t0 = time.perf_counter()
    trainer.train()
    wall = time.perf_counter() - t0

    # Check training health
    loss_logs = [l["loss"] for l in trainer.state.log_history if "loss" in l]
    if loss_logs:
        print(f"  [TRAIN] loss: {loss_logs[0]:.3f} -> {loss_logs[-1]:.3f} "
              f"({len(loss_logs)} steps)")
        if any(np.isnan(l) or np.isinf(l) for l in loss_logs):
            print("  [WARN] NaN/Inf in loss!")

    print(f"[EVAL] ROUGE on 500 val samples...")
    rouge = evaluate_rouge(model, tokenizer, num_samples=500, batch_size=16)
    print(f"  [RESULT] R1={rouge['rouge1']:.4f} R2={rouge['rouge2']:.4f} "
          f"RL={rouge['rougeL']:.4f} | Wall={wall:.1f}s")

    return rouge, wall


# ══════════════════════════════════════════════════════════════════════════════
# Study runner
# ══════════════════════════════════════════════════════════════════════════════

def run_study(methods=None, shots=None, seeds=None):
    methods = methods or METHODS
    shots   = shots   or SHOTS
    seeds   = seeds   or SEED_LIST

    completed = {}
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                key = (row["task"], int(row["shot"]), row["method"], int(row["seed"]))
                try:
                    completed[key] = {k: float(row[k])
                                      for k in ("rouge1", "rouge2", "rougeL")}
                except (ValueError, KeyError):
                    pass
        print(f"[RESUME] {len(completed)} completed rows")

    need_header = not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0
    csv_fh = open(RESULTS_CSV, "a", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
    if need_header:
        writer.writeheader()

    all_results = {}

    for shot in shots:
        for method in methods:
            scores_list = []
            for seed in seeds:
                key = ("EdinburghNLP/xsum", shot, method, seed)
                if key in completed:
                    scores_list.append(completed[key])
                    print(f"  [SKIP] xsum-{shot}-{method}-s{seed}: "
                          f"R1={completed[key]['rouge1']:.4f}")
                    continue
                try:
                    rouge, wall = run_gen_experiment(shot, seed, method)
                    scores_list.append(rouge)
                    completed[key] = rouge
                    writer.writerow({"task": "EdinburghNLP/xsum", "shot": shot,
                                     "method": method, "seed": seed,
                                     **{k: f"{v:.6f}" for k, v in rouge.items()}})
                    csv_fh.flush()
                except Exception as e:
                    print(f"  CRASH xsum-{shot}-{method}-s{seed}: {e}")
                    import traceback; traceback.print_exc()

            if scores_list:
                r1 = np.mean([s['rouge1'] for s in scores_list])
                r2 = np.mean([s['rouge2'] for s in scores_list])
                rL = np.mean([s['rougeL'] for s in scores_list])
                all_results[(shot, method)] = (r1, r2, rL)
                print(f">>> SUMMARY xsum {shot}-shot {method}: "
                      f"R1={r1:.4f} R2={r2:.4f} RL={rL:.4f}")

    csv_fh.close()

    # Final table
    col_w = 26
    print(f"\n{'='*110}")
    print("XSUM GENERATION RESULTS  (GPT-2 small | ROUGE-1 / ROUGE-2 / ROUGE-L)")
    print(f"Hyperparams: rank={RANK} alpha={ALPHA} lr={LEARNING_RATE} "
          f"lr_ratio={LR_RATIO} pivot={PIVOT_EPOCH} epochs={EPOCHS}")
    print(f"{'='*110}")
    print(f"  {'Method':<20}", end="")
    for s in shots:
        print(f"{str(s)+'-shot':^{col_w}}", end="")
    print()
    print(f"  {'─'*20}" + f"{'─'*col_w}" * len(shots))
    for method in methods:
        print(f"  {method:<20}", end="")
        for shot in shots:
            if (shot, method) in all_results:
                r1, r2, rL = all_results[(shot, method)]
                print(f"  {r1:.3f}/{r2:.3f}/{rL:.3f}{'':6}", end="")
            else:
                print(f"  {'N/A':^{col_w-2}}  ", end="")
        print()
    print(f"{'='*110}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--shots", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    if args.smoke:
        print("\n  SMOKE TEST\n")
        EPOCHS = 2
        run_study(methods=["lora", "pivotlora"], shots=[64], seeds=[42])
    else:
        seeds = SEED_LIST[:args.seeds] if args.seeds else SEED_LIST
        run_study(methods=args.methods, shots=args.shots, seeds=seeds)