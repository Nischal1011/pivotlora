import os
import torch
import torch.nn as nn
import numpy as np
import warnings
import scipy.linalg
import hashlib
from collections import defaultdict
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import string, re, collections, csv, time
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from peft import LoraConfig, AdaLoraConfig, get_peft_model, TaskType
from transformers import AutoModelForQuestionAnswering

# ==================== CONFIGURATION ====================
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SEED_LIST = [42, 43, 44, 45, 46]
MODEL_NAME = "bert-base-uncased"
RANK = 8
ALPHA = 16
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
EPOCHS = 15

# PivotLoRA hyperparameters
PIVOT_EPOCH = 3
PIVOT_K_FRACTION = 0.5
PIVOT_WARMUP_STEPS = 10

TASK_INFO = {
    "sst2":     {"dataset": ("nyu-mll/glue", "sst2"),    "text_fields": ["sentence"],
                 "num_labels": 2, "metric": "accuracy", "is_regression": False, "task_type": "cls"},
    "agnews":   {"dataset": ("fancyzhx/ag_news",),         "text_fields": ["text"],
                 "num_labels": 4, "metric": "accuracy", "is_regression": False, "task_type": "cls"},
    "rte":      {"dataset": ("nyu-mll/glue", "rte"),     "text_fields": ["sentence1", "sentence2"],
                 "num_labels": 2, "metric": "accuracy", "is_regression": False, "task_type": "cls"},
    "cola":     {"dataset": ("nyu-mll/glue", "cola"),    "text_fields": ["sentence"],
                 "num_labels": 2, "metric": "matthews",  "is_regression": False, "task_type": "cls"},
    "mnli":     {"dataset": ("nyu-mll/glue", "mnli"),    "text_fields": ["premise", "hypothesis"],
                 "num_labels": 3, "metric": "accuracy", "is_regression": False, "task_type": "cls"},
    "qnli":     {"dataset": ("nyu-mll/glue", "qnli"),    "text_fields": ["question", "sentence"],
                 "num_labels": 2, "metric": "accuracy", "is_regression": False, "task_type": "cls"},
    # Auxiliary task for the alignment analysis (not in the default task list).
    # Labels are the Boolean "answer" field, cast to {0, 1}.
    "boolq":    {"dataset": ("google/boolq",),           "text_fields": ["question", "passage"],
                 "label_key": "answer",
                 "num_labels": 2, "metric": "accuracy", "is_regression": False, "task_type": "cls"},
}

# ==================== METHOD CONFIG TABLE ====================
METHOD_CONFIGS = {
    "adalora":      dict(is_peft=True,  subspace_type="none",      lr_ratio=1,  peft_variant="adalora"),
    "dora":         dict(is_peft=False, subspace_type="none",      lr_ratio=1,  use_dora=True),
    "lora":       dict(is_peft=False, subspace_type="none",      lr_ratio=1),
    "lora_plus":    dict(is_peft=False, subspace_type="none",      lr_ratio=16),
    "pissa":        dict(is_peft=False, subspace_type="pissa",     lr_ratio=1,  alpha_override="rank"),
    "lora_ga":      dict(is_peft=False, subspace_type="lora_ga",   lr_ratio=1),
    "lda_lora":  dict(is_peft=False, subspace_type="lda",       lr_ratio=16),
    "eigenlora":       dict(is_peft=False, subspace_type="pca_eigen", lr_ratio=16),
    "pivotlora":   dict(is_peft=False, subspace_type="pca_eigen", lr_ratio=16, use_pivot=True),
    "eva":          dict(is_peft=False, subspace_type="eva",        lr_ratio=1),
    "srlora":       dict(is_peft=False, subspace_type="none",       lr_ratio=1,  sr_lora=True),
    # ── NEW: LoRAM (Zhang et al. NeurIPS 2025) ──
    # Magnitude-driven init: deterministic orthogonal basis scaled by log(1 + sigma_max(W)).
    # Claims spectral init gains come from magnitude amplification, not subspace quality.
    # Uniform LR (lr_ratio=1). No full SVD needed — just top singular value via power iteration.
    "loram":        dict(is_peft=False, subspace_type="loram",     lr_ratio=1),
    # ── NEW: LoRA-One (Zhang et al. ICML 2025 Oral) ──
    # One-step full gradient SVD init. Both A and B get singular value scaling.
    # Proves LoRA adapters align with gradient singular subspaces.
    # Stronger theoretical baseline than LoRA-GA.
    "lora_one":     dict(is_peft=False, subspace_type="lora_one",  lr_ratio=1),
}

# ==================== DATA LOADING ====================
def load_task_with_splits(task_name, n_shot, seed):
    print(f"\n[DATA] Loading {task_name}, n_shot={n_shot}, seed={seed}...")
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    info = TASK_INFO[task_name]
    dataset = load_dataset(*info["dataset"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    task_type = info.get("task_type", "cls")

    if task_type == "qa":
        def tokenize_qa(examples):
            enc = tokenizer(
                examples["question"], examples["context"],
                truncation=True, max_length=384, padding="max_length",
                return_offsets_mapping=True,
            )
            offset_mapping = enc.pop("offset_mapping")
            start_positions, end_positions = [], []
            for i, offsets in enumerate(offset_mapping):
                ans = examples["answers"][i]
                if len(ans["answer_start"]) == 0:
                    start_positions.append(0); end_positions.append(0); continue
                ans_start = ans["answer_start"][0]
                ans_end   = ans_start + len(ans["text"][0])
                seq_ids   = enc.sequence_ids(i)
                ctx_toks  = [ti for ti, s in enumerate(seq_ids) if s == 1]
                sp = ep = None
                for ti in ctx_toks:
                    s, e = offsets[ti]
                    if sp is None and s <= ans_start < e:
                        sp = ti
                    if s < ans_end <= e:
                        ep = ti
                if sp is None: sp = 0
                if ep is None: ep = sp
                if ep < sp:    ep = sp
                start_positions.append(sp)
                end_positions.append(ep)
            enc["start_positions"] = start_positions
            enc["end_positions"]   = end_positions
            return enc

        raw_val_split    = dataset["validation"]
        raw_val_contexts = raw_val_split["context"]
        raw_val_answers  = raw_val_split["answers"]

        dataset = dataset.map(tokenize_qa, batched=True,
                               remove_columns=dataset["train"].column_names)

        val_offset_mapping = []
        val_sequence_ids   = []
        for ex in raw_val_split:
            enc = tokenizer(
                ex["question"], ex["context"],
                truncation=True, max_length=384, padding="max_length",
                return_offsets_mapping=True,
            )
            val_offset_mapping.append(enc["offset_mapping"])
            val_sequence_ids.append(enc.sequence_ids())

        qa_eval_meta = {
            "answers":       raw_val_answers,
            "contexts":      raw_val_contexts,
            "offset_mapping": val_offset_mapping,
            "sequence_ids":   val_sequence_ids,
        }

        dataset.set_format(type="torch",
                           columns=["input_ids", "attention_mask",
                                    "start_positions", "end_positions"])
        train_full = dataset["train"]
        val_ds = dataset["validation"]
        val_ds._qa_meta = qa_eval_meta

        if n_shot == "full" or n_shot is None:
            print(f"[DATA] QA full train: {len(train_full)} | Val: {len(val_ds)}")
            return train_full, val_ds, info

        rng = np.random.default_rng(seed)
        idxs = np.arange(len(train_full))
        rng.shuffle(idxs)
        train_ds = train_full.select(idxs[:n_shot].tolist())
        print(f"[DATA] QA Train: {len(train_ds)} (requested {n_shot}) | Val: {len(val_ds)}")
        return train_ds, val_ds, info

    # ── Classification ───────────────────────────────────────────────────────
    def tokenize(examples):
        fields = info["text_fields"]
        args = (examples[fields[0]],) if len(fields) == 1 \
               else (examples[fields[0]], examples[fields[1]])
        return tokenizer(*args, truncation=True, max_length=128, padding="max_length")

    dataset = dataset.map(tokenize, batched=True)
    if "label_key" in info:
        key = info["label_key"]
        dataset = dataset.map(lambda ex: {"label": [int(v) for v in ex[key]]}, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_full = dataset["train"]
    if "validation" in dataset:
        val_ds = dataset["validation"]
    elif "validation_matched" in dataset:
        val_ds = dataset["validation_matched"]
        print("[DATA] Using validation_matched split (MNLI)")
    elif "test" in dataset:
        val_ds = dataset["test"]
        print("[DATA] Using test split as validation (AG News / no val split available)")
    else:
        raise KeyError(f"No validation split found in dataset. Available: {list(dataset.keys())}")

    if n_shot == "full" or n_shot is None:
        print(f"[DATA] Using full train set: {len(train_full)} | Val: {len(val_ds)}")
        return train_full, val_ds, info

    rng = np.random.default_rng(seed)
    labels_arr = np.array(train_full["labels"])
    indices = np.arange(len(train_full))
    train_indices = []
    for label in range(info["num_labels"]):
        lbl_indices = indices[labels_arr == label]
        rng.shuffle(lbl_indices)
        n_per_class = min(n_shot // info["num_labels"], len(lbl_indices))
        train_indices.extend(lbl_indices[:n_per_class].tolist())
    train_indices = np.array(train_indices)
    rng.shuffle(train_indices)

    train_ds = train_full.select(train_indices.tolist())
    print(f"[DATA] Train: {len(train_ds)} (requested {n_shot}) | Val: {len(val_ds)}")
    return train_ds, val_ds, info


# ==================== MATH: ROBUST LDA ====================
def compute_multiclass_lda(X_np, Y_np, target_rank, num_classes, rng):
    N, D = X_np.shape
    unique_classes = np.unique(Y_np)
    directions = []
    X_centered = X_np - X_np.mean(axis=0)

    if N <= num_classes + 1 or N < 4 or len(unique_classes) < 2:
        k = min(target_rank, max(1, min(N, D)))
        pca = PCA(n_components=k)
        pca.fit(X_np)
        return torch.tensor(
            np.linalg.qr(pca.components_.T)[0][:, :target_rank], dtype=torch.float32
        )

    means = {}
    X_class_centered = np.zeros_like(X_np)
    for c in unique_classes:
        mask = (Y_np == c)
        X_c = X_np[mask]
        mu_c = X_c.mean(axis=0)
        means[c] = mu_c
        X_class_centered[mask] = X_c - mu_c

    lw = LedoitWolf(store_precision=False, assume_centered=True)
    lw.fit(X_class_centered)
    Sw = (lw.covariance_ + lw.covariance_.T) / 2.0

    if len(unique_classes) == 2:
        c0, c1 = unique_classes
        diff = means[c1] - means[c0]
        try:
            L = scipy.linalg.cholesky(Sw, lower=True)
            w = scipy.linalg.cho_solve((L, True), diff)
        except Exception:
            w = scipy.linalg.pinv(Sw) @ diff
        w /= (np.linalg.norm(w) + 1e-9)
        directions.append(w)
    else:
        global_mean = X_np.mean(axis=0)
        Sb = np.zeros_like(Sw)
        for c, mu_c in means.items():
            n_c = (Y_np == c).sum()
            diff = mu_c - global_mean
            Sb += n_c * np.outer(diff, diff)
        Sb = (Sb + Sb.T) / 2.0
        max_disc = min(target_rank, max(1, N - 1), len(unique_classes) - 1)
        try:
            vals, vecs = scipy.linalg.eigh(Sb, Sw)
            idx = np.argsort(vals)[::-1]
            for i in range(max_disc):
                directions.append(vecs[:, idx[i]])
        except (scipy.linalg.LinAlgError, np.linalg.LinAlgError, ValueError) as _eigh_err:
            print(f"  [LDA] eigh fallback (pinv): {_eigh_err}")
            try:
                Sw_inv = np.linalg.pinv(Sw)
                M = Sw_inv @ Sb
                M = (M + M.T) / 2.0
                vals2, vecs2 = np.linalg.eigh(M)
                idx2 = np.argsort(vals2)[::-1]
                for i in range(max_disc):
                    directions.append(vecs2[:, idx2[i]])
            except Exception as _pinv_err:
                print(f"  [LDA] pinv fallback also failed ({_pinv_err}), using PCA")
                k = min(target_rank, max(1, min(N, D)))
                pca = PCA(n_components=k)
                pca.fit(X_np)
                for comp in pca.components_[:max_disc]:
                    directions.append(comp)

    if len(directions) < target_rank:
        Q_so_far = np.stack(directions, axis=1) if directions else np.empty((D, 0))
        Q_so_far, _ = np.linalg.qr(Q_so_far)
        X_resid = X_centered - (X_centered @ Q_so_far @ Q_so_far.T)
        needed = target_rank - len(directions)
        max_pca = min(needed, max(0, N - 1 - len(directions)))
        if max_pca > 0:
            pca = PCA(n_components=max_pca)
            pca.fit(X_resid)
            for comp in pca.components_:
                directions.append(comp)

    if len(directions) < target_rank:
        current_Q = np.stack(directions, axis=1) if directions else np.empty((D, 0))
        current_Q, _ = np.linalg.qr(current_Q)
        for _ in range(target_rank - len(directions)):
            v = rng.standard_normal(D)
            for j in range(current_Q.shape[1]):
                v = v - np.dot(v, current_Q[:, j]) * current_Q[:, j]
            v = v / (np.linalg.norm(v) + 1e-12)
            current_Q = np.concatenate([current_Q, v[:, None]], axis=1)
            directions.append(v)

    Q = np.stack(directions[:target_rank], axis=1)
    Q_orth, _ = np.linalg.qr(Q)
    return torch.tensor(Q_orth[:, :target_rank], dtype=torch.float32)


def compute_pca_lda(X_np, Y_np, target_rank, num_classes, rng, pca_dim=32):
    N, D = X_np.shape
    effective_pca_dim = max(1, min(pca_dim, N - 1, D))
    pca = PCA(n_components=effective_pca_dim)
    X_reduced = pca.fit_transform(X_np)
    Q_lda_reduced = compute_multiclass_lda(X_reduced, Y_np, target_rank, num_classes, rng)
    Q_768 = pca.components_.T @ Q_lda_reduced.numpy()
    Q_orth, _ = np.linalg.qr(Q_768)
    return torch.tensor(Q_orth[:, :target_rank], dtype=torch.float32)


def compute_pca_lda_complement(X_np, Y_np, target_rank, num_classes, rng, pca_dim=32):
    N, D = X_np.shape
    effective_pca_dim = max(1, min(pca_dim, N - 1, D))
    pca = PCA(n_components=effective_pca_dim)
    X_reduced = pca.fit_transform(X_np)
    Q_lda_reduced = compute_multiclass_lda(X_reduced, Y_np, target_rank, num_classes, rng)
    Q_lda_768 = pca.components_.T @ Q_lda_reduced.numpy()
    Q_lda_768, _ = np.linalg.qr(Q_lda_768)
    Q_lda_768 = Q_lda_768[:, :target_rank]

    X_centered = X_np - X_np.mean(axis=0)
    X_resid = X_centered - (X_centered @ Q_lda_768 @ Q_lda_768.T)
    n_complement = max(1, min(target_rank, N - 1, D - target_rank))
    pca_resid = PCA(n_components=n_complement)
    pca_resid.fit(X_resid)
    Q_complement = pca_resid.components_.T

    if n_complement < target_rank:
        Q_combined = np.concatenate([Q_lda_768, Q_complement], axis=1)
        Q_combined, _ = np.linalg.qr(Q_combined)
        current_Q = Q_combined
        for _ in range(target_rank - n_complement):
            v = rng.standard_normal(D)
            for j in range(current_Q.shape[1]):
                v = v - np.dot(v, current_Q[:, j]) * current_Q[:, j]
            v = v / (np.linalg.norm(v) + 1e-12)
            Q_complement = np.concatenate([Q_complement, v[:, None]], axis=1)
            current_Q = np.concatenate([current_Q, v[:, None]], axis=1)

    Q_complement = Q_complement[:, :target_rank]

    for j in range(Q_lda_768.shape[1]):
        lda_col = Q_lda_768[:, j]
        for k in range(Q_complement.shape[1]):
            Q_complement[:, k] -= np.dot(Q_complement[:, k], lda_col) * lda_col

    for k in range(Q_complement.shape[1]):
        col_norm = np.linalg.norm(Q_complement[:, k])
        if col_norm < 1e-6:
            accepted = np.concatenate([Q_lda_768, Q_complement[:, :k]], axis=1)
            accepted_q, _ = np.linalg.qr(accepted) if accepted.shape[1] > 0 \
                             else (np.empty((D, 0)), None)
            v = rng.standard_normal(D)
            for j in range(accepted_q.shape[1]):
                v -= np.dot(v, accepted_q[:, j]) * accepted_q[:, j]
            col_norm = np.linalg.norm(v)
            Q_complement[:, k] = v / (col_norm + 1e-12)
        else:
            Q_complement[:, k] /= col_norm

    Q_orth, _ = np.linalg.qr(Q_complement)
    if not np.isfinite(Q_orth).all():
        Q_orth = np.linalg.qr(rng.standard_normal((D, target_rank)))[0]
    return torch.tensor(Q_orth[:, :target_rank], dtype=torch.float32)


# ==================== PCA INIT WITH EIGENVALUES (EigenLoRA) ====================
def compute_pca_eigen(X_np, target_rank):
    N, D = X_np.shape
    n_components = min(target_rank, N - 1, D)
    pca = PCA(n_components=n_components)
    pca.fit(X_np)
    Q = pca.components_.T

    if n_components < target_rank:
        rng = np.random.default_rng(42)
        Q_orth, _ = np.linalg.qr(Q)
        padding_needed = target_rank - n_components
        for _ in range(padding_needed):
            v = rng.standard_normal(D)
            for j in range(Q_orth.shape[1]):
                v = v - np.dot(v, Q_orth[:, j]) * Q_orth[:, j]
            v = v / (np.linalg.norm(v) + 1e-12)
            Q_orth = np.concatenate([Q_orth, v[:, None]], axis=1)
        Q = Q_orth[:, :target_rank]
        eigenvalues = np.concatenate([
            pca.explained_variance_,
            np.full(padding_needed, pca.explained_variance_.min() * 0.1 + 1e-8)
        ])
    else:
        Q_orth, _ = np.linalg.qr(Q)
        Q = Q_orth[:, :target_rank]
        eigenvalues = pca.explained_variance_[:target_rank]

    return torch.tensor(Q, dtype=torch.float32), eigenvalues


# ==================== EVA INIT ====================
def compute_eva(model, dataset, rank):
    print("[EVA] Computing activation SVD (all token positions)...")
    model.eval().to(DEVICE)
    activations = defaultdict(list)

    def make_hook(idx):
        def hook(module, inputs):
            x = inputs[0].detach()
            B, T, D = x.shape
            activations[idx].append(x.reshape(-1, D).cpu())
        return hook

    handles = [
        layer.attention.self.query.register_forward_pre_hook(make_hook(i))
        for i, layer in enumerate(model.bert.encoder.layer)
    ]
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(DEVICE) for k, v in batch.items()
                      if k not in ("labels", "start_positions", "end_positions")}
            model(**inputs)
    for h in handles:
        h.remove()

    subspaces = {}
    for idx, acts in activations.items():
        X = torch.cat(acts, dim=0).numpy()
        n_comp = min(rank, X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(X)
        Q = pca.components_.T
        if n_comp < rank:
            rng = np.random.default_rng(42 + idx)
            Q_orth, _ = np.linalg.qr(Q)
            for _ in range(rank - n_comp):
                v = rng.standard_normal(X.shape[1])
                for j in range(Q_orth.shape[1]):
                    v -= np.dot(v, Q_orth[:, j]) * Q_orth[:, j]
                v /= np.linalg.norm(v) + 1e-12
                Q_orth = np.concatenate([Q_orth, v[:, None]], axis=1)
            Q = Q_orth[:, :rank]
        else:
            Q, _ = np.linalg.qr(Q)
            Q = Q[:, :rank]
        subspaces[idx] = torch.tensor(Q, dtype=torch.float32).to(DEVICE)
        expl = pca.explained_variance_ratio_[:n_comp].sum() * 100
        print(f"  [EVA] Layer {idx:>2}: explained_var={expl:.1f}%")

    print("[EVA] Done.")
    return subspaces


# ==================== PISSA INIT ====================
def compute_pissa_init(weight, rank):
    W = weight.detach().float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r  = U[:, :rank]
    S_r  = S[:rank]
    Vh_r = Vh[:rank, :]
    A_init = Vh_r.T
    B_init = (U_r * S_r.unsqueeze(0)).T
    return A_init.cpu(), B_init.cpu()


# ==================== LORA-GA INIT ====================
def compute_lora_ga_init(model, layer_module, target_attr, dataset, rank, seed):
    original = getattr(layer_module, target_attr)
    W = original.weight
    W.requires_grad_(True)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=min(16, len(dataset)), shuffle=False)
    batch = next(iter(loader))

    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    inputs = {k: v.to(DEVICE) for k, v in batch.items()
              if k not in ("start_positions", "end_positions")}
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()

    G = W.grad.detach().float()
    W.requires_grad_(False)
    model.eval()

    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    U_r  = U[:, :rank]
    S_r  = S[:rank]
    Vh_r = Vh[:rank, :]
    A_init = Vh_r.T
    B_init = (U_r * S_r.unsqueeze(0)).T
    return A_init.cpu(), B_init.cpu()


# ══════════════════════════════════════════════════════════════════════════════
# NEW: LoRAM INIT (Zhang et al. NeurIPS 2025)
# ══════════════════════════════════════════════════════════════════════════════
def compute_loram_init(weight, rank):
    """
    LoRAM (Zhang et al. NeurIPS 2025): magnitude-driven initialization.

    1. Compute sigma_max(W) via power iteration (no full SVD).
    2. Build A from deterministic orthogonal basis (QR of fixed-seed random matrix).
    3. Scale A by log(1 + sigma_max) to match spectral init magnitude.
    4. B = None (zeros, standard LoRA convention).

    Returns (A_init, B_init) where A is (d_in, rank), B_init is None.
    """
    W = weight.detach().float()  # (d_out, d_in)
    d_out, d_in = W.shape

    # Top singular value via power iteration (10 iters, no full SVD)
    v = torch.randn(d_in, device=W.device)
    v = v / v.norm()
    for _ in range(10):
        u = W @ v
        u = u / u.norm().clamp(min=1e-8)
        v = W.T @ u
        v = v / v.norm().clamp(min=1e-8)
    sigma_max = abs((u @ W @ v).item())

    # Deterministic orthogonal basis (fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    random_matrix = rng.standard_normal((d_in, rank)).astype(np.float32)
    Q, _ = np.linalg.qr(random_matrix)
    Q = Q[:, :rank]

    # Scale by log(1 + sigma_max) — the magnitude driver
    magnitude = np.log(1.0 + sigma_max)
    A_init = torch.tensor(Q * magnitude, dtype=torch.float32)

    print(f"    [LoRAM] sigma_max={sigma_max:.2f} -> magnitude={magnitude:.4f}")
    return A_init, None  # B=None -> zeros in LoRALayer


# ══════════════════════════════════════════════════════════════════════════════
# NEW: LoRA-One INIT (Zhang et al. ICML 2025 Oral)
# ══════════════════════════════════════════════════════════════════════════════
def compute_lora_one_init(model, layer_module, target_attr, dataset, rank, seed,
                          alpha=16):
    """
    LoRA-One: one-step full gradient SVD init with balanced singular value scaling.

    1. Compute gradient ∇_W L using the few-shot batch.
    2. SVD of gradient: U, S, Vh = SVD(∇_W L).
    3. A = (1/√γ) * Vh_r^T * diag(S_r)    shape: (d_in, r)
       B = (1/√γ) * diag(S_r) * U_r^T     shape: (r, d_out)
       where γ = α / √r

    Both A and B get singular value scaling (unlike LoRA-GA which only scales B).
    """
    original = getattr(layer_module, target_attr)
    W = original.weight
    W.requires_grad_(True)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=min(16, len(dataset)), shuffle=False)
    batch = next(iter(loader))

    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    inputs = {k: v.to(DEVICE) for k, v in batch.items()
              if k not in ("start_positions", "end_positions")}
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()

    G = W.grad.detach().float()  # (d_out, d_in)
    W.requires_grad_(False)
    model.eval()

    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    U_r = U[:, :rank]       # (d_out, r)
    S_r = S[:rank]           # (r,)
    Vh_r = Vh[:rank, :]      # (r, d_in)

    gamma = alpha / np.sqrt(rank)
    inv_sqrt_gamma = 1.0 / np.sqrt(gamma)

    # Our convention: A is (d_in, r), B is (r, d_out)
    # We need B^T @ A^T = (1/γ) * U_r @ diag(S_r^2) @ Vh_r
    A_init = inv_sqrt_gamma * (Vh_r.T * S_r.unsqueeze(0))  # (d_in, r)
    B_init = inv_sqrt_gamma * (S_r.unsqueeze(1) * U_r.T)   # (r, d_out)

    print(f"    [LoRA-One] ||G||={G.norm():.2f}, S[:3]={S_r[:3].tolist()}, "
          f"γ={gamma:.2f}, 1/√γ={inv_sqrt_gamma:.4f}")

    return A_init.cpu(), B_init.cpu()


# ==================== SUBSPACE COMPUTATION ====================
def get_block_subspaces(model, dataset, rank, num_classes, seed, subspace_type="lda"):
    print(f"[SUBSPACE] Computing: {subspace_type}")
    model.eval().to(DEVICE)
    activations = defaultdict(list)
    labels_collected = []

    def get_qv_prehook(layer_idx):
        def prehook(module, inputs):
            x = inputs[0].detach()
            activations[layer_idx].append(x[:, 0, :].cpu())
        return prehook

    handles = [
        layer.attention.self.query.register_forward_pre_hook(get_qv_prehook(i))
        for i, layer in enumerate(model.bert.encoder.layer)
    ]
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            label_key = "labels" if "labels" in batch else None
            inputs = {k: v.to(DEVICE) for k, v in batch.items()
                      if k not in ("labels", "start_positions", "end_positions")}
            model(**inputs)
            if label_key and label_key in batch:
                labels_collected.extend(batch[label_key].numpy())
    for h in handles:
        h.remove()

    Y_np = np.array(labels_collected) if labels_collected else None
    subspaces = {}
    eigenvalues_dict = {}

    for idx, acts in activations.items():
        X_np = torch.cat(acts, dim=0).numpy()
        layer_seed = (seed * 10007 + idx) % (2 ** 32)
        layer_rng = np.random.default_rng(layer_seed)

        if subspace_type == "lda":
            Q = compute_multiclass_lda(X_np, Y_np, rank, num_classes, layer_rng)
            subspaces[idx] = Q.to(DEVICE)
        elif subspace_type == "pca_lda_complement":
            Q = compute_pca_lda_complement(X_np, Y_np, rank, num_classes, layer_rng, pca_dim=32)
            subspaces[idx] = Q.to(DEVICE)
        elif subspace_type == "pca_eigen":
            Q, eigs = compute_pca_eigen(X_np, rank)
            subspaces[idx] = Q.to(DEVICE)
            eigenvalues_dict[idx] = eigs
            print(f"  [PCA-EIGEN] Layer {idx:>2}: eig {eigs.min():.2f}..{eigs.max():.2f} "
                  f"mean={eigs.mean():.2f} max/mean={eigs.max()/eigs.mean():.2f}x")
        elif subspace_type in ("pissa", "lora_ga", "loram", "lora_one"):
            pass  # handled directly in run_experiment
        elif subspace_type == "eva":
            pass  # handled directly in run_experiment
        else:
            raise ValueError(f"Unknown subspace_type: {subspace_type}")

    return subspaces, eigenvalues_dict


# ==================== LORA LAYER ====================
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank, alpha,
                 init_A=None, init_B=None, track_A_init=False,
                 b_lr_scales=None, is_dora=False):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.is_dora = is_dora
        d_in = original_layer.in_features
        d_out = original_layer.out_features
        device = original_layer.weight.device

        if init_A is not None:
            scale = np.sqrt(5.0) / np.sqrt(d_in)
            self.lora_A = nn.Parameter(
                init_A.to(device=device, dtype=torch.float32) * scale
            )
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
            self.register_buffer(
                "b_lr_scales",
                torch.tensor(b_lr_scales, dtype=torch.float32, device=device)
            )
        else:
            if init_B is not None:
                self.lora_B = nn.Parameter(
                    init_B.to(device=device, dtype=torch.float32).clone())
            else:
                self.lora_B = nn.Parameter(torch.zeros(rank, d_out, device=device))
            self.lora_B_rows = None
            self.b_lr_scales = None

        if is_dora:
            col_norms = original_layer.weight.data.norm(dim=0, keepdim=True)
            self.dora_m = nn.Parameter(col_norms.to(dtype=torch.float32))
        else:
            self.dora_m = None

        if track_A_init:
            self.register_buffer("A_init_snapshot",
                                 self.lora_A.data.clone().detach())
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
        if self.is_dora:
            delta_W = (self.lora_A @ self.lora_B_matrix).T * self.scaling
            W_combined = self.original_layer.weight + delta_W
            col_norms = W_combined.norm(dim=0, keepdim=True).clamp(min=1e-8)
            W_dora = self.dora_m * (W_combined / col_norms)
            out = x @ W_dora.T
            if self.original_layer.bias is not None:
                out = out + self.original_layer.bias
            return out
        return self.original_layer(x) + (x @ self.lora_A @ self.lora_B_matrix) * self.scaling


# ==================== CALLBACKS ====================
class LoRAPlusCosineCallback(TrainerCallback):
    def __init__(self, optimizer, total_steps):
        self.optimizer   = optimizer
        self.total_steps = max(1, total_steps)
        self.init_lrs    = [g["lr"] for g in optimizer.param_groups]
        self._step       = 0

    def on_step_end(self, args, state, control, **kwargs):
        self._step += 1
        import math
        frac = self._step / self.total_steps
        cos_scale = 0.5 * (1.0 + math.cos(math.pi * frac))
        for g, lr0 in zip(self.optimizer.param_groups, self.init_lrs):
            g["lr"] = lr0 * cos_scale


class PivotCallback(TrainerCallback):
    def __init__(self, pivot_epoch, k_fraction, warmup_steps, lr_ratio, base_lr):
        self.pivot_epoch  = pivot_epoch
        self.k_fraction   = k_fraction
        self.warmup_steps = warmup_steps
        self.lr_ratio     = lr_ratio
        self.base_lr      = base_lr
        self.pivot_done   = False
        self.warmup_remaining = 0
        self._optimizer   = None

    def on_epoch_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        if self.pivot_done or state.epoch is None:
            return
        epoch_f = state.epoch
        if epoch_f + 1e-4 < self.pivot_epoch:
            return
        if epoch_f > self.pivot_epoch + 1.0:
            return

        opt = optimizer or self._optimizer
        print(f"\n[PivotLoRA] === PIVOT at epoch {epoch_f:.3f} (target={self.pivot_epoch}) ===")
        k = max(1, int(RANK * self.k_fraction))
        pivoted = 0

        for name, m in model.named_modules():
            if not isinstance(m, LoRALayer) or m.A_init_snapshot is None:
                continue
            with torch.no_grad():
                delta_A = m.lora_A.data - m.A_init_snapshot
                delta_norm = torch.norm(delta_A).item()
                if delta_norm < 1e-8:
                    print(f"  {name}: Delta_A={delta_norm:.2e} -> skip")
                    continue
                try:
                    U, S, _ = torch.linalg.svd(delta_A, full_matrices=False)
                except Exception as e:
                    print(f"  {name}: SVD failed ({e}) -> skip")
                    continue

                V_k = U[:, :k]
                A_current = m.lora_A.data.clone()
                combined = torch.cat([A_current, V_k], dim=1)
                Q_aug, _ = torch.linalg.qr(combined)
                A_new = Q_aug[:, :RANK]

                proj  = A_new.T @ A_current
                B_old = m.lora_B_matrix
                B_new = proj @ B_old

                m.lora_A.data.copy_(A_new)
                if m._split_b:
                    for ri in range(m.rank):
                        m.lora_B_rows[ri].data.copy_(B_new[ri:ri+1, :])
                else:
                    m.lora_B.data.copy_(B_new)

                if opt is not None:
                    for group in opt.param_groups:
                        for p in group["params"]:
                            if p is m.lora_A:
                                st = opt.state.get(p, {})
                                if "exp_avg" in st:
                                    st["exp_avg"].zero_()
                                    st["exp_avg_sq"].zero_()
                                    if "step" in st:
                                        st["step"] = (torch.zeros_like(st["step"])
                                                      if isinstance(st["step"], torch.Tensor) else 0)
                print(f"  {name}: norm={delta_norm:.3f} S[:3]={S[:3].tolist()} k={k}")
                pivoted += 1

        print(f"[PivotLoRA] Pivot done: {pivoted} layers. Starting {self.warmup_steps}-step B warmup.")
        self.pivot_done = True
        self.warmup_remaining = self.warmup_steps
        if opt is not None:
            self._set_b_lr(opt, self.base_lr)

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        if not self.pivot_done or self.warmup_remaining <= 0:
            return
        opt = optimizer or self._optimizer
        if opt is None:
            return
        steps_done = self.warmup_steps - self.warmup_remaining
        frac = (steps_done + 1) / self.warmup_steps
        current_b_lr = self.base_lr + frac * (self.lr_ratio - 1) * self.base_lr
        self._set_b_lr(opt, current_b_lr)
        self.warmup_remaining -= 1
        if self.warmup_remaining == 0:
            print(f"[PivotLoRA] Warmup done. B LR={self.lr_ratio * self.base_lr:.2e}")

    def _set_b_lr(self, opt, lr):
        for group in opt.param_groups:
            if group.get("_is_B_group", False):
                group["lr"] = lr


class SRLoRACallback(TrainerCallback):
    def __init__(self, interval_steps, fuse_frac=0.25):
        self.interval  = interval_steps
        self.fuse_frac = fuse_frac
        self._step     = 0
        self._next_svd_idx = {}

    def on_step_end(self, args, state, control,
                    model=None, optimizer=None, **kwargs):
        self._step += 1
        if self._step % self.interval != 0:
            return

        print(f"\n[SRLoRA] Recomposition at step {self._step}")
        recomp_count = 0

        for name, m in model.named_modules():
            if not isinstance(m, LoRALayer):
                continue

            with torch.no_grad():
                A = m.lora_A.data
                B = m.lora_B_matrix
                rank = m.rank
                n_fuse = max(1, int(rank * self.fuse_frac))

                scores = (B.norm(dim=1) * A.norm(dim=0)) * m.scaling
                low_idx = scores.argsort()[:n_fuse].tolist()

                W_frozen = m.original_layer.weight.data
                for ri in low_idx:
                    dW_ri = (A[:, ri:ri+1] @ B[ri:ri+1, :]).T * m.scaling
                    W_frozen.add_(dW_ri)

                key = name
                if key not in self._next_svd_idx:
                    self._next_svd_idx[key] = 0

                try:
                    U, S, Vh = torch.linalg.svd(W_frozen.float(), full_matrices=False)
                except Exception as e:
                    print(f"  {name}: SVD failed ({e}), skipping")
                    continue

                d_out, d_in = W_frozen.shape
                max_svd = min(d_in, d_out)

                kept_idx = [ri for ri in range(rank) if ri not in low_idx]
                A_kept = A[:, kept_idx]

                new_A = A.clone()
                new_B_rows = []

                for ri in low_idx:
                    svd_i = self._next_svd_idx[key] % max_svd
                    self._next_svd_idx[key] += 1

                    v = Vh[svd_i, :].clone()
                    for j in range(A_kept.shape[1]):
                        v = v - (v @ A_kept[:, j]) * A_kept[:, j]
                    norm_v = v.norm()
                    if norm_v < 1e-7:
                        v = torch.randn_like(v)
                        v /= v.norm().clamp(min=1e-8)
                    else:
                        v /= norm_v

                    new_A[:, ri] = v
                    new_B_rows.append(ri)

                m.lora_A.data.copy_(new_A)
                if m._split_b:
                    for ri in new_B_rows:
                        m.lora_B_rows[ri].data.zero_()
                else:
                    for ri in new_B_rows:
                        m.lora_B.data[ri, :] = 0.0

                if optimizer is not None:
                    for group in optimizer.param_groups:
                        for p in group["params"]:
                            if p is m.lora_A:
                                st = optimizer.state.get(p, {})
                                if "exp_avg" in st:
                                    st["exp_avg"].zero_()
                                    st["exp_avg_sq"].zero_()
                            if m._split_b:
                                for ri in new_B_rows:
                                    if p is m.lora_B_rows[ri]:
                                        st = optimizer.state.get(p, {})
                                        if "exp_avg" in st:
                                            st["exp_avg"].zero_()
                                            st["exp_avg_sq"].zero_()
                            elif p is m.lora_B:
                                st = optimizer.state.get(p, {})
                                if "exp_avg" in st:
                                    for ri in new_B_rows:
                                        st["exp_avg"][ri, :] = 0.0
                                        st["exp_avg_sq"][ri, :] = 0.0

                recomp_count += 1

        print(f"[SRLoRA] Recomposed {recomp_count} layers (fused {n_fuse}/{rank} pairs each).")


# ==================== INJECTOR ====================
def inject_model(model, rank, block_subspaces,
                 use_lda=False, apply_rotation=True,
                 eigenvalues_dict=None, track_A_init=False,
                 b_inits=None, is_dora=False,
                 effective_alpha=None,
                 subtract_residual=False):
    if effective_alpha is None:
        effective_alpha = ALPHA
    print(f"[INJECT] use_lda={use_lda}, rank={rank}")
    target_modules = ["query", "value"]
    count = 0

    for i, layer_module in enumerate(model.bert.encoder.layer):
        init_A = None
        init_B = None

        if b_inits is not None and i in b_inits:
            a_b_pair = b_inits[i]
        else:
            a_b_pair = None

        if use_lda and block_subspaces is not None and a_b_pair is None:
            init_A = block_subspaces.get(i)

        for target in target_modules:
            original = getattr(layer_module.attention.self, target)
            curr_A = init_A
            curr_B = init_B

            if a_b_pair is not None and target in a_b_pair:
                curr_A, curr_B = a_b_pair[target]

            if curr_A is not None and apply_rotation and curr_B is None:
                rot_seed = f"layer_{i}_{target}"
                seed_bytes = hashlib.sha256(rot_seed.encode()).digest()
                seed_int = int.from_bytes(seed_bytes[:4], "little")
                g = torch.Generator(device=DEVICE)
                g.manual_seed(seed_int)
                rot, _ = torch.linalg.qr(
                    torch.randn(rank, rank, generator=g, device=DEVICE)
                )
                curr_A = curr_A.to(DEVICE) @ rot

            b_scales = None
            if eigenvalues_dict is not None and i in eigenvalues_dict:
                eigs = eigenvalues_dict[i]
                sigma_mean = float(eigs.mean()) + 1e-8
                b_scales = (eigs / sigma_mean).astype(np.float32)

            adapter = LoRALayer(original, rank, effective_alpha,
                                  init_A=curr_A,
                                  init_B=curr_B,
                                  track_A_init=track_A_init,
                                  b_lr_scales=b_scales,
                                  is_dora=is_dora)

            if subtract_residual and curr_A is not None and curr_B is not None:
                with torch.no_grad():
                    A = adapter.lora_A.data
                    B = adapter.lora_B_matrix
                    delta_W = (A @ B).T * adapter.scaling
                    original.weight.data -= delta_W
                    print(f"  [RESIDUAL] layer {i} {target}: ||delta_W||={delta_W.norm():.4f}")

            setattr(layer_module.attention.self, target, adapter)
            count += 1

    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALayer):
            m.lora_A.requires_grad = True
            if m._split_b:
                for row in m.lora_B_rows:
                    row.requires_grad = True
            else:
                m.lora_B.requires_grad = True
            if m.dora_m is not None:
                m.dora_m.requires_grad = True
    for head_name in ("classifier", "qa_outputs"):
        head = getattr(model, head_name, None)
        if head is not None:
            for p in head.parameters():
                p.requires_grad = True

    print(f"[INJECT] Injected {count} adapters.")
    return model


# ==================== METRICS ====================
def normalize_answer(s):
    def remove_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t): return " ".join(t.split())
    def remove_punc(t): return "".join(ch for ch in t if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _squad_text_f1_em(pred_text, gold_texts):
    pred_norm = normalize_answer(pred_text)
    best_f1, best_em = 0.0, 0
    for gold in gold_texts:
        gold_norm = normalize_answer(gold)
        em = int(pred_norm == gold_norm)
        best_em = max(best_em, em)
        pred_toks = pred_norm.split()
        gold_toks = gold_norm.split()
        common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        prec = num_same / len(pred_toks)
        rec  = num_same / len(gold_toks)
        best_f1 = max(best_f1, 2 * prec * rec / (prec + rec))
    return best_f1, best_em


def make_squad_compute_metrics(val_ds):
    meta = val_ds._qa_meta
    answers_all      = meta["answers"]
    contexts_all     = meta["contexts"]
    offset_maps_all  = meta["offset_mapping"]
    seq_ids_all      = meta["sequence_ids"]
    MAX_ANSWER_LEN   = 30

    def compute_metrics(p):
        start_logits, end_logits = p.predictions
        N = len(start_logits)
        f1_scores, em_scores = [], []
        for i in range(N):
            sl      = start_logits[i]
            el      = end_logits[i]
            offsets = offset_maps_all[i]
            seq_ids = seq_ids_all[i]
            context = contexts_all[i]
            answers = answers_all[i]
            ctx_idx = [ti for ti, s in enumerate(seq_ids) if s == 1]
            if not ctx_idx:
                f1_scores.append(0.0); em_scores.append(0); continue
            ctx_start_tok = ctx_idx[0]
            ctx_end_tok   = ctx_idx[-1]
            sl_masked = sl.copy()
            el_masked = el.copy()
            sl_masked[:ctx_start_tok]     = -1e9
            sl_masked[ctx_end_tok + 1:]   = -1e9
            el_masked[:ctx_start_tok]     = -1e9
            el_masked[ctx_end_tok + 1:]   = -1e9
            from collections import deque
            best_score = -1e18
            best_sp = best_ep = ctx_start_tok
            window = deque()
            for ep in ctx_idx:
                while window and ep - window[0][1] >= MAX_ANSWER_LEN:
                    window.popleft()
                sp_val = float(sl_masked[ep])
                while window and window[-1][0] <= sp_val:
                    window.pop()
                window.append((sp_val, ep))
                score = window[0][0] + float(el_masked[ep])
                if score > best_score:
                    best_score = score
                    best_ep = ep
                    best_sp = window[0][1]
            char_start = offsets[best_sp][0]
            char_end   = offsets[best_ep][1]
            pred_text  = context[char_start:char_end]
            gold_texts = answers["text"] if answers["text"] else [""]
            f1, em = _squad_text_f1_em(pred_text, gold_texts)
            f1_scores.append(f1)
            em_scores.append(em)
        return {"squad_f1":    float(np.mean(f1_scores)),
                "exact_match": float(np.mean(em_scores))}
    return compute_metrics


def get_compute_metrics(info, val_ds_raw=None):
    if info["metric"] == "squad_f1":
        if val_ds_raw is None:
            raise ValueError("squad_f1 metric requires val_ds_raw with offset metadata")
        return make_squad_compute_metrics(val_ds_raw)

    def compute_metrics(p):
        preds  = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        m = info["metric"]
        if m == "accuracy":
            return {"accuracy": accuracy_score(labels, preds)}
        elif m == "f1":
            return {"f1": f1_score(labels, preds),
                    "accuracy": accuracy_score(labels, preds)}
        elif m == "matthews":
            return {"matthews": matthews_corrcoef(labels, preds)}
        return {}
    return compute_metrics


# ==================== EXPERIMENT RUNNER ====================
def run_experiment(task, shot, seed, method):
    cfg = METHOD_CONFIGS[method]
    shot_label = shot if shot != "full" else "FULL"
    print(f"\n{'='*80}")
    print(f"Task={task.upper()} | Shot={shot_label} | Seed={seed} | Method={method}")
    print(f"{'='*80}")

    if shot == "full":
        actual_epochs = 3
        actual_batch  = 32
        actual_lr     = 4e-4
    else:
        actual_epochs = EPOCHS
        actual_batch  = BATCH_SIZE
        actual_lr     = LEARNING_RATE

    train_ds, val_ds, info = load_task_with_splits(task, shot, seed)
    if info.get("task_type") == "qa":
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=info["num_labels"]
        ).to(DEVICE)

    cbs = []
    eigenvalues_dict = None

    # ── PEFT baselines ──────────────────────────────────────────────────────
    if cfg["is_peft"]:
        num_samples = len(train_ds)
        steps_per_epoch = max(1, (num_samples + actual_batch - 1) // actual_batch)
        total_steps = steps_per_epoch * actual_epochs

        peft_variant = cfg.get("peft_variant", method)
        task_type_peft = TaskType.QUESTION_ANS if info.get("task_type") == "qa" else TaskType.SEQ_CLS

        if peft_variant == "adalora":
            MIN_ADALORA_STEPS = 10
            if total_steps < MIN_ADALORA_STEPS:
                print(f"  [AdaLoRA] total_steps={total_steps} < {MIN_ADALORA_STEPS}; "
                      f"falling back to plain LoRA.")
                config = LoraConfig(
                    task_type=task_type_peft, r=RANK, lora_alpha=ALPHA,
                    target_modules=["query", "value"],
                )
            else:
                tfinal = max(1, int(total_steps * 0.75))
                tinit  = max(0, int(tfinal * 0.1))
                deltaT = max(1, (tfinal - tinit) // 10)
                if tfinal + deltaT > total_steps:
                    tfinal = total_steps - deltaT
                config = AdaLoraConfig(
                    task_type=task_type_peft, r=RANK, lora_alpha=ALPHA,
                    target_modules=["query", "value"], init_r=16, target_r=RANK,
                    tinit=tinit, tfinal=tfinal, deltaT=deltaT, total_step=total_steps,
                )
        else:
            config = LoraConfig(
                task_type=task_type_peft, r=RANK, lora_alpha=ALPHA,
                target_modules=["query", "value"],
            )

        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable > 2000, f"PEFT adapters not attached ({trainable} params)"

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params, lr=actual_lr,
            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
        )

    # ── Custom methods ──────────────────────────────────────────────────────
    else:
        subspace_type  = cfg["subspace_type"]
        do_pivot  = cfg.get("use_pivot", False)
        lr_ratio       = cfg["lr_ratio"]

        effective_subspace = subspace_type
        if info.get("task_type") == "qa":
            if subspace_type in ("lda", "pca_lda_complement"):
                print(f"  [SUBSPACE] QA task: {subspace_type} -> pca_eigen (no class labels)")
                effective_subspace = "pca_eigen"
            elif subspace_type in ("lora_ga", "lora_one"):
                print(f"  [SUBSPACE] QA task: {subspace_type} -> none (no scalar loss; using random init)")
                effective_subspace = "none"

        alpha_override = cfg.get("alpha_override", None)
        effective_alpha = RANK if alpha_override == "rank" else ALPHA

        # ── PiSSA / LoRA-GA / LoRAM / LoRA-One: compute per-layer init pairs ──
        b_inits = None
        if effective_subspace in ("pissa", "lora_ga", "loram", "lora_one"):
            target_modules = ["query", "value"]
            b_inits = {}
            print(f"[{effective_subspace.upper()}] Computing weight/gradient inits...")
            for i, layer_module in enumerate(model.bert.encoder.layer):
                b_inits[i] = {}
                for target in target_modules:
                    original = getattr(layer_module.attention.self, target)
                    if effective_subspace == "pissa":
                        A_i, B_i = compute_pissa_init(original.weight, RANK)
                    elif effective_subspace == "loram":
                        A_i, B_i = compute_loram_init(original.weight, RANK)
                    elif effective_subspace == "lora_one":
                        A_i, B_i = compute_lora_one_init(
                            model, layer_module.attention.self, target, train_ds, RANK, seed,
                            alpha=effective_alpha)
                    else:  # lora_ga
                        A_i, B_i = compute_lora_ga_init(
                            model, layer_module.attention.self, target, train_ds, RANK, seed)
                    b_inits[i][target] = (A_i, B_i)
            block_subspaces  = None
            eigenvalues_dict = {}
            print(f"[{effective_subspace.upper()}] Done.")

        elif effective_subspace == "eva":
            block_subspaces  = compute_eva(model, train_ds, RANK)
            eigenvalues_dict = {}

        elif effective_subspace == "none":
            block_subspaces  = None
            eigenvalues_dict = {}
        else:
            block_subspaces, eigenvalues_dict = get_block_subspaces(
                model, train_ds, RANK, info["num_labels"], seed,
                subspace_type=effective_subspace,
            )

        use_structured_init = (effective_subspace not in ("none", "pissa", "lora_ga", "loram", "lora_one", "eva")) or effective_subspace == "eva"
        # LoRAM has B=None (zeros) so no residual subtraction needed
        # LoRA-One has non-zero A AND B, so subtract residual (like PiSSA and LoRA-GA)
        subtract_residual = effective_subspace in ("pissa", "lora_ga", "lora_one")
        model = inject_model(
            model, RANK, block_subspaces,
            use_lda=use_structured_init,
            apply_rotation=True,
            eigenvalues_dict=eigenvalues_dict,
            track_A_init=do_pivot,
            b_inits=b_inits,
            is_dora=cfg.get("use_dora", False),
            effective_alpha=effective_alpha,
            subtract_residual=subtract_residual,
        )

        use_eigen_lr = (eigenvalues_dict is not None and len(eigenvalues_dict) > 0)

        if lr_ratio > 1 and use_eigen_lr:
            A_params   = [p for n, p in model.named_parameters()
                          if p.requires_grad and "lora_A" in n]
            cls_params = [p for n, p in model.named_parameters()
                          if p.requires_grad and "classifier" in n]
            param_groups = [
                {"params": A_params, "lr": actual_lr, "weight_decay": 0.01},
            ]
            for mod_name, m in model.named_modules():
                if not isinstance(m, LoRALayer):
                    continue
                if m._split_b:
                    for rank_i, row_param in enumerate(m.lora_B_rows):
                        scale_i = m.b_lr_scales[rank_i].item()
                        param_groups.append({
                            "params": [row_param],
                            "lr": scale_i * lr_ratio * actual_lr,
                            "weight_decay": 0.01,
                            "_is_B_group": True,
                        })
                else:
                    param_groups.append({
                        "params": [m.lora_B],
                        "lr": lr_ratio * actual_lr,
                        "weight_decay": 0.01,
                        "_is_B_group": True,
                    })
            if cls_params:
                param_groups.append(
                    {"params": cls_params, "lr": actual_lr, "weight_decay": 0.01}
                )
            optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
            n_b = sum(1 for g in param_groups if g.get("_is_B_group"))
            print(f"  [OPT] EigenLoRA: A={actual_lr:.2e}, {n_b} B groups (per-rank)")

        elif lr_ratio > 1:
            A_params   = [p for n, p in model.named_parameters()
                          if p.requires_grad and "lora_A" in n]
            B_params   = [p for n, p in model.named_parameters()
                          if p.requires_grad and "lora_B" in n]
            cls_params = [p for n, p in model.named_parameters()
                          if p.requires_grad and "classifier" in n]
            param_groups = [
                {"params": A_params, "lr": actual_lr, "weight_decay": 0.01},
                {"params": B_params, "lr": lr_ratio * actual_lr,
                 "weight_decay": 0.01, "_is_B_group": True},
            ]
            if cls_params:
                param_groups.append(
                    {"params": cls_params, "lr": actual_lr, "weight_decay": 0.01}
                )
            optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
            print(f"  [OPT] Differential LR: A={actual_lr:.2e} B={lr_ratio*actual_lr:.2e}")

        else:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params, lr=actual_lr,
                betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
            )
            print(f"  [OPT] Uniform LR: {actual_lr:.2e}")

        if do_pivot:
            warm_cb = PivotCallback(
                pivot_epoch=PIVOT_EPOCH,
                k_fraction=PIVOT_K_FRACTION,
                warmup_steps=PIVOT_WARMUP_STEPS,
                lr_ratio=lr_ratio,
                base_lr=actual_lr,
            )
            warm_cb._optimizer = optimizer
            cbs.append(warm_cb)

        if method == "lora_plus":
            num_samples_lp  = len(train_ds)
            steps_per_ep_lp = max(1, (num_samples_lp + actual_batch - 1) // actual_batch)
            total_steps_lp  = steps_per_ep_lp * actual_epochs
            lp_cb = LoRAPlusCosineCallback(optimizer, total_steps_lp)
            cbs.append(lp_cb)
            print(f"  [LoRA+] Cosine decay over {total_steps_lp} steps")

        if cfg.get("sr_lora", False):
            num_samples    = len(train_ds)
            steps_per_epoch = max(1, (num_samples + actual_batch - 1) // actual_batch)
            total_steps    = steps_per_epoch * actual_epochs
            sr_interval    = max(5, total_steps // 10)
            sr_cb = SRLoRACallback(interval_steps=sr_interval, fuse_frac=0.25)
            cbs.append(sr_cb)
            print(f"  [SRLoRA] Recomposition every {sr_interval} steps "
                  f"(total={total_steps}, fuse_frac=0.25)")

    _ta_kwargs = dict(
        output_dir=f"tmp/{task}_{method}_{seed}",
        learning_rate=actual_lr,
        num_train_epochs=actual_epochs,
        per_device_train_batch_size=actual_batch,
        gradient_accumulation_steps=2 if shot == "full" else 1,
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        seed=seed,
        dataloader_num_workers=4,
        dataloader_persistent_workers=True,
        fp16=True,
        tf32=False,
        disable_tqdm=True,
    )
    try:
        training_args = TrainingArguments(eval_strategy="epoch", **_ta_kwargs)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy="epoch", **_ta_kwargs)

    val_ds_for_metric = val_ds if info.get("task_type") == "qa" else None
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=get_compute_metrics(info, val_ds_raw=val_ds_for_metric),
        callbacks=cbs,
        optimizers=(optimizer, None),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train()
    wall_secs = time.perf_counter() - t0
    peak_mb = (torch.cuda.max_memory_allocated() / 1024**2
               if torch.cuda.is_available() else float("nan"))

    best_metric = 0.0
    metric_key = f"eval_{info['metric']}"
    for log in trainer.state.log_history:
        if metric_key in log:
            best_metric = max(best_metric, log[metric_key])

    print(f"  [RESULT] Best {info['metric']}: {best_metric:.4f} | "
          f"Wall: {wall_secs:.1f}s | Peak VRAM: {peak_mb:.0f} MB")
    return best_metric, wall_secs, peak_mb


# ==================== STUDY ====================
def run_study(tasks=None, shots=None, methods=None):
    TASKS   = tasks   or ["sst2", "cola", "mnli", "qnli", "rte", "agnews"]
    SHOTS   = shots   or [16, 32, 64, 128, 256]
    METHODS = methods or [
        "dora", "adalora", "pissa", "lora_ga", "lora", "lora_plus",
        "lda_lora", "eigenlora", "pivotlora", "eva", "srlora", "loram", "lora_one",
    ]

    import os as _os

    results_csv = "results.csv"
    csv_fields  = ["task", "shot", "method", "seed", "score"]

    completed = {}
    if _os.path.exists(results_csv):
        with open(results_csv, newline="") as f:
            for row in csv.DictReader(f):
                raw_shot = row["shot"]
                shot_key = "full" if raw_shot == "full" else int(raw_shot)
                key = (row["task"], shot_key, row["method"], int(row["seed"]))
                try:
                    completed[key] = float(row["score"])
                except ValueError:
                    pass
        print(f"[RESUME] Loaded {len(completed)} completed results from {results_csv}")

    csv_file = open(results_csv, "a", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if _os.path.getsize(results_csv) == 0:
        csv_writer.writeheader()

    all_results = {}
    cost_stats  = {m: [] for m in METHODS}

    for task in TASKS:
        for shot in SHOTS:
            print(f"\n>>> GROUP: {task.upper()} {shot}-shot <<<")
            for method in METHODS:
                scores = []
                for seed in SEED_LIST:
                    key = (task, shot, method, seed)
                    if key in completed:
                        scores.append(completed[key])
                        print(f"  [SKIP] {task}-{shot}-{method}-{seed}: {completed[key]:.4f} (cached)")
                        continue
                    try:
                        s, wall, peak = run_experiment(task, shot, seed, method)
                        scores.append(s)
                        completed[key] = s
                        cost_stats[method].append((wall, peak))
                        csv_writer.writerow({"task": task, "shot": shot,
                                             "method": method, "seed": seed,
                                             "score": s})
                        csv_file.flush()
                    except Exception as e:
                        print(f"CRASH: {task}-{shot}-{method}-{seed}: {e}")
                        import traceback; traceback.print_exc()
                mu  = np.mean(scores) if scores else float("nan")
                std = np.std(scores)  if scores else float("nan")
                all_results[(task, shot, method)] = (mu, std)
                print(f">>> SUMMARY {method} {shot}-shot: {mu:.4f} ± {std:.4f}")

    csv_file.close()

    # Print results
    col_w = 20
    print("\n" + "="*120)
    print("FINAL RESULTS")
    print("="*120)

    for task in TASKS:
        info = TASK_INFO[task]
        print(f"\n{'─'*120}")
        print(f"  TASK: {task.upper()} | Metric: {info['metric'].upper()}")
        print(f"{'─'*120}")
        header = f"  {'Method':<24}" + "".join(f"{str(s)+'-Shot':^{col_w}}" for s in SHOTS)
        print(header)
        print(f"  {'─'*24}" + f"{'─'*col_w}" * len(SHOTS))
        for method in METHODS:
            row = f"  {method:<24}"
            for shot in SHOTS:
                mu, std = all_results.get((task, shot, method), (float("nan"), float("nan")))
                cell = f"{mu:.4f}±{std:.4f}" if not np.isnan(mu) else "CRASH"
                row += f"{cell:^{col_w}}"
            print(row)

    print("="*120)


# ==================== FULL-DATA REFERENCE ====================
def run_full_reference(tasks=None):
    import os as _os
    REF_TASKS   = tasks or ["sst2", "cola", "mnli", "qnli", "rte", "agnews"]
    REF_SEED    = 42
    REF_METHODS = ["lora", "pivotlora"]
    REF_CSV     = "results_full_reference.csv"
    REF_FIELDS  = ["task", "shot", "method", "seed", "score"]

    completed_ref = {}
    if _os.path.exists(REF_CSV):
        with open(REF_CSV, newline="") as f:
            for row in csv.DictReader(f):
                key = (row["task"], row["shot"], row["method"], int(row["seed"]))
                try:
                    completed_ref[key] = float(row["score"])
                except ValueError:
                    pass
        print(f"[FULL-REF] Loaded {len(completed_ref)} cached rows from {REF_CSV}")

    ref_fh = open(REF_CSV, "a", newline="")
    ref_w  = csv.DictWriter(ref_fh, fieldnames=REF_FIELDS)
    if _os.path.getsize(REF_CSV) == 0:
        ref_w.writeheader()

    full_ref_results = {}

    for method in REF_METHODS:
        for task in REF_TASKS:
            key = (task, "full", method, REF_SEED)
            if key in completed_ref:
                score = completed_ref[key]
                print(f"  [SKIP] {task}: {score:.4f} (cached)")
                full_ref_results[(task, method)] = score
                continue
            try:
                score, wall, _ = run_experiment(task, "full", REF_SEED, method)
                full_ref_results[(task, method)] = score
                completed_ref[key] = score
                ref_w.writerow({"task": task, "shot": "full",
                                "method": method, "seed": REF_SEED,
                                "score": score})
                ref_fh.flush()
                print(f"  [DONE] {task}: {score:.4f}  ({wall:.1f}s)")
            except Exception as e:
                print(f"  CRASH {task} {method}: {e}")
                import traceback; traceback.print_exc()
                full_ref_results[(task, method)] = float("nan")

    ref_fh.close()

    print(f"\n{'='*80}")
    print("FULL-DATA REFERENCE TABLE")
    print(f"{'='*80}")
    for task in REF_TASKS:
        info    = TASK_INFO[task]
        r_score = full_ref_results.get((task, "lora"),     float("nan"))
        w_score = full_ref_results.get((task, "pivotlora"), float("nan"))
        gain    = f"{w_score - r_score:+.4f}" if not np.isnan(r_score) and not np.isnan(w_score) else "N/A"
        print(f"  {task:<12} {info['metric']:<12} lora={r_score:.4f}  pivotlora={w_score:.4f}  gain={gain}")
    print(f"{'='*80}")
    return full_ref_results


if __name__ == "__main__":
    import argparse
    import sys
    # The summary tables use box-drawing characters; avoid cp1252 errors on Windows consoles.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke_methods", nargs="+", default=["lora", "eigenlora"])
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--shots", nargs="+", type=int, default=None)
    parser.add_argument("--full_ref", action="store_true")
    args = parser.parse_args()

    if args.seeds is not None:
        SEED_LIST[:] = SEED_LIST[:args.seeds]
        print(f"[CONFIG] Using {len(SEED_LIST)} seed(s): {SEED_LIST}")

    if args.smoke:
        EPOCHS = 1
        SEED_LIST[:] = [42]
        run_study(tasks=["sst2"], shots=[16], methods=args.smoke_methods)
    elif args.full_ref:
        run_full_reference(tasks=args.tasks)
    else:
        run_study(tasks=args.tasks, shots=args.shots, methods=args.methods)