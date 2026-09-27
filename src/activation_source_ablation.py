"""
activation_source_ablation.py
=============================

Projection-specific activation-source ablation for PivotLoRA.

This script imports the submitted train_classification.py harness and changes
only the activation source used to initialize query/value adapters.

Configurations
--------------
1. target_only
      Query: CLS/target-token PCA
      Value: CLS/target-token PCA

   This reproduces the activation-source choice in the submitted harness.

2. all_token
      Query: PCA over all valid token activations
      Value: PCA over all valid token activations

3. q_target_v_all
      Query: CLS/target-token PCA
      Value: PCA over all valid token activations

4. q_target_v_attention
      Query: CLS/target-token PCA
      Value: PCA weighted by frozen CLS-to-token attention

The final condition directly tests whether value projections benefit from
a source that represents information distributed across multiple tokens.

Attention-weighted PCA
----------------------
For layer l and example i, let x_ij be the input activation of source
token j and let w_ij be the frozen attention from CLS to token j,
averaged across heads. Padding tokens are removed and the weights are
renormalized within each example.

The weighted covariance is

    mu = sum_ij w_ij x_ij / sum_ij w_ij

    C = sum_ij w_ij (x_ij - mu)(x_ij - mu)^T / sum_ij w_ij

The leading rank-r eigenvectors of C are used for the value projection.

Usage
-----
Probe configuration without training:

    python activation_source_ablation.py --probe

One-run smoke test:

    python activation_source_ablation.py --smoke

Minimal experiment, AG News only, 3 seeds, 3 core sources:

    python activation_source_ablation.py \
        --tasks agnews \
        --sources target_only all_token q_target_v_attention \
        --seeds 42 43 44

Recommended experiment, AG News and SST-2, all four sources:

    python activation_source_ablation.py \
        --tasks agnews sst2 \
        --sources target_only all_token q_target_v_all q_target_v_attention \
        --seeds 42 43 44

Aggregate existing results without training:

    python activation_source_ablation.py --aggregate

Outputs
-------
results_projection_source/
    projection_source_raw.csv
    projection_source_summary.csv
    projection_source_diagnostics.csv
    projection_source_table.tex
    subspace_cache/
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
from sklearn.decomposition import PCA
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    import train_classification as be
except ImportError as exc:
    raise ImportError(
        "Could not import train_classification.py. Place "
        "activation_source_ablation.py in the same directory as "
        "train_classification.py, or add that directory to PYTHONPATH."
    ) from exc


# =====================================================================
# Experiment configuration
# =====================================================================

VALID_SOURCES = [
    "target_only",
    "all_token",
    "q_target_v_all",
    "q_target_v_attention",
]

DEFAULT_TASKS = ["agnews", "sst2"]
DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_SOURCES = VALID_SOURCES.copy()

OUTPUT_DIR = Path("results_projection_source")
RAW_CSV = OUTPUT_DIR / "projection_source_raw.csv"
SUMMARY_CSV = OUTPUT_DIR / "projection_source_summary.csv"
DIAGNOSTICS_CSV = OUTPUT_DIR / "projection_source_diagnostics.csv"
LATEX_FILE = OUTPUT_DIR / "projection_source_table.tex"
CACHE_DIR = OUTPUT_DIR / "subspace_cache"

MAX_SEQ_LEN = 128


# =====================================================================
# General utilities
# =====================================================================

def ensure_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_lr(lr):
    return f"{float(lr):.0e}"


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"[WARN] No rows to write: {path}")
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(path, row, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    needs_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)

        if needs_header:
            writer.writeheader()

        writer.writerow(row)


def source_description(source):
    descriptions = {
        "target_only": "Q=CLS, V=CLS",
        "all_token": "Q=all valid tokens, V=all valid tokens",
        "q_target_v_all": "Q=CLS, V=all valid tokens",
        "q_target_v_attention": (
            "Q=CLS, V=CLS-attention-weighted valid tokens"
        ),
    }
    return descriptions[source]


def set_all_seeds(seed):
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# PCA helpers
# =====================================================================

def pad_orthonormal_basis(Q, target_rank, dimension, seed):
    """
    Deterministically complete an orthonormal basis when the available
    sample rank is smaller than the requested adapter rank.
    """
    Q = np.asarray(Q, dtype=np.float64)

    if Q.size == 0:
        Q = np.empty((dimension, 0), dtype=np.float64)

    if Q.shape[1] > 0:
        Q, _ = np.linalg.qr(Q)

    rng = np.random.default_rng(seed)

    while Q.shape[1] < target_rank:
        vector = rng.standard_normal(dimension)

        if Q.shape[1] > 0:
            vector -= Q @ (Q.T @ vector)

        norm = np.linalg.norm(vector)

        if norm < 1e-10:
            continue

        vector /= norm
        Q = np.concatenate([Q, vector[:, None]], axis=1)

    return Q[:, :target_rank]


def unweighted_pca_eigen(X, target_rank, seed):
    """
    Match the submitted PCA-eigen initialization while additionally
    returning diagnostics.
    """
    X = np.asarray(X, dtype=np.float32)
    n_samples, dimension = X.shape

    n_components = min(
        target_rank,
        n_samples - 1,
        dimension,
    )

    if n_components <= 0:
        raise ValueError(
            f"Cannot compute PCA from X with shape {X.shape}."
        )

    # Match the original harness convention.
    pca = PCA(n_components=n_components)
    pca.fit(X)

    Q = pca.components_.T
    eigenvalues = np.asarray(
        pca.explained_variance_,
        dtype=np.float64,
    )

    if n_components < target_rank:
        Q = pad_orthonormal_basis(
            Q,
            target_rank=target_rank,
            dimension=dimension,
            seed=seed,
        )

        smallest = (
            float(eigenvalues.min())
            if len(eigenvalues)
            else 1e-8
        )

        padding = np.full(
            target_rank - n_components,
            smallest * 0.1 + 1e-8,
            dtype=np.float64,
        )

        eigenvalues = np.concatenate(
            [eigenvalues, padding]
        )
    else:
        Q, _ = np.linalg.qr(Q)
        Q = Q[:, :target_rank]
        eigenvalues = eigenvalues[:target_rank]

    total_variance = float(
        np.var(X, axis=0, ddof=1).sum()
    )

    captured_variance = float(
        eigenvalues[:n_components].sum()
    )

    explained_fraction = (
        captured_variance / total_variance
        if total_variance > 1e-12
        else 0.0
    )

    return (
        Q.astype(np.float32),
        eigenvalues.astype(np.float64),
        {
            "num_vectors": int(n_samples),
            "total_weight": float(n_samples),
            "explained_fraction": explained_fraction,
            "top_eigenvalue": float(eigenvalues[0]),
            "bottom_retained_eigenvalue": float(
                eigenvalues[min(target_rank, len(eigenvalues)) - 1]
            ),
        },
    )


def weighted_pca_eigen(X, weights, target_rank, seed):
    """
    Compute PCA from a weighted covariance matrix.

    X:
        Array of valid token activations, shape [M, D].

    weights:
        Nonnegative attention weights, shape [M].

    The weights are assumed to have been normalized within each example.
    """
    X = np.asarray(X, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)

    if len(X) != len(weights):
        raise ValueError(
            f"X and weights disagree: {len(X)} vs. {len(weights)}."
        )

    positive = weights > 0
    X = X[positive]
    weights = weights[positive]

    if len(X) < 2:
        raise ValueError(
            "Attention-weighted PCA requires at least two "
            "positive-weight token activations."
        )

    weight_sum = float(weights.sum())

    if weight_sum <= 1e-12:
        raise ValueError("Attention weights sum to zero.")

    normalized_weights = weights / weight_sum

    weighted_mean = np.sum(
        normalized_weights[:, None] * X,
        axis=0,
    )

    centered = X - weighted_mean

    covariance = (
        centered.T
        @ (centered * normalized_weights[:, None])
    )

    covariance = 0.5 * (covariance + covariance.T)

    dimension = covariance.shape[0]
    retained_rank = min(target_rank, dimension)

    # Compute only the largest retained eigenpairs.
    low_index = max(0, dimension - retained_rank)

    eigenvalues, eigenvectors = scipy.linalg.eigh(
        covariance,
        subset_by_index=[low_index, dimension - 1],
        check_finite=False,
        driver="evr",
    )

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    Q = eigenvectors[:, order]

    positive_rank = int(np.sum(eigenvalues > 1e-12))
    effective_rank = min(retained_rank, positive_rank)

    if effective_rank <= 0:
        raise ValueError(
            "Weighted covariance has no positive eigenvalues."
        )

    Q = Q[:, :effective_rank]
    eigenvalues = eigenvalues[:effective_rank]

    if effective_rank < target_rank:
        Q = pad_orthonormal_basis(
            Q,
            target_rank=target_rank,
            dimension=dimension,
            seed=seed,
        )

        smallest = (
            float(eigenvalues.min())
            if len(eigenvalues)
            else 1e-8
        )

        eigenvalues = np.concatenate([
            eigenvalues,
            np.full(
                target_rank - effective_rank,
                smallest * 0.1 + 1e-8,
                dtype=np.float64,
            ),
        ])
    else:
        Q, _ = np.linalg.qr(Q)
        Q = Q[:, :target_rank]
        eigenvalues = eigenvalues[:target_rank]

    total_variance = float(np.trace(covariance))
    captured_variance = float(
        eigenvalues[:effective_rank].sum()
    )

    explained_fraction = (
        captured_variance / total_variance
        if total_variance > 1e-12
        else 0.0
    )

    return (
        Q.astype(np.float32),
        eigenvalues.astype(np.float64),
        {
            "num_vectors": int(len(X)),
            "total_weight": weight_sum,
            "explained_fraction": explained_fraction,
            "top_eigenvalue": float(eigenvalues[0]),
            "bottom_retained_eigenvalue": float(
                eigenvalues[min(target_rank, len(eigenvalues)) - 1]
            ),
        },
    )


def mean_principal_angle_degrees(Q1, Q2):
    angles = scipy.linalg.subspace_angles(
        np.asarray(Q1, dtype=np.float64),
        np.asarray(Q2, dtype=np.float64),
    )

    return float(np.degrees(angles).mean())


# =====================================================================
# Frozen activation collection
# =====================================================================

def source_cache_path(task, shot, seed):
    model_key = be.MODEL_NAME.replace("/", "_")
    return CACHE_DIR / (
        f"{task}_shot{shot}_seed{seed}_"
        f"{model_key}_rank{be.RANK}_maxlen{MAX_SEQ_LEN}.npz"
    )


def save_source_bank(path, source_bank, metadata, diagnostics):
    arrays = {
        "metadata_json": np.asarray(json.dumps(metadata)),
        "diagnostics_json": np.asarray(json.dumps(diagnostics)),
    }

    for source_name, source_data in source_bank.items():
        for layer_index, layer_data in source_data.items():
            arrays[f"{source_name}_Q_{layer_index}"] = (
                layer_data["Q"].astype(np.float32)
            )
            arrays[f"{source_name}_eigs_{layer_index}"] = (
                layer_data["eigenvalues"].astype(np.float64)
            )

    np.savez_compressed(path, **arrays)


def load_source_bank(path):
    loaded = np.load(path, allow_pickle=False)

    metadata = json.loads(
        loaded["metadata_json"].item()
    )
    diagnostics = json.loads(
        loaded["diagnostics_json"].item()
    )

    source_bank = {
        "target": {},
        "all": {},
        "attention": {},
    }

    for source_name in source_bank:
        for layer_index in range(12):
            q_key = f"{source_name}_Q_{layer_index}"
            eig_key = f"{source_name}_eigs_{layer_index}"

            source_bank[source_name][layer_index] = {
                "Q": loaded[q_key],
                "eigenvalues": loaded[eig_key],
            }

    return source_bank, metadata, diagnostics


def build_source_bank(
    model,
    dataset,
    task,
    shot,
    seed,
    batch_size,
    force_recompute=False,
):
    """
    Collect three frozen activation sources for all 12 BERT layers:

      target:
          layer-input CLS activations

      all:
          all valid layer-input token activations

      attention:
          all valid layer-input token activations plus frozen,
          head-averaged CLS-to-token attention weights
    """
    ensure_directories()
    cache_path = source_cache_path(task, shot, seed)

    if cache_path.exists() and not force_recompute:
        print(f"[SOURCE] Loading cache: {cache_path}")
        return load_source_bank(cache_path)

    print("[SOURCE] Computing frozen activation-source bank...")
    print(f"[SOURCE] Cache destination: {cache_path}")

    model.eval().to(be.DEVICE)

    # Force eager attention so output_attentions=True returns the actual
    # attention matrices rather than None under SDPA/flash backends.
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"

    target_activations = defaultdict(list)
    all_activations = defaultdict(list)
    attention_activations = defaultdict(list)
    attention_weights = defaultdict(list)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    with torch.inference_mode():
        for batch_number, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(be.DEVICE)
            attention_mask = batch["attention_mask"].to(be.DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )

            hidden_states = outputs.hidden_states
            attentions = outputs.attentions

            if hidden_states is None:
                raise RuntimeError(
                    "Model did not return hidden states."
                )

            if attentions is None or any(
                attention is None for attention in attentions
            ):
                raise RuntimeError(
                    "Model did not return attention matrices. "
                    "Ensure eager attention is enabled."
                )

            # hidden_states[0] is the embedding output and is the input
            # to encoder layer 0. hidden_states[i] is therefore the input
            # to the query/value projections in encoder layer i.
            for layer_index in range(12):
                layer_input = hidden_states[layer_index]
                valid_mask = attention_mask.bool()

                target_activations[layer_index].append(
                    layer_input[:, 0, :]
                    .detach()
                    .cpu()
                    .float()
                )

                valid_tokens = layer_input[valid_mask]

                all_activations[layer_index].append(
                    valid_tokens.detach().cpu().float()
                )

                # [batch, heads, target_position, source_position]
                layer_attention = attentions[layer_index]

                # Frozen CLS-to-token attention, averaged over heads.
                cls_attention = (
                    layer_attention[:, :, 0, :]
                    .mean(dim=1)
                )

                # Remove padding and renormalize over valid tokens.
                cls_attention = (
                    cls_attention
                    * attention_mask.to(cls_attention.dtype)
                )

                denominator = cls_attention.sum(
                    dim=1,
                    keepdim=True,
                ).clamp(min=1e-12)

                cls_attention = (
                    cls_attention / denominator
                )

                valid_weights = cls_attention[valid_mask]

                attention_activations[layer_index].append(
                    valid_tokens.detach().cpu().float()
                )
                attention_weights[layer_index].append(
                    valid_weights.detach().cpu().double()
                )

            print(
                f"  [SOURCE] batch {batch_number + 1}/"
                f"{len(loader)}"
            )

            del outputs, hidden_states, attentions

    source_bank = {
        "target": {},
        "all": {},
        "attention": {},
    }

    diagnostics = []

    for layer_index in range(12):
        print(f"[SOURCE] PCA layer {layer_index:02d}/11")

        X_target = torch.cat(
            target_activations[layer_index],
            dim=0,
        ).numpy()

        X_all = torch.cat(
            all_activations[layer_index],
            dim=0,
        ).numpy()

        X_attention = torch.cat(
            attention_activations[layer_index],
            dim=0,
        ).numpy()

        W_attention = torch.cat(
            attention_weights[layer_index],
            dim=0,
        ).numpy()

        target_Q, target_eigs, target_diag = (
            unweighted_pca_eigen(
                X_target,
                target_rank=be.RANK,
                seed=seed * 10007 + layer_index,
            )
        )

        all_Q, all_eigs, all_diag = (
            unweighted_pca_eigen(
                X_all,
                target_rank=be.RANK,
                seed=seed * 10007 + layer_index + 1000,
            )
        )

        attention_Q, attention_eigs, attention_diag = (
            weighted_pca_eigen(
                X_attention,
                W_attention,
                target_rank=be.RANK,
                seed=seed * 10007 + layer_index + 2000,
            )
        )

        source_bank["target"][layer_index] = {
            "Q": target_Q,
            "eigenvalues": target_eigs,
        }
        source_bank["all"][layer_index] = {
            "Q": all_Q,
            "eigenvalues": all_eigs,
        }
        source_bank["attention"][layer_index] = {
            "Q": attention_Q,
            "eigenvalues": attention_eigs,
        }

        target_all_angle = mean_principal_angle_degrees(
            target_Q,
            all_Q,
        )
        target_attention_angle = (
            mean_principal_angle_degrees(
                target_Q,
                attention_Q,
            )
        )
        all_attention_angle = (
            mean_principal_angle_degrees(
                all_Q,
                attention_Q,
            )
        )

        for source_name, diag in [
            ("target", target_diag),
            ("all", all_diag),
            ("attention", attention_diag),
        ]:
            diagnostics.append({
                "task": task,
                "shot": shot,
                "seed": seed,
                "layer": layer_index,
                "source": source_name,
                "num_vectors": diag["num_vectors"],
                "total_weight": diag["total_weight"],
                "explained_fraction": diag[
                    "explained_fraction"
                ],
                "top_eigenvalue": diag[
                    "top_eigenvalue"
                ],
                "bottom_retained_eigenvalue": diag[
                    "bottom_retained_eigenvalue"
                ],
                "target_all_mean_angle_deg": (
                    target_all_angle
                ),
                "target_attention_mean_angle_deg": (
                    target_attention_angle
                ),
                "all_attention_mean_angle_deg": (
                    all_attention_angle
                ),
            })

        print(
            f"  target/all angle={target_all_angle:.1f}°, "
            f"target/attention angle="
            f"{target_attention_angle:.1f}°"
        )

        del (
            X_target,
            X_all,
            X_attention,
            W_attention,
        )
        gc.collect()

    metadata = {
        "task": task,
        "shot": shot,
        "seed": seed,
        "model": be.MODEL_NAME,
        "rank": be.RANK,
        "max_seq_len": MAX_SEQ_LEN,
        "num_examples": len(dataset),
        "attention_definition": (
            "head-averaged frozen CLS-to-token attention; "
            "padding removed; weights renormalized per example"
        ),
    }

    save_source_bank(
        cache_path,
        source_bank,
        metadata,
        diagnostics,
    )

    print(f"[SOURCE] Saved cache: {cache_path}")

    # Free activation collections before adapter training.
    del (
        target_activations,
        all_activations,
        attention_activations,
        attention_weights,
    )
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return source_bank, metadata, diagnostics


# =====================================================================
# Projection-specific adapter injection
# =====================================================================

def source_assignment(source_name):
    if source_name == "target_only":
        return {
            "query": "target",
            "value": "target",
        }

    if source_name == "all_token":
        return {
            "query": "all",
            "value": "all",
        }

    if source_name == "q_target_v_all":
        return {
            "query": "target",
            "value": "all",
        }

    if source_name == "q_target_v_attention":
        return {
            "query": "target",
            "value": "attention",
        }

    raise ValueError(f"Unknown source: {source_name}")


def deterministic_rotation(
    basis,
    layer_index,
    target,
    rank,
    device,
):
    rotation_key = f"layer_{layer_index}_{target}"
    digest = hashlib.sha256(
        rotation_key.encode("utf-8")
    ).digest()
    rotation_seed = int.from_bytes(
        digest[:4],
        "little",
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(rotation_seed)

    rotation, _ = torch.linalg.qr(
        torch.randn(
            rank,
            rank,
            generator=generator,
            device=device,
        )
    )

    return basis.to(device) @ rotation


def inject_projection_specific_pivotlora(
    model,
    source_bank,
    source_name,
):
    """
    Inject the exact LoRALayer used by train_classification.py, but permit
    query and value to receive different PCA subspaces/eigenvalues.
    """
    assignment = source_assignment(source_name)

    print(
        f"[INJECT-SOURCE] {source_name}: "
        f"{source_description(source_name)}"
    )

    adapter_count = 0

    for layer_index, encoder_layer in enumerate(
        model.bert.encoder.layer
    ):
        attention_module = encoder_layer.attention.self

        for target in ["query", "value"]:
            selected_source = assignment[target]
            source_data = source_bank[
                selected_source
            ][layer_index]

            basis = torch.tensor(
                source_data["Q"],
                dtype=torch.float32,
            )

            eigenvalues = np.asarray(
                source_data["eigenvalues"],
                dtype=np.float64,
            )

            original = getattr(
                attention_module,
                target,
            )

            basis = deterministic_rotation(
                basis=basis,
                layer_index=layer_index,
                target=target,
                rank=be.RANK,
                device=original.weight.device,
            )

            eigenvalue_mean = (
                float(eigenvalues.mean()) + 1e-8
            )
            b_lr_scales = (
                eigenvalues / eigenvalue_mean
            ).astype(np.float32)

            adapter = be.LoRALayer(
                original_layer=original,
                rank=be.RANK,
                alpha=be.ALPHA,
                init_A=basis,
                init_B=None,
                track_A_init=True,
                b_lr_scales=b_lr_scales,
                is_dora=False,
            )

            setattr(
                attention_module,
                target,
                adapter,
            )

            adapter_count += 1

    # Match the submitted harness: freeze everything first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Enable adapter parameters.
    for module in model.modules():
        if not isinstance(module, be.LoRALayer):
            continue

        module.lora_A.requires_grad = True

        if module._split_b:
            for row in module.lora_B_rows:
                row.requires_grad = True
        else:
            module.lora_B.requires_grad = True

    # Enable classification head.
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    print(
        f"[INJECT-SOURCE] Injected "
        f"{adapter_count} adapters."
    )

    return model


# =====================================================================
# Optimizer
# =====================================================================

def build_pivot_optimizer(model, base_lr):
    """
    Match the EigenLoRA/PivotLoRA optimizer in train_classification.py:
      - A at base LR
      - each B row at eigenvalue-scaled 16x LR
      - classifier at base LR
    """
    A_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_A" in name
    ]

    classifier_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "classifier" in name
    ]

    parameter_groups = [
        {
            "params": A_parameters,
            "lr": base_lr,
            "weight_decay": 0.01,
        }
    ]

    b_group_count = 0

    for _, module in model.named_modules():
        if not isinstance(module, be.LoRALayer):
            continue

        if module._split_b:
            for rank_index, row_parameter in enumerate(
                module.lora_B_rows
            ):
                scale = float(
                    module.b_lr_scales[rank_index].item()
                )

                parameter_groups.append({
                    "params": [row_parameter],
                    "lr": (
                        scale
                        * 16.0
                        * base_lr
                    ),
                    "weight_decay": 0.01,
                    "_is_B_group": True,
                })

                b_group_count += 1
        else:
            parameter_groups.append({
                "params": [module.lora_B],
                "lr": 16.0 * base_lr,
                "weight_decay": 0.01,
                "_is_B_group": True,
            })

            b_group_count += 1

    if classifier_parameters:
        parameter_groups.append({
            "params": classifier_parameters,
            "lr": base_lr,
            "weight_decay": 0.01,
        })

    optimizer = torch.optim.AdamW(
        parameter_groups,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    print(
        f"[OPT] A LR={base_lr:.2e}; "
        f"{b_group_count} eigen-scaled B groups; "
        f"nominal B ratio=16"
    )

    return optimizer


# =====================================================================
# Training
# =====================================================================

def make_training_arguments(
    task,
    source_name,
    seed,
    epochs,
    batch_size,
    learning_rate,
    dataloader_workers,
):
    kwargs = dict(
        output_dir=(
            f"tmp/projection_source_"
            f"{task}_{source_name}_{seed}"
        ),
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        seed=seed,
        dataloader_num_workers=dataloader_workers,
        dataloader_persistent_workers=(
            dataloader_workers > 0
        ),
        fp16=torch.cuda.is_available(),
        tf32=False,
        disable_tqdm=True,
    )

    try:
        return TrainingArguments(
            eval_strategy="epoch",
            **kwargs,
        )
    except TypeError:
        return TrainingArguments(
            evaluation_strategy="epoch",
            **kwargs,
        )


def run_projection_source_experiment(
    task,
    shot,
    seed,
    source_name,
    learning_rate,
    epochs,
    batch_size,
    dataloader_workers,
    force_recompute_subspaces=False,
):
    set_all_seeds(seed)

    print("\n" + "=" * 80)
    print(
        f"Task={task.upper()} | shot={shot} | seed={seed} | "
        f"source={source_name} | LR={learning_rate:.2e}"
    )
    print(source_description(source_name))
    print("=" * 80)

    train_dataset, validation_dataset, task_info = (
        be.load_task_with_splits(
            task,
            shot,
            seed,
        )
    )

    model_kwargs = {
        "num_labels": task_info["num_labels"],
    }

    # Recent Transformers versions accept this argument directly.
    # Older versions are handled by the fallback.
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            be.MODEL_NAME,
            attn_implementation="eager",
            **model_kwargs,
        ).to(be.DEVICE)
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            be.MODEL_NAME,
            **model_kwargs,
        ).to(be.DEVICE)

        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"

    source_bank, _, diagnostics = build_source_bank(
        model=model,
        dataset=train_dataset,
        task=task,
        shot=shot,
        seed=seed,
        batch_size=batch_size,
        force_recompute=force_recompute_subspaces,
    )

    model = inject_projection_specific_pivotlora(
        model=model,
        source_bank=source_bank,
        source_name=source_name,
    )

    optimizer = build_pivot_optimizer(
        model=model,
        base_lr=learning_rate,
    )

    pivot_callback = be.PivotCallback(
        pivot_epoch=be.PIVOT_EPOCH,
        k_fraction=be.PIVOT_K_FRACTION,
        warmup_steps=be.PIVOT_WARMUP_STEPS,
        lr_ratio=16,
        base_lr=learning_rate,
    )
    pivot_callback._optimizer = optimizer

    training_arguments = make_training_arguments(
        task=task,
        source_name=source_name,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dataloader_workers=dataloader_workers,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=be.get_compute_metrics(task_info),
        callbacks=[pivot_callback],
        optimizers=(optimizer, None),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    trainer.train()
    wall_seconds = time.perf_counter() - start_time

    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else float("nan")
    )

    metric_key = f"eval_{task_info['metric']}"
    metric_values = [
        float(log[metric_key])
        for log in trainer.state.log_history
        if metric_key in log
    ]

    if not metric_values:
        raise RuntimeError(
            f"No {metric_key} values were recorded."
        )

    # Match submitted harness behavior.
    best_metric = max(0.0, max(metric_values))

    result = {
        "task": task,
        "shot": shot,
        "source": source_name,
        "query_source": source_assignment(
            source_name
        )["query"],
        "value_source": source_assignment(
            source_name
        )["value"],
        "seed": seed,
        "lr": learning_rate,
        "epochs": epochs,
        "score": best_metric,
        "metric": task_info["metric"],
        "wall_s": wall_seconds,
        "peak_memory_mb": peak_memory_mb,
        "pivot_done": bool(pivot_callback.pivot_done),
    }

    print(
        f"[RESULT] {task} {source_name} seed={seed}: "
        f"{best_metric:.6f} | "
        f"wall={wall_seconds:.1f}s | "
        f"pivot_done={pivot_callback.pivot_done}"
    )

    del trainer, optimizer, model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, diagnostics


# =====================================================================
# Aggregation
# =====================================================================

def load_raw_results():
    if not RAW_CSV.exists():
        return []

    with RAW_CSV.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(csv.DictReader(handle))


def completed_keys(rows):
    keys = set()

    for row in rows:
        keys.add((
            row["task"],
            int(row["shot"]),
            row["source"],
            int(row["seed"]),
            float(row["lr"]),
        ))

    return keys


def aggregate_results():
    rows = load_raw_results()

    if not rows:
        print(f"[AGGREGATE] No raw results found at {RAW_CSV}")
        return []

    grouped = defaultdict(list)

    for row in rows:
        key = (
            row["task"],
            int(row["shot"]),
            row["source"],
            row["metric"],
            float(row["lr"]),
        )

        grouped[key].append(
            float(row["score"])
        )

    summary_rows = []

    for key in sorted(grouped):
        task, shot, source, metric, lr = key
        values = np.asarray(
            grouped[key],
            dtype=np.float64,
        )

        target_values_by_seed = {
            int(row["seed"]): float(row["score"])
            for row in rows
            if (
                row["task"] == task
                and int(row["shot"]) == shot
                and row["source"] == "target_only"
                and abs(float(row["lr"]) - lr) < 1e-15
            )
        }

        source_values_by_seed = {
            int(row["seed"]): float(row["score"])
            for row in rows
            if (
                row["task"] == task
                and int(row["shot"]) == shot
                and row["source"] == source
                and abs(float(row["lr"]) - lr) < 1e-15
            )
        }

        paired_differences = []

        for seed in sorted(
            set(target_values_by_seed)
            & set(source_values_by_seed)
        ):
            paired_differences.append(
                source_values_by_seed[seed]
                - target_values_by_seed[seed]
            )

        paired_differences = np.asarray(
            paired_differences,
            dtype=np.float64,
        )

        summary_rows.append({
            "task": task,
            "shot": shot,
            "metric": metric,
            "source": source,
            "query_source": source_assignment(source)[
                "query"
            ],
            "value_source": source_assignment(source)[
                "value"
            ],
            "lr": lr,
            "n_seeds": len(values),
            "mean_score": float(np.mean(values)),
            "std_score": (
                float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0
            ),
            "minimum_score": float(np.min(values)),
            "maximum_score": float(np.max(values)),
            "mean_paired_delta_vs_target": (
                float(np.mean(paired_differences))
                if len(paired_differences)
                else float("nan")
            ),
            "wins_vs_target": (
                int(np.sum(paired_differences > 0))
                if len(paired_differences)
                else 0
            ),
            "paired_count": int(
                len(paired_differences)
            ),
        })

    write_csv(SUMMARY_CSV, summary_rows)

    print("\n" + "=" * 100)
    print("PROJECTION-SOURCE ABLATION SUMMARY")
    print("=" * 100)

    tasks = sorted(
        set(row["task"] for row in summary_rows)
    )

    sources = [
        source
        for source in VALID_SOURCES
        if any(
            row["source"] == source
            for row in summary_rows
        )
    ]

    for task in tasks:
        task_rows = {
            row["source"]: row
            for row in summary_rows
            if row["task"] == task
        }

        print(f"\nTask: {task.upper()}")
        print(
            f"{'Source':<24} "
            f"{'Mean ± SD':>18} "
            f"{'Δ vs target':>14} "
            f"{'Wins':>9}"
        )
        print("-" * 72)

        for source in sources:
            row = task_rows.get(source)

            if row is None:
                continue

            delta = row[
                "mean_paired_delta_vs_target"
            ]

            delta_text = (
                f"{delta:+.4f}"
                if np.isfinite(delta)
                else "N/A"
            )

            print(
                f"{source:<24} "
                f"{row['mean_score']:.4f} ± "
                f"{row['std_score']:.4f} "
                f"{delta_text:>14} "
                f"{row['wins_vs_target']}/"
                f"{row['paired_count']:>7}"
            )

    write_latex_table(summary_rows)
    return summary_rows


def write_latex_table(summary_rows):
    tasks = sorted(
        set(row["task"] for row in summary_rows)
    )

    sources = [
        source
        for source in VALID_SOURCES
        if any(
            row["source"] == source
            for row in summary_rows
        )
    ]

    latex_names = {
        "target_only": "Target-only",
        "all_token": "All-token",
        "q_target_v_all": "$Q$: target, $V$: all",
        "q_target_v_attention": (
            "$Q$: target, $V$: attention"
        ),
    }

    lines = [
        "\\begin{tabular}{l"
        + "c" * len(tasks)
        + "}",
        "\\toprule",
        "Activation source & "
        + " & ".join(
            task.upper() for task in tasks
        )
        + " \\\\",
        "\\midrule",
    ]

    for source in sources:
        cells = []

        for task in tasks:
            matching = [
                row
                for row in summary_rows
                if (
                    row["task"] == task
                    and row["source"] == source
                )
            ]

            if matching:
                row = matching[0]
                cells.append(
                    f"{row['mean_score']:.3f}"
                    f"$\\pm$"
                    f"{row['std_score']:.3f}"
                )
            else:
                cells.append("--")

        lines.append(
            latex_names[source]
            + " & "
            + " & ".join(cells)
            + " \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
    ])

    LATEX_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# =====================================================================
# Study driver
# =====================================================================

RAW_FIELDS = [
    "task",
    "shot",
    "source",
    "query_source",
    "value_source",
    "seed",
    "lr",
    "epochs",
    "score",
    "metric",
    "wall_s",
    "peak_memory_mb",
    "pivot_done",
]


def run_study(args):
    ensure_directories()

    existing_rows = load_raw_results()
    completed = completed_keys(existing_rows)

    diagnostics_by_key = {}

    total_requested = (
        len(args.tasks)
        * len(args.sources)
        * len(args.seeds)
    )

    completed_this_call = 0
    failed = []

    print("\n" + "=" * 80)
    print("PROJECTION-SOURCE ABLATION")
    print("=" * 80)
    print(f"Tasks:    {args.tasks}")
    print(f"Sources:  {args.sources}")
    print(f"Seeds:    {args.seeds}")
    print(f"Shot:     {args.shot}")
    print(f"LR:       {args.lr:.2e}")
    print(f"Epochs:   {args.epochs}")
    print(f"Runs:     {total_requested}")
    print("=" * 80)

    for task in args.tasks:
        if task not in be.TASK_INFO:
            raise ValueError(
                f"Unknown task {task!r}. Available tasks: "
                f"{sorted(be.TASK_INFO)}"
            )

        if be.TASK_INFO[task].get(
            "task_type",
            "cls",
        ) != "cls":
            raise ValueError(
                "This ablation currently supports discrete-label "
                "classification tasks only."
            )

        for seed in args.seeds:
            for source_name in args.sources:
                key = (
                    task,
                    args.shot,
                    source_name,
                    seed,
                    float(args.lr),
                )

                if key in completed:
                    print(
                        f"[SKIP] {task} shot={args.shot} "
                        f"source={source_name} seed={seed}"
                    )
                    continue

                try:
                    result, diagnostics = (
                        run_projection_source_experiment(
                            task=task,
                            shot=args.shot,
                            seed=seed,
                            source_name=source_name,
                            learning_rate=args.lr,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            dataloader_workers=args.workers,
                            force_recompute_subspaces=(
                                args.force_recompute_subspaces
                            ),
                        )
                    )

                    append_csv_row(
                        RAW_CSV,
                        result,
                        RAW_FIELDS,
                    )

                    completed.add(key)
                    completed_this_call += 1

                    diagnostic_key = (
                        task,
                        args.shot,
                        seed,
                    )

                    diagnostics_by_key[
                        diagnostic_key
                    ] = diagnostics

                except Exception as exc:
                    print(
                        f"[CRASH] task={task} "
                        f"source={source_name} "
                        f"seed={seed}: {exc}"
                    )
                    traceback.print_exc()

                    failed.append({
                        "task": task,
                        "shot": args.shot,
                        "source": source_name,
                        "seed": seed,
                        "error": repr(exc),
                    })

    all_diagnostics = []

    # Include diagnostics generated during this call.
    for diagnostics in diagnostics_by_key.values():
        all_diagnostics.extend(diagnostics)

    # Include diagnostics from every available cache so aggregate mode
    # remains reproducible after resumed runs.
    for cache_path in sorted(
        CACHE_DIR.glob("*.npz")
    ):
        try:
            _, _, diagnostics = load_source_bank(
                cache_path
            )
            all_diagnostics.extend(diagnostics)
        except Exception as exc:
            print(
                f"[WARN] Could not read diagnostics from "
                f"{cache_path}: {exc}"
            )

    # Deduplicate diagnostics.
    unique_diagnostics = {}

    for row in all_diagnostics:
        diagnostic_key = (
            row["task"],
            int(row["shot"]),
            int(row["seed"]),
            int(row["layer"]),
            row["source"],
        )
        unique_diagnostics[diagnostic_key] = row

    if unique_diagnostics:
        write_csv(
            DIAGNOSTICS_CSV,
            [
                unique_diagnostics[key]
                for key in sorted(unique_diagnostics)
            ],
        )

    if failed:
        write_csv(
            OUTPUT_DIR / "projection_source_failures.csv",
            failed,
        )

    print(
        f"\n[DONE] New completed runs: "
        f"{completed_this_call}"
    )
    print(f"[DONE] Failures: {len(failed)}")

    aggregate_results()


# =====================================================================
# Probe
# =====================================================================

def run_probe(args):
    ensure_directories()

    print("=" * 80)
    print("PROJECTION-SOURCE ABLATION PROBE")
    print("=" * 80)
    print(f"Harness:              {Path(be.__file__).resolve()}")
    print(f"Device:               {be.DEVICE}")
    print(f"Model:                {be.MODEL_NAME}")
    print(f"Rank:                 {be.RANK}")
    print(f"Alpha:                {be.ALPHA}")
    print(f"Base LR:              {args.lr:.2e}")
    print(f"Pivot epoch:          {be.PIVOT_EPOCH}")
    print(f"Pivot fraction:       {be.PIVOT_K_FRACTION}")
    print(f"Pivot warmup:         {be.PIVOT_WARMUP_STEPS}")
    print(f"Tasks:                {args.tasks}")
    print(f"Seeds:                {args.seeds}")
    print(f"Sources:              {args.sources}")
    print(f"Output:               {OUTPUT_DIR.resolve()}")
    print()
    print("Source definitions:")

    for source_name in args.sources:
        print(
            f"  {source_name:<24} "
            f"{source_description(source_name)}"
        )

    print()
    print("Checks:")

    required_symbols = [
        "LoRALayer",
        "PivotCallback",
        "load_task_with_splits",
        "get_compute_metrics",
        "TASK_INFO",
    ]

    for symbol in required_symbols:
        present = hasattr(be, symbol)
        marker = "OK" if present else "MISSING"
        print(f"  {symbol:<30} {marker}")

        if not present:
            raise RuntimeError(
                f"train_classification.py lacks required symbol "
                f"{symbol!r}."
            )

    for source_name in args.sources:
        if source_name not in VALID_SOURCES:
            raise ValueError(
                f"Unknown source: {source_name}"
            )

    print("\nProbe passed.")
    print("=" * 80)


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Projection-specific activation-source ablation "
            "for the submitted PivotLoRA BERT harness."
        )
    )

    parser.add_argument(
        "--probe",
        action="store_true",
        help="Validate imports and print the experiment configuration.",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run SST-2, 16-shot, seed 42, one epoch, "
            "using all source configurations."
        ),
    )

    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate existing raw results without training.",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        choices=sorted(be.TASK_INFO.keys()),
    )

    parser.add_argument(
        "--sources",
        nargs="+",
        default=DEFAULT_SOURCES,
        choices=VALID_SOURCES,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--shot",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=be.LEARNING_RATE,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=be.EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=be.BATCH_SIZE,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--force-recompute-subspaces",
        action="store_true",
        help="Ignore cached activation subspaces.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    ensure_directories()

    if args.aggregate:
        aggregate_results()
        return

    if args.probe:
        run_probe(args)
        return

    if args.smoke:
        args.tasks = ["sst2"]
        args.sources = VALID_SOURCES.copy()
        args.seeds = [42]
        args.shot = 16
        args.epochs = 1
        args.workers = 0

    run_study(args)


if __name__ == "__main__":
    main()
