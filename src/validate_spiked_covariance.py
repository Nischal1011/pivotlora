from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import subspace_angles
from sklearn.utils.extmath import randomized_svd


# ============================================================
# Global configuration
# ============================================================

SEED = 42

D = 768
R = 8

SYNTHETIC_SEEDS = 10
BERT_SUBSAMPLE_SEEDS = 10

# These are excess spike strengths above the isotropic noise floor.
SPIKE_STRENGTHS = np.array(
    [8.0, 6.0, 4.5, 3.5, 2.5, 2.0, 1.5, 1.0],
    dtype=np.float64,
)

NOISE_STD = 1.0

# Use disjoint reference and sampling pools.
BERT_REFERENCE_SIZE = 5000
BERT_POOL_SIZE = 5000

# hidden_states[0] is the embedding output and hidden_states[12]
# is the output of the final encoder block in BERT-base.
BERT_HIDDEN_STATE_INDEX = 12

MAX_LENGTH = 128
BERT_BATCH_SIZE = 64

FIGURE_DIR = Path("figures")
CACHE_DIR = Path("cache")


# ============================================================
# General utilities
# ============================================================

def top_r_right_singular_vectors(
    X: np.ndarray,
    r: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Return the top-r right singular vectors of a mean-centered matrix.

    Parameters
    ----------
    X:
        Array of shape (n, d). Centering is performed internally.
    r:
        Number of right singular vectors.
    seed:
        Random seed for randomized SVD.

    Returns
    -------
    V:
        Array of shape (d, r) with orthonormal columns.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {X.shape}.")

    n, d = X.shape

    # Centering reduces rank by at least one, so require n - 1 >= r.
    if n <= r:
        raise ValueError(
            f"Need n > r after centering; received n={n}, r={r}."
        )

    X_centered = X - X.mean(axis=0, keepdims=True)

    _, _, Vt = randomized_svd(
        X_centered,
        n_components=r,
        n_iter=7,
        random_state=seed,
    )

    V = Vt.T

    # Numerical orthonormalization.
    V, _ = np.linalg.qr(V)

    return V[:, :r]


def subspace_metrics(
    V_est: np.ndarray,
    V_ref: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute three subspace-error metrics.

    Returns
    -------
    normalized_frobenius_sin_theta:
        ||sin Theta||_F / sqrt(r), in [0, 1].

    mean_principal_angle_degrees:
        Arithmetic mean of principal angles in degrees.

    maximum_principal_angle_degrees:
        Largest principal angle in degrees.
    """
    if V_est.shape != V_ref.shape:
        raise ValueError(
            f"Subspaces must have matching shapes, got "
            f"{V_est.shape} and {V_ref.shape}."
        )

    r = V_est.shape[1]
    angles = subspace_angles(V_est, V_ref)

    normalized_error = (
        np.linalg.norm(np.sin(angles), ord=2) / np.sqrt(r)
    )
    mean_angle = float(np.degrees(angles).mean())
    max_angle = float(np.degrees(angles).max())

    return float(normalized_error), mean_angle, max_angle


def allocation_by_class(
    total: int,
    classes: np.ndarray,
) -> Dict[int, int]:
    """
    Allocate approximately equal counts across classes.

    Any remainder is assigned to the first classes in sorted order.
    """
    classes = np.sort(np.asarray(classes))
    n_classes = len(classes)

    base = total // n_classes
    remainder = total % n_classes

    return {
        int(c): base + (i < remainder)
        for i, c in enumerate(classes)
    }


def balanced_sample_indices(
    labels: np.ndarray,
    total: int,
    seed: int,
) -> np.ndarray:
    """
    Select an approximately class-balanced sample without replacement.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    allocation = allocation_by_class(total, classes)

    rng = np.random.RandomState(seed)
    selected = []

    for c in classes:
        candidates = np.flatnonzero(labels == c)
        needed = allocation[int(c)]

        if len(candidates) < needed:
            raise ValueError(
                f"Class {c} has {len(candidates)} examples, "
                f"but {needed} were requested."
            )

        chosen = rng.choice(candidates, size=needed, replace=False)
        selected.extend(chosen.tolist())

    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)

    return selected


def balanced_disjoint_indices(
    labels: np.ndarray,
    n_reference: int,
    n_pool: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct disjoint, approximately class-balanced reference and
    subsampling pools.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)

    ref_allocation = allocation_by_class(n_reference, classes)
    pool_allocation = allocation_by_class(n_pool, classes)

    rng = np.random.RandomState(seed)

    reference_indices = []
    pool_indices = []

    for c in classes:
        candidates = np.flatnonzero(labels == c).copy()
        rng.shuffle(candidates)

        n_ref_c = ref_allocation[int(c)]
        n_pool_c = pool_allocation[int(c)]
        required = n_ref_c + n_pool_c

        if len(candidates) < required:
            raise ValueError(
                f"Class {c} has {len(candidates)} examples, but "
                f"{required} are required for the disjoint pools."
            )

        reference_indices.extend(candidates[:n_ref_c].tolist())
        pool_indices.extend(
            candidates[n_ref_c:n_ref_c + n_pool_c].tolist()
        )

    reference_indices = np.asarray(reference_indices, dtype=np.int64)
    pool_indices = np.asarray(pool_indices, dtype=np.int64)

    rng.shuffle(reference_indices)
    rng.shuffle(pool_indices)

    if np.intersect1d(reference_indices, pool_indices).size != 0:
        raise RuntimeError("Reference and subsampling pools overlap.")

    return reference_indices, pool_indices


# ============================================================
# Part 1: Synthetic spiked-covariance experiment
# ============================================================

def generate_spiked_data(
    d: int,
    n: int,
    r: int,
    spike_strengths: np.ndarray,
    noise_std: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Gaussian data from

        Sigma = V diag(a_1, ..., a_r) V^T + sigma^2 I,

    where `spike_strengths` contains the excess spike eigenvalues a_i.

    Therefore:
        signal-direction eigenvalues = sigma^2 + a_i,
        noise eigenvalues = sigma^2,
        eigengap = a_r.

    Returns
    -------
    X:
        Data matrix of shape (n, d).

    V_true:
        Population signal subspace of shape (d, r).
    """
    spike_strengths = np.asarray(spike_strengths, dtype=np.float64)

    if len(spike_strengths) != r:
        raise ValueError(
            f"Expected {r} spike strengths, got "
            f"{len(spike_strengths)}."
        )

    if np.any(spike_strengths <= 0):
        raise ValueError("Excess spike strengths must be positive.")

    rng = np.random.RandomState(seed)

    V_true, _ = np.linalg.qr(rng.randn(d, r))
    V_true = V_true[:, :r]

    Z_signal = rng.randn(n, r)
    Z_noise = rng.randn(n, d)

    signal = (
        Z_signal
        @ np.diag(np.sqrt(spike_strengths))
        @ V_true.T
    )
    noise = noise_std * Z_noise

    X = signal + noise

    return X, V_true


def synthetic_population_parameters(
    d: int,
    spike_strengths: np.ndarray,
    noise_std: float,
) -> Dict[str, float]:
    """
    Compute population quantities for the synthetic covariance.
    """
    sigma2 = noise_std ** 2
    spike_strengths = np.asarray(spike_strengths)

    eigengap = float(spike_strengths[-1])
    operator_norm = float(sigma2 + spike_strengths[0])
    trace = float(d * sigma2 + spike_strengths.sum())
    effective_rank = trace / operator_norm

    return {
        "noise_variance": sigma2,
        "eigengap": eigengap,
        "operator_norm": operator_norm,
        "trace": trace,
        "effective_rank": effective_rank,
    }


def regime_rate_proxy(
    n: int,
    operator_norm: float,
    eigengap: float,
    effective_rank: float,
) -> float:
    """
    Unit-constant proxy for the normalized Frobenius/operator-norm
    Davis-Kahan regime expression:

        (||Sigma||_op / delta)
        * (sqrt(d_eff / N) + d_eff / N).

    This is NOT treated as a numerically calibrated theorem bound
    because the concentration and perturbation inequalities contain
    unspecified constants.
    """
    rate = (
        operator_norm
        / eigengap
        * (
            np.sqrt(effective_rank / n)
            + effective_rank / n
        )
    )

    return float(rate)


def run_synthetic_experiment():
    """
    Measure subspace estimation error over a range of d/N values.
    """
    d = D
    r = R

    # Avoid n <= r because centering limits matrix rank to n - 1.
    n_values = [
        12, 16, 24, 32, 48, 64, 96, 128, 192,
        256, 384, 512, 768, 1024, 2048, 4096, 8192,
    ]

    population = synthetic_population_parameters(
        d=d,
        spike_strengths=SPIKE_STRENGTHS,
        noise_std=NOISE_STD,
    )

    print("=" * 72)
    print("Part 1: Synthetic spiked-covariance experiment")
    print("=" * 72)
    print(f"d                         : {d}")
    print(f"r                         : {r}")
    print(f"excess spike strengths    : {SPIKE_STRENGTHS}")
    print(f"noise variance            : {population['noise_variance']:.4f}")
    print(f"population eigengap       : {population['eigengap']:.4f}")
    print(f"operator norm             : {population['operator_norm']:.4f}")
    print(f"trace                     : {population['trace']:.4f}")
    print(f"effective rank            : {population['effective_rank']:.4f}")
    print()

    results = {}

    for n in n_values:
        dn = d / n

        normalized_errors = []
        mean_angles = []
        maximum_angles = []

        for seed in range(SYNTHETIC_SEEDS):
            X, V_true = generate_spiked_data(
                d=d,
                n=n,
                r=r,
                spike_strengths=SPIKE_STRENGTHS,
                noise_std=NOISE_STD,
                seed=seed,
            )

            V_est = top_r_right_singular_vectors(
                X=X,
                r=r,
                seed=seed,
            )

            error, mean_angle, max_angle = subspace_metrics(
                V_est=V_est,
                V_ref=V_true,
            )

            normalized_errors.append(error)
            mean_angles.append(mean_angle)
            maximum_angles.append(max_angle)

        rate_proxy = regime_rate_proxy(
            n=n,
            operator_norm=population["operator_norm"],
            eigengap=population["eigengap"],
            effective_rank=population["effective_rank"],
        )

        results[dn] = {
            "n": n,
            "normalized_error": normalized_errors,
            "mean_angle": mean_angles,
            "maximum_angle": maximum_angles,
            "regime_rate_proxy": rate_proxy,
        }

        print(
            f"d/N={dn:>7.2f}  N={n:>5d}  "
            f"normalized error="
            f"{np.mean(normalized_errors):.3f} "
            f"± {np.std(normalized_errors):.3f}  "
            f"mean angle="
            f"{np.mean(mean_angles):.1f}° "
            f"± {np.std(mean_angles):.1f}°  "
            f"unit-constant rate proxy={rate_proxy:.3f}"
        )

    print()
    print(
        "Note: The rate proxy is reported only as a diagnostic. "
        "It is not plotted or treated as a calibrated upper bound."
    )

    return results, population


# ============================================================
# Part 2: Real BERT-base CLS activations
# ============================================================

def extract_cls_activations(
    texts: Iterable[str],
    tokenizer,
    model,
    device: str,
    hidden_state_index: int,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """
    Extract CLS representations at a specified BERT hidden-state index.
    """
    import torch

    texts = list(texts)
    all_cls = []

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = outputs.hidden_states

            if not -len(hidden_states) <= hidden_state_index < len(hidden_states):
                raise ValueError(
                    f"hidden_state_index={hidden_state_index} is invalid; "
                    f"model returned {len(hidden_states)} hidden states."
                )

            cls_hidden = hidden_states[
                hidden_state_index
            ][:, 0, :]

            all_cls.append(
                cls_hidden.detach().float().cpu().numpy()
            )

    return np.concatenate(all_cls, axis=0)


def load_or_extract_task_activations(
    task_name: str,
    dataset_name: str,
    dataset_config: Optional[str],
    text_key: str,
    label_key: str,
    reference_size: int,
    pool_size: int,
    tokenizer,
    model,
    device: str,
    seed: int,
):
    """
    Load cached activations or construct disjoint class-balanced
    reference and subsampling activation pools.
    """
    from datasets import load_dataset

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    safe_task = (
        task_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    cache_path = CACHE_DIR / (
        f"{safe_task}"
        f"_ref{reference_size}"
        f"_pool{pool_size}"
        f"_layer{BERT_HIDDEN_STATE_INDEX}"
        f"_maxlen{MAX_LENGTH}.npz"
    )

    if cache_path.exists():
        print(f"    Loading cached activations: {cache_path}")

        cache = np.load(cache_path)

        return {
            "X_reference": cache["X_reference"],
            "y_reference": cache["y_reference"],
            "X_pool": cache["X_pool"],
            "y_pool": cache["y_pool"],
        }

    if dataset_config is None:
        dataset = load_dataset(
            dataset_name,
            split="train",
        )
    else:
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split="train",
        )

    labels = np.asarray(dataset[label_key])

    reference_indices, pool_indices = balanced_disjoint_indices(
        labels=labels,
        n_reference=reference_size,
        n_pool=pool_size,
        seed=seed,
    )

    combined_indices = np.concatenate(
        [reference_indices, pool_indices]
    )

    selected_dataset = dataset.select(combined_indices.tolist())
    selected_texts = selected_dataset[text_key]

    print(
        f"    Extracting {len(selected_texts)} disjoint-pool "
        f"CLS representations..."
    )

    X_combined = extract_cls_activations(
        texts=selected_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        hidden_state_index=BERT_HIDDEN_STATE_INDEX,
        batch_size=BERT_BATCH_SIZE,
        max_length=MAX_LENGTH,
    )

    X_reference = X_combined[:reference_size]
    X_pool = X_combined[reference_size:]

    y_reference = labels[reference_indices]
    y_pool = labels[pool_indices]

    if X_reference.shape[0] != reference_size:
        raise RuntimeError("Reference activation size mismatch.")

    if X_pool.shape[0] != pool_size:
        raise RuntimeError("Subsampling-pool activation size mismatch.")

    np.savez_compressed(
        cache_path,
        X_reference=X_reference,
        y_reference=y_reference,
        X_pool=X_pool,
        y_pool=y_pool,
    )

    print(f"    Cached activations at: {cache_path}")

    return {
        "X_reference": X_reference,
        "y_reference": y_reference,
        "X_pool": X_pool,
        "y_pool": y_pool,
    }


def run_real_bert_experiment():
    """
    Compare few-shot BERT CLS PCA subspaces with a large-sample
    reference PCA subspace estimated from a disjoint pool.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        from datasets import load_dataset  # noqa: F401
    except ImportError as exc:
        print()
        print(
            "Skipping real BERT experiment because an optional "
            f"dependency is unavailable: {exc}"
        )
        return None

    print()
    print("=" * 72)
    print("Part 2: Real BERT-base CLS experiment")
    print("=" * 72)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device                    : {device}")
    print(f"Hidden-state index        : {BERT_HIDDEN_STATE_INDEX}")
    print(f"Reference-pool size       : {BERT_REFERENCE_SIZE}")
    print(f"Subsampling-pool size     : {BERT_POOL_SIZE}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased"
    )
    model = AutoModel.from_pretrained(
        "bert-base-uncased"
    )
    model = model.to(device)
    model.eval()

    task_specs = [
        {
            "task_name": "AG News",
            "dataset_name": "ag_news",
            "dataset_config": None,
            "text_key": "text",
            "label_key": "label",
        },
        {
            "task_name": "SST-2",
            "dataset_name": "glue",
            "dataset_config": "sst2",
            "text_key": "sentence",
            "label_key": "label",
        },
    ]

    n_values = [
        16, 32, 64, 128, 256, 512, 1024, 2048
    ]
    n_values = [
        n for n in n_values
        if n <= BERT_POOL_SIZE and n > R
    ]

    results = {}

    for task in task_specs:
        task_name = task["task_name"]

        print(f"Task: {task_name}")

        activations = load_or_extract_task_activations(
            task_name=task_name,
            dataset_name=task["dataset_name"],
            dataset_config=task["dataset_config"],
            text_key=task["text_key"],
            label_key=task["label_key"],
            reference_size=BERT_REFERENCE_SIZE,
            pool_size=BERT_POOL_SIZE,
            tokenizer=tokenizer,
            model=model,
            device=device,
            seed=SEED,
        )

        X_reference = activations["X_reference"]
        X_pool = activations["X_pool"]
        y_pool = activations["y_pool"]

        V_reference = top_r_right_singular_vectors(
            X=X_reference,
            r=R,
            seed=SEED,
        )

        task_results = {}

        for n in n_values:
            dn = D / n

            normalized_errors = []
            mean_angles = []
            maximum_angles = []

            for subsample_seed in range(BERT_SUBSAMPLE_SEEDS):
                selected = balanced_sample_indices(
                    labels=y_pool,
                    total=n,
                    seed=subsample_seed,
                )

                X_subsample = X_pool[selected]

                V_est = top_r_right_singular_vectors(
                    X=X_subsample,
                    r=R,
                    seed=subsample_seed,
                )

                error, mean_angle, max_angle = subspace_metrics(
                    V_est=V_est,
                    V_ref=V_reference,
                )

                normalized_errors.append(error)
                mean_angles.append(mean_angle)
                maximum_angles.append(max_angle)

            task_results[dn] = {
                "n": n,
                "normalized_error": normalized_errors,
                "mean_angle": mean_angles,
                "maximum_angle": maximum_angles,
            }

            print(
                f"    d/N={dn:>6.2f}  N={n:>4d}  "
                f"normalized error="
                f"{np.mean(normalized_errors):.3f} "
                f"± {np.std(normalized_errors):.3f}  "
                f"mean angle="
                f"{np.mean(mean_angles):.1f}° "
                f"± {np.std(mean_angles):.1f}°"
            )

        results[task_name] = task_results
        print()

    del model

    if device == "cuda":
        torch.cuda.empty_cache()

    return results


# ============================================================
# Plotting
# ============================================================

def means_and_stds(
    results: Dict,
    metric: str,
):
    """
    Extract sorted d/N values and metric summaries.
    """
    dns = sorted(results.keys())

    means = [
        np.mean(results[dn][metric])
        for dn in dns
    ]
    stds = [
        np.std(results[dn][metric])
        for dn in dns
    ]

    return (
        np.asarray(dns),
        np.asarray(means),
        np.asarray(stds),
    )


def make_figure(
    synthetic_results: Dict,
    real_results: Optional[Dict],
):
    """
    Generate corrected three-panel figure.

    No fitted theoretical curve is included.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 15,
        "axes.titlesize": 17,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "lines.linewidth": 2.8,
        "lines.markersize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5.0),
        sharey=True,
    )

    c_synthetic = "#1f77b4"
    c_agnews = "#ff7f0e"
    c_sst2 = "#2ca02c"
    c_boundary = "#d62728"

    def format_axis(ax):
        ax.set_xscale("log")
        ax.set_xlim(0.08, 120)
        ax.set_ylim(0.0, 1.02)
        ax.set_yticks(
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        )
        ax.set_xlabel(
            r"Dimension-to-sample ratio ($d/N$)"
        )
        ax.grid(
            True,
            which="major",
            linestyle="--",
            alpha=0.35,
            linewidth=1.0,
        )

        # This is an empirical visual guide, not a theorem threshold.
        ax.axvline(
            x=10,
            color=c_boundary,
            linestyle=":",
            linewidth=2.2,
            alpha=0.75,
        )

    # --------------------------------------------------------
    # Panel (a): Synthetic
    # --------------------------------------------------------

    ax = axes[0]

    dns, means, stds = means_and_stds(
        synthetic_results,
        metric="normalized_error",
    )

    ax.errorbar(
        dns,
        means,
        yerr=stds,
        fmt="o-",
        color=c_synthetic,
        markeredgecolor="white",
        markeredgewidth=1.3,
        capsize=4,
        elinewidth=1.8,
        label="Synthetic",
    )

    format_axis(ax)

    ax.set_ylabel(
        r"Normalized subspace error "
        r"$\|\sin\Theta\|_F/\sqrt{r}$"
    )
    ax.set_title(
        "(a) Synthetic spiked covariance",
        pad=14,
    )
    ax.legend(
        loc="upper left",
        framealpha=0.9,
        edgecolor="none",
    )

    # --------------------------------------------------------
    # Panel (b): Real BERT
    # --------------------------------------------------------

    ax = axes[1]

    if real_results is not None:
        task_styles = [
            ("AG News", c_agnews, "s"),
            ("SST-2", c_sst2, "^"),
        ]

        for task_name, color, marker in task_styles:
            if task_name not in real_results:
                continue

            dns_real, means_real, stds_real = means_and_stds(
                real_results[task_name],
                metric="normalized_error",
            )

            ax.errorbar(
                dns_real,
                means_real,
                yerr=stds_real,
                fmt=f"{marker}-",
                color=color,
                markeredgecolor="white",
                markeredgewidth=1.3,
                capsize=4,
                elinewidth=1.8,
                label=task_name,
            )

    format_axis(ax)
    ax.set_title(
        "(b) BERT-base final-layer [CLS]",
        pad=14,
    )
    ax.legend(
        loc="upper left",
        framealpha=0.9,
        edgecolor="none",
    )

    # --------------------------------------------------------
    # Panel (c): Overlay
    # --------------------------------------------------------

    ax = axes[2]

    ax.plot(
        dns,
        means,
        "o-",
        color=c_synthetic,
        alpha=0.55,
        markeredgecolor="white",
        markeredgewidth=1.3,
        label="Synthetic",
    )

    if real_results is not None:
        task_styles = [
            ("AG News", c_agnews, "s"),
            ("SST-2", c_sst2, "^"),
        ]

        for task_name, color, marker in task_styles:
            if task_name not in real_results:
                continue

            dns_real, means_real, _ = means_and_stds(
                real_results[task_name],
                metric="normalized_error",
            )

            ax.plot(
                dns_real,
                means_real,
                f"{marker}-",
                color=color,
                markeredgecolor="white",
                markeredgewidth=1.3,
                label=task_name,
            )

    format_axis(ax)

    ax.axvspan(
        10,
        120,
        alpha=0.07,
        color=c_boundary,
        zorder=0,
    )

    # Add the text label with a white bounding box so it contrasts well against the pink shading
    ax.text(
        11.5,
        0.05,
        r"$d/n = 10$",
        fontsize=12,
        color=c_boundary,
        ha="left",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        }
    )

    ax.text(
        38,
        0.18,
        "High-$d/n$\nregion",
        fontsize=11,
        color=c_boundary,
        ha="center",
        va="center",
        style="italic",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.75,
        },
    )

    ax.set_title(
        "(c) Qualitative trend comparison",
        pad=14,
    )
    ax.legend(
        loc="upper left",
        framealpha=0.9,
        edgecolor="none",
    )

    plt.tight_layout(w_pad=2.0)

    pdf_path = (
        FIGURE_DIR
        / "subspace_estimation_corrected.pdf"
    )
    png_path = (
        FIGURE_DIR
        / "subspace_estimation_corrected.png"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        dpi=300,
    )
    fig.savefig(
        png_path,
        bbox_inches="tight",
        dpi=300,
    )

    plt.close(fig)

    print("=" * 72)
    print("Saved corrected figure")
    print("=" * 72)
    print(pdf_path)
    print(png_path)


# ============================================================
# Summary
# ============================================================

def print_summary(
    synthetic_results: Dict,
    real_results: Optional[Dict],
):
    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)

    print()
    print("Synthetic experiment:")

    for dn_target in [3, 6, 12, 24, 48]:
        closest = min(
            synthetic_results.keys(),
            key=lambda x: abs(x - dn_target),
        )

        record = synthetic_results[closest]

        print(
            f"  target d/N={dn_target:>2d}; "
            f"closest={closest:>6.2f}; "
            f"N={record['n']:>4d}; "
            f"normalized error="
            f"{np.mean(record['normalized_error']):.3f}; "
            f"mean angle="
            f"{np.mean(record['mean_angle']):.1f}°"
        )

    if real_results is not None:
        for task_name in ["AG News", "SST-2"]:
            if task_name not in real_results:
                continue

            print()
            print(f"{task_name}:")

            for dn in sorted(real_results[task_name]):
                record = real_results[task_name][dn]

                print(
                    f"  d/N={dn:>6.2f}; "
                    f"N={record['n']:>4d}; "
                    f"normalized error="
                    f"{np.mean(record['normalized_error']):.3f}; "
                    f"mean angle="
                    f"{np.mean(record['mean_angle']):.1f}°"
                )

    print()
    print("Permitted interpretation:")
    print(
        "  Empirical subspace error generally increases as d/N "
        "increases in both the synthetic model and BERT activations."
    )
    print(
        "  The BERT experiments show the same qualitative monotonic "
        "trend as the synthetic spiked-covariance experiment."
    )
    print(
        "  The line at d/N=10 is an empirical visual guide for the "
        "reported setup, not a universal or theoretical threshold."
    )
    print(
        "  The experiments do not claim that a numerically calibrated "
        "Davis-Kahan bound tightly predicts the observed errors."
    )


# ============================================================
# Main
# ============================================================

def main():
    np.random.seed(SEED)

    synthetic_results, population = run_synthetic_experiment()
    real_results = run_real_bert_experiment()

    make_figure(
        synthetic_results=synthetic_results,
        real_results=real_results,
    )

    print_summary(
        synthetic_results=synthetic_results,
        real_results=real_results,
    )


if __name__ == "__main__":
    main()