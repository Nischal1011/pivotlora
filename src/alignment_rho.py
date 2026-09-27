import argparse
import csv
import json
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------

MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LEN = 128
RANK = 8
REQUESTED_SHOTS = 64
PROVISIONAL_ALPHA_INDICATOR = 0.6

# Ten independently sampled few-shot subsets.
DEFAULT_SUBSET_SEEDS = [
    42, 43, 44, 45, 46,
    47, 48, 49, 50, 51,
]

# Existing results from the manuscript.
# Gain is expressed as a fraction:
#     0.193 = 19.3 percentage points.
PAPER_RESULTS = {
    "AG News": {
        "alpha": 0.843,
        "gain": 0.193,
    },
    "SST-2": {
        "alpha": 0.667,
        "gain": 0.143,
    },
    "QNLI": {
        "alpha": 0.458,
        "gain": 0.039,
    },
    "MNLI": {
        "alpha": 0.419,
        "gain": 0.001,
    },
    "BoolQ": {
        "alpha": 0.402,
        "gain": 0.016,
    },
    "RTE": {
        "alpha": 0.293,
        "gain": 0.018,
    },
}

TASKS = {
    "SST-2": {
        "dataset": ("nyu-mll/glue", "sst2"),
        "split": "train",
        "text_fields": ["sentence"],
        "label_key": "label",
        "num_labels": 2,
    },
    "AG News": {
        "dataset": ("fancyzhx/ag_news",),
        "split": "train",
        "text_fields": ["text"],
        "label_key": "label",
        "num_labels": 4,
    },
    "MNLI": {
        "dataset": ("nyu-mll/glue", "mnli"),
        "split": "train",
        "text_fields": ["premise", "hypothesis"],
        "label_key": "label",
        "num_labels": 3,
    },
    "RTE": {
        "dataset": ("nyu-mll/glue", "rte"),
        "split": "train",
        "text_fields": ["sentence1", "sentence2"],
        "label_key": "label",
        "num_labels": 2,
    },
    "QNLI": {
        "dataset": ("nyu-mll/glue", "qnli"),
        "split": "train",
        "text_fields": ["question", "sentence"],
        "label_key": "label",
        "num_labels": 2,
    },
    "BoolQ": {
        "dataset": ("google/boolq",),
        "split": "train",
        "text_fields": ["question", "passage"],
        "label_key": "answer",
        "num_labels": 2,
    },
}

IMBALANCE_PROFILES = {
    "balanced": [16, 16, 16, 16],
    "moderate": [28, 16, 12, 8],
    "severe": [40, 12, 8, 4],
}


# ---------------------------------------------------------------------
# Reproducibility and I/O
# ---------------------------------------------------------------------

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_name(value):
    return (
        value.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"Warning: no rows to write to {path}")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_cache(path, X, y, indices, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        indices=np.asarray(indices, dtype=np.int64),
        metadata=np.asarray(json.dumps(metadata)),
    )


def load_cache(path):
    cached = np.load(path, allow_pickle=False)

    return {
        "X": cached["X"],
        "y": cached["y"],
        "indices": cached["indices"],
        "metadata": json.loads(cached["metadata"].item()),
    }


# ---------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------

def normalize_labels(values, label_key):
    if label_key == "answer":
        # BoolQ labels are Boolean values.
        return np.asarray(values, dtype=np.int64)

    return np.asarray(values, dtype=np.int64)


def get_texts(dataset_subset, task_info):
    fields = task_info["text_fields"]

    if len(fields) == 1:
        return list(dataset_subset[fields[0]])

    if fields == ["question", "passage"]:
        # Match the preprocessing in the supplied verification script.
        return [
            f"{question} [SEP] {passage[:200]}"
            for question, passage in zip(
                dataset_subset["question"],
                dataset_subset["passage"],
            )
        ]

    return [
        f"{first} {second}"
        for first, second in zip(
            dataset_subset[fields[0]],
            dataset_subset[fields[1]],
        )
    ]


def sample_balanced_indices(
    labels,
    num_labels,
    requested_shots,
    seed,
):
    """
    Match the original few-shot sampling logic.

    For MNLI:
        requested_shots = 64
        num_labels = 3
        examples per class = 64 // 3 = 21
        actual total = 63
    """
    rng = np.random.default_rng(seed)
    examples_per_class = requested_shots // num_labels
    selected = []

    for class_id in range(num_labels):
        class_indices = np.where(labels == class_id)[0].copy()

        if len(class_indices) < examples_per_class:
            raise ValueError(
                f"Class {class_id} contains only {len(class_indices)} "
                f"examples; {examples_per_class} are required."
            )

        rng.shuffle(class_indices)
        selected.extend(
            class_indices[:examples_per_class].tolist()
        )

    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


# ---------------------------------------------------------------------
# Frozen BERT embeddings
# ---------------------------------------------------------------------

def get_cls_embeddings(
    model,
    tokenizer,
    texts,
    batch_size,
):
    model.eval()
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            return_tensors="pt",
        )

        encoded = {
            key: value.to("cpu")
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            outputs = model(**encoded)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(
            cls_embeddings.cpu().numpy().astype(np.float32)
        )

    return np.concatenate(all_embeddings, axis=0)


# ---------------------------------------------------------------------
# Deterministic PCA-class alignment
# ---------------------------------------------------------------------

def pca_class_alignment(X, y, rank=RANK):
    """
    Compute the PCA-class alignment diagnostic alpha.

    This uses exact deterministic PCA:

        svd_solver="full"

    This removes randomized-SVD variability while computing the same
    mathematical rank-r principal subspace.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    classes = np.unique(y)
    overall_mean = X.mean(axis=0)

    n_components = min(
        rank,
        X.shape[0] - 1,
        X.shape[1],
    )

    if n_components <= 0:
        return 0.0

    pca = PCA(
        n_components=n_components,
        svd_solver="full",
    )
    pca.fit(X)

    principal_basis = pca.components_.T

    projected_between_class_energy = 0.0
    total_between_class_energy = 0.0

    for class_id in classes:
        mask = y == class_id
        n_class = int(mask.sum())

        class_mean = X[mask].mean(axis=0)
        difference = class_mean - overall_mean

        total_between_class_energy += (
            n_class * float(np.dot(difference, difference))
        )

        projected = principal_basis.T @ difference

        projected_between_class_energy += (
            n_class * float(np.dot(projected, projected))
        )

    if total_between_class_energy < 1e-10:
        return 0.0

    return float(
        projected_between_class_energy
        / total_between_class_energy
    )


# ---------------------------------------------------------------------
# Task embedding cache
# ---------------------------------------------------------------------

def get_task_subset_embeddings(
    task_name,
    task_info,
    dataset,
    model,
    tokenizer,
    seed,
    cache_dir,
    batch_size,
    force_recompute,
):
    cache_path = (
        Path(cache_dir)
        / f"{safe_name(task_name)}_shot64_seed{seed}.npz"
    )

    if cache_path.exists() and not force_recompute:
        cached = load_cache(cache_path)
        print(f"      Loaded cache: {cache_path.name}")
        return cached

    labels_all = normalize_labels(
        dataset[task_info["label_key"]],
        task_info["label_key"],
    )

    indices = sample_balanced_indices(
        labels=labels_all,
        num_labels=task_info["num_labels"],
        requested_shots=REQUESTED_SHOTS,
        seed=seed,
    )

    subset = dataset.select(indices.tolist())
    texts = get_texts(subset, task_info)
    labels = labels_all[indices]

    embeddings = get_cls_embeddings(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        batch_size=batch_size,
    )

    metadata = {
        "task": task_name,
        "seed": seed,
        "model": MODEL_NAME,
        "max_seq_len": MAX_SEQ_LEN,
        "rank": RANK,
        "requested_shots": REQUESTED_SHOTS,
        "actual_examples": int(len(indices)),
        "num_labels": task_info["num_labels"],
    }

    save_cache(
        path=cache_path,
        X=embeddings,
        y=labels,
        indices=indices,
        metadata=metadata,
    )

    print(f"      Saved cache:  {cache_path.name}")

    return {
        "X": embeddings,
        "y": labels,
        "indices": indices,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------
# Few-shot subset sensitivity
# ---------------------------------------------------------------------

def summarize_subset_values(task_name, values):
    values = np.asarray(values, dtype=np.float64)

    paper_result = PAPER_RESULTS.get(task_name, {})

    return {
        "task": task_name,
        "num_subsets": int(len(values)),
        "mean_alpha": float(np.mean(values)),
        "std_alpha": float(np.std(values, ddof=1)),
        "minimum_alpha": float(np.min(values)),
        "percentile_10": float(np.percentile(values, 10)),
        "median_alpha": float(np.median(values)),
        "percentile_90": float(np.percentile(values, 90)),
        "maximum_alpha": float(np.max(values)),
        "fraction_alpha_ge_0.6": float(
            np.mean(values >= PROVISIONAL_ALPHA_INDICATOR)
        ),
        "count_alpha_ge_0.6": int(
            np.sum(values >= PROVISIONAL_ALPHA_INDICATOR)
        ),
        "provisional_indicator": PROVISIONAL_ALPHA_INDICATOR,
        "paper_alpha": paper_result.get("alpha", ""),
        "pivot_gain_fraction": paper_result.get("gain", ""),
        "pivot_gain_percentage_points": (
            100.0 * paper_result["gain"]
            if "gain" in paper_result
            else ""
        ),
    }


# ---------------------------------------------------------------------
# AG News imbalance analysis
# ---------------------------------------------------------------------

def get_agnews_pool(
    dataset,
    model,
    tokenizer,
    cache_dir,
    batch_size,
    force_recompute,
    examples_per_class=64,
    seed=2026,
):
    cache_path = (
        Path(cache_dir)
        / f"ag_news_pool_{examples_per_class}_per_class_seed{seed}.npz"
    )

    if cache_path.exists() and not force_recompute:
        cached = load_cache(cache_path)
        print(f"    Loaded pool cache: {cache_path.name}")
        return cached

    labels_all = np.asarray(dataset["label"], dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = []

    for class_id in range(4):
        class_indices = np.where(labels_all == class_id)[0].copy()

        if len(class_indices) < examples_per_class:
            raise ValueError(
                f"AG News class {class_id} contains only "
                f"{len(class_indices)} examples."
            )

        rng.shuffle(class_indices)
        selected.extend(
            class_indices[:examples_per_class].tolist()
        )

    selected = np.asarray(selected, dtype=np.int64)
    subset = dataset.select(selected.tolist())

    embeddings = get_cls_embeddings(
        model=model,
        tokenizer=tokenizer,
        texts=list(subset["text"]),
        batch_size=batch_size,
    )

    labels = labels_all[selected]

    metadata = {
        "task": "AG News imbalance pool",
        "model": MODEL_NAME,
        "seed": seed,
        "examples_per_class": examples_per_class,
        "total_examples": int(len(selected)),
        "max_seq_len": MAX_SEQ_LEN,
        "rank": RANK,
    }

    save_cache(
        path=cache_path,
        X=embeddings,
        y=labels,
        indices=selected,
        metadata=metadata,
    )

    print(f"    Saved pool cache:  {cache_path.name}")

    return {
        "X": embeddings,
        "y": labels,
        "indices": selected,
        "metadata": metadata,
    }


def sample_imbalance_profile(X, y, counts, seed):
    rng = np.random.default_rng(seed)
    selected = []

    for class_id, requested_count in enumerate(counts):
        class_indices = np.where(y == class_id)[0]

        if len(class_indices) < requested_count:
            raise ValueError(
                f"Requested {requested_count} examples from class "
                f"{class_id}, but only {len(class_indices)} are cached."
            )

        chosen = rng.choice(
            class_indices,
            size=requested_count,
            replace=False,
        )

        selected.extend(chosen.tolist())

    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)

    return X[selected], y[selected]


def run_imbalance_analysis(
    pool,
    repeats,
    base_seed=5000,
):
    per_repeat_rows = []
    summary_rows = []

    X_pool = pool["X"]
    y_pool = pool["y"]

    for profile_number, (profile_name, counts) in enumerate(
        IMBALANCE_PROFILES.items()
    ):
        values = []

        for repeat in range(repeats):
            seed = base_seed + profile_number * 10000 + repeat

            X_subset, y_subset = sample_imbalance_profile(
                X=X_pool,
                y=y_pool,
                counts=counts,
                seed=seed,
            )

            alpha = pca_class_alignment(
                X_subset,
                y_subset,
                rank=RANK,
            )

            values.append(alpha)

            per_repeat_rows.append({
                "profile": profile_name,
                "repeat": repeat,
                "seed": seed,
                "class_0": counts[0],
                "class_1": counts[1],
                "class_2": counts[2],
                "class_3": counts[3],
                "total_examples": sum(counts),
                "alpha": alpha,
                "alpha_ge_0.6": (
                    alpha >= PROVISIONAL_ALPHA_INDICATOR
                ),
            })

        values = np.asarray(values, dtype=np.float64)

        summary_rows.append({
            "profile": profile_name,
            "counts": "/".join(str(value) for value in counts),
            "repeats": repeats,
            "mean_alpha": float(np.mean(values)),
            "std_alpha": float(np.std(values, ddof=1)),
            "minimum_alpha": float(np.min(values)),
            "percentile_10": float(np.percentile(values, 10)),
            "median_alpha": float(np.median(values)),
            "percentile_90": float(np.percentile(values, 90)),
            "maximum_alpha": float(np.max(values)),
            "fraction_alpha_ge_0.6": float(
                np.mean(values >= PROVISIONAL_ALPHA_INDICATOR)
            ),
        })

    return per_repeat_rows, summary_rows


# ---------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------

def average_rank(values):
    """
    Average ranks for Spearman correlation without requiring scipy.
    """
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]

    ranks = np.empty(len(values), dtype=np.float64)

    start = 0
    while start < len(values):
        end = start + 1

        while (
            end < len(values)
            and sorted_values[end] == sorted_values[start]
        ):
            end += 1

        # Ranks are one-indexed.
        average = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average
        start = end

    return ranks


def safe_correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) < 2:
        return float("nan")

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def calculate_correlations(summary_rows):
    tasks = []
    recomputed_alphas = []
    paper_alphas = []
    gains = []

    for row in summary_rows:
        task_name = row["task"]

        if task_name not in PAPER_RESULTS:
            continue

        tasks.append(task_name)
        recomputed_alphas.append(float(row["mean_alpha"]))
        paper_alphas.append(
            float(PAPER_RESULTS[task_name]["alpha"])
        )
        gains.append(
            float(PAPER_RESULTS[task_name]["gain"])
        )

    return {
        "tasks": tasks,
        "recomputed_mean_alphas": recomputed_alphas,
        "paper_alphas": paper_alphas,
        "gains": gains,
        "recomputed_pearson_r": safe_correlation(
            recomputed_alphas,
            gains,
        ),
        "recomputed_spearman_rho": safe_correlation(
            average_rank(recomputed_alphas),
            average_rank(gains),
        ),
        "paper_pearson_r": safe_correlation(
            paper_alphas,
            gains,
        ),
        "paper_spearman_rho": safe_correlation(
            average_rank(paper_alphas),
            average_rank(gains),
        ),
    }


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def create_plots(
    output_dir,
    subset_rows,
    subset_summary_rows,
    imbalance_summary_rows,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "\nmatplotlib is unavailable; skipping plots. "
            "CSV and JSON results were still saved."
        )
        return

    output_dir = Path(output_dir)

    # -------------------------------------------------------------
    # Plot 1: alpha against gain
    # -------------------------------------------------------------

    figure, axis = plt.subplots(figsize=(7.2, 5.2))

    for row in subset_summary_rows:
        task_name = row["task"]

        if task_name not in PAPER_RESULTS:
            continue

        mean_alpha = float(row["mean_alpha"])
        std_alpha = float(row["std_alpha"])
        gain_pp = (
            100.0 * PAPER_RESULTS[task_name]["gain"]
        )

        axis.errorbar(
            mean_alpha,
            gain_pp,
            xerr=std_alpha,
            fmt="o",
            capsize=4,
            markersize=7,
        )

        axis.annotate(
            task_name,
            xy=(mean_alpha, gain_pp),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    axis.axvline(
        PROVISIONAL_ALPHA_INDICATOR,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label=(
            rf"Provisional indicator "
            rf"($\alpha={PROVISIONAL_ALPHA_INDICATOR}$)"
        ),
    )

    axis.set_xlabel(r"Mean PCA-class alignment $\alpha$")
    axis.set_ylabel("PivotLoRA gain (percentage points)")
    axis.set_title(
        "Alignment diagnostic versus observed PivotLoRA gain"
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()

    figure.savefig(
        output_dir / "alpha_vs_gain.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    # -------------------------------------------------------------
    # Plot 2: per-subset variability
    # -------------------------------------------------------------

    task_order = [
        "AG News",
        "SST-2",
        "QNLI",
        "MNLI",
        "BoolQ",
        "RTE",
    ]

    grouped = {
        task_name: []
        for task_name in task_order
    }

    for row in subset_rows:
        grouped[row["task"]].append(float(row["alpha"]))

    data = [
        grouped[task_name]
        for task_name in task_order
    ]

    figure, axis = plt.subplots(figsize=(8.0, 5.2))

    axis.boxplot(
        data,
        labels=task_order,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "tab:red",
            "markeredgecolor": "tab:red",
        },
    )

    axis.axhline(
        PROVISIONAL_ALPHA_INDICATOR,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label=(
            rf"Provisional indicator "
            rf"($\alpha={PROVISIONAL_ALPHA_INDICATOR}$)"
        ),
    )

    axis.set_ylabel(r"PCA-class alignment $\alpha$")
    axis.set_title(
        "Sensitivity to the sampled 64-shot training subset"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()

    figure.savefig(
        output_dir / "alpha_subset_variability.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    # -------------------------------------------------------------
    # Plot 3: AG News imbalance
    # -------------------------------------------------------------

    profile_names = [
        row["profile"]
        for row in imbalance_summary_rows
    ]
    means = np.asarray([
        float(row["mean_alpha"])
        for row in imbalance_summary_rows
    ])
    standard_deviations = np.asarray([
        float(row["std_alpha"])
        for row in imbalance_summary_rows
    ])

    figure, axis = plt.subplots(figsize=(6.6, 4.8))

    axis.bar(
        profile_names,
        means,
        yerr=standard_deviations,
        capsize=5,
        color=["tab:blue", "tab:orange", "tab:red"],
        alpha=0.85,
    )

    axis.axhline(
        PROVISIONAL_ALPHA_INDICATOR,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=(
            rf"$\alpha={PROVISIONAL_ALPHA_INDICATOR}$"
        ),
    )

    axis.set_ylabel(r"PCA-class alignment $\alpha$")
    axis.set_title("AG News class-imbalance sensitivity")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()

    figure.savefig(
        output_dir / "agnews_imbalance.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only repeated-subset and class-imbalance analysis "
            "for PCA-class alignment alpha."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_alpha_uncertainty",
    )

    parser.add_argument(
        "--num-subsets",
        type=int,
        default=10,
        help=(
            "Number of independent few-shot subsets per task. "
            "Maximum supported by the default seed list is 10."
        ),
    )

    parser.add_argument(
        "--imbalance-repeats",
        type=int,
        default=30,
        help=(
            "Number of independently sampled AG News subsets "
            "per imbalance profile."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Frozen BERT inference batch size.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of PyTorch CPU threads.",
    )

    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached embeddings and recompute them.",
    )

    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip matplotlib plots.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_arguments()

    if args.num_subsets < 2:
        raise ValueError(
            "--num-subsets must be at least 2 to calculate variability."
        )

    if args.num_subsets > len(DEFAULT_SUBSET_SEEDS):
        raise ValueError(
            f"--num-subsets cannot exceed "
            f"{len(DEFAULT_SUBSET_SEEDS)} with the default seed list."
        )

    if args.imbalance_repeats < 3:
        raise ValueError(
            "--imbalance-repeats must be at least 3."
        )

    subset_seeds = DEFAULT_SUBSET_SEEDS[:args.num_subsets]

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)
    set_global_seed(12345)

    print("=" * 78)
    print("ALPHA SUBSET-SENSITIVITY AND CLASS-IMBALANCE ANALYSIS")
    print("=" * 78)
    print(f"Device:                  cpu")
    print(f"Model:                   {MODEL_NAME}")
    print(f"PCA solver:              full (exact/deterministic)")
    print(f"Rank:                    {RANK}")
    print(f"Requested shots:         {REQUESTED_SHOTS}")
    print(f"Subset seeds:            {subset_seeds}")
    print(f"Imbalance repetitions:   {args.imbalance_repeats}")
    print(f"CPU threads:             {args.threads}")
    print(f"Embedding batch size:    {args.batch_size}")
    print(f"Output directory:        {output_dir.resolve()}")
    print("=" * 78)

    print("\nLoading tokenizer and frozen BERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to("cpu")
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    subset_rows = []
    subset_summary_rows = []

    # -------------------------------------------------------------
    # Part 1: repeated independent few-shot subsets
    # -------------------------------------------------------------

    print("\n" + "=" * 78)
    print("PART 1: REPEATED INDEPENDENT FEW-SHOT SUBSETS")
    print("=" * 78)

    for task_name, task_info in TASKS.items():
        print("\n" + "-" * 78)
        print(f"Task: {task_name}")
        print("-" * 78)

        dataset_dict = load_dataset(*task_info["dataset"])
        train_dataset = dataset_dict[task_info["split"]]

        alpha_values = []

        for subset_number, seed in enumerate(subset_seeds):
            print(
                f"    Subset {subset_number + 1:02d}/"
                f"{len(subset_seeds):02d}, seed={seed}"
            )

            cached = get_task_subset_embeddings(
                task_name=task_name,
                task_info=task_info,
                dataset=train_dataset,
                model=model,
                tokenizer=tokenizer,
                seed=seed,
                cache_dir=cache_dir,
                batch_size=args.batch_size,
                force_recompute=args.force_recompute,
            )

            alpha = pca_class_alignment(
                X=cached["X"],
                y=cached["y"],
                rank=RANK,
            )

            alpha_values.append(alpha)

            subset_rows.append({
                "task": task_name,
                "subset_number": subset_number + 1,
                "seed": seed,
                "requested_shots": REQUESTED_SHOTS,
                "actual_examples": len(cached["y"]),
                "num_classes": task_info["num_labels"],
                "alpha": alpha,
                "alpha_ge_0.6": (
                    alpha >= PROVISIONAL_ALPHA_INDICATOR
                ),
                "paper_alpha": PAPER_RESULTS[task_name]["alpha"],
                "pivot_gain_percentage_points": (
                    100.0 * PAPER_RESULTS[task_name]["gain"]
                ),
            })

            print(f"      alpha={alpha:.4f}")

        summary = summarize_subset_values(
            task_name=task_name,
            values=alpha_values,
        )
        subset_summary_rows.append(summary)

        print(
            f"\n    Mean ± SD: "
            f"{summary['mean_alpha']:.4f} "
            f"± {summary['std_alpha']:.4f}"
        )
        print(
            f"    Range:     "
            f"[{summary['minimum_alpha']:.4f}, "
            f"{summary['maximum_alpha']:.4f}]"
        )
        print(
            f"    alpha >= 0.6: "
            f"{summary['count_alpha_ge_0.6']}/"
            f"{summary['num_subsets']}"
        )

    write_csv(
        output_dir / "alpha_by_subset.csv",
        subset_rows,
    )
    write_csv(
        output_dir / "alpha_subset_sensitivity.csv",
        subset_summary_rows,
    )

    # -------------------------------------------------------------
    # Correlation analysis
    # -------------------------------------------------------------

    correlations = calculate_correlations(
        subset_summary_rows
    )

    print("\n" + "=" * 78)
    print("ALPHA VERSUS EXISTING PIVOTLORA GAIN")
    print("=" * 78)

    print(
        f"{'Task':<10} "
        f"{'Mean alpha':>12} "
        f"{'Paper alpha':>12} "
        f"{'Gain':>10}"
    )
    print("-" * 50)

    for task_name, recomputed, paper_alpha, gain in zip(
        correlations["tasks"],
        correlations["recomputed_mean_alphas"],
        correlations["paper_alphas"],
        correlations["gains"],
    ):
        print(
            f"{task_name:<10} "
            f"{recomputed:>12.3f} "
            f"{paper_alpha:>12.3f} "
            f"{gain * 100:>+9.1f}pp"
        )

    print("\nRepeated-subset mean alpha:")
    print(
        f"  Pearson r:    "
        f"{correlations['recomputed_pearson_r']:.4f}"
    )
    print(
        f"  Spearman rho: "
        f"{correlations['recomputed_spearman_rho']:.4f}"
    )

    print("\nSubmitted paper alpha:")
    print(
        f"  Pearson r:    "
        f"{correlations['paper_pearson_r']:.4f}"
    )
    print(
        f"  Spearman rho: "
        f"{correlations['paper_spearman_rho']:.4f}"
    )

    # -------------------------------------------------------------
    # Part 2: AG News imbalance
    # -------------------------------------------------------------

    print("\n" + "=" * 78)
    print("PART 2: AG NEWS CLASS-IMBALANCE SENSITIVITY")
    print("=" * 78)

    agnews_dataset = load_dataset("fancyzhx/ag_news")["train"]

    agnews_pool = get_agnews_pool(
        dataset=agnews_dataset,
        model=model,
        tokenizer=tokenizer,
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        force_recompute=args.force_recompute,
        examples_per_class=64,
        seed=2026,
    )

    imbalance_rows, imbalance_summary_rows = (
        run_imbalance_analysis(
            pool=agnews_pool,
            repeats=args.imbalance_repeats,
            base_seed=5000,
        )
    )

    write_csv(
        output_dir / "agnews_imbalance_by_repeat.csv",
        imbalance_rows,
    )
    write_csv(
        output_dir / "agnews_imbalance_summary.csv",
        imbalance_summary_rows,
    )

    print()
    print(
        f"{'Profile':<12} "
        f"{'Counts':<14} "
        f"{'Mean':>9} "
        f"{'SD':>9} "
        f"{'Range':>22} "
        f"{'Frac >=.6':>12}"
    )
    print("-" * 84)

    for row in imbalance_summary_rows:
        print(
            f"{row['profile']:<12} "
            f"{row['counts']:<14} "
            f"{row['mean_alpha']:>9.4f} "
            f"{row['std_alpha']:>9.4f} "
            f"[{row['minimum_alpha']:.4f}, "
            f"{row['maximum_alpha']:.4f}] "
            f"{row['fraction_alpha_ge_0.6']:>12.2f}"
        )

    # -------------------------------------------------------------
    # JSON summary
    # -------------------------------------------------------------

    analysis_summary = {
        "configuration": {
            "model": MODEL_NAME,
            "device": "cpu",
            "max_seq_len": MAX_SEQ_LEN,
            "rank": RANK,
            "pca_solver": "full",
            "requested_shots": REQUESTED_SHOTS,
            "subset_seeds": subset_seeds,
            "imbalance_repeats": args.imbalance_repeats,
            "imbalance_profiles": IMBALANCE_PROFILES,
            "provisional_alpha_indicator": (
                PROVISIONAL_ALPHA_INDICATOR
            ),
        },
        "subset_sensitivity": subset_summary_rows,
        "correlations": correlations,
        "agnews_imbalance": imbalance_summary_rows,
        "interpretation_limits": [
            (
                "Variability across independently sampled few-shot "
                "subsets is reported instead of an ordinary bootstrap."
            ),
            (
                "The value alpha=0.6 is a provisional empirical "
                "high-alignment indicator for the evaluated tasks, "
                "not a universal theoretical threshold."
            ),
            (
                "The six-task correlations are descriptive because "
                "the number of independent tasks is small."
            ),
            (
                "The imbalance experiment evaluates stability of "
                "alpha, not PivotLoRA training performance under "
                "class imbalance."
            ),
            (
                "The deterministic full PCA solver computes the exact "
                "rank-r sample principal subspace and removes randomized "
                "PCA solver variability."
            ),
        ],
    }

    with (output_dir / "analysis_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            analysis_summary,
            handle,
            indent=2,
        )

    # -------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------

    if not args.skip_plots:
        create_plots(
            output_dir=output_dir,
            subset_rows=subset_rows,
            subset_summary_rows=subset_summary_rows,
            imbalance_summary_rows=imbalance_summary_rows,
        )

    # -------------------------------------------------------------
    # Final output
    # -------------------------------------------------------------

    print("\n" + "=" * 78)
    print("COMPLETED")
    print("=" * 78)
    print(f"Results directory: {output_dir.resolve()}")
    print("\nPrimary files:")
    print(f"  {output_dir / 'alpha_by_subset.csv'}")
    print(f"  {output_dir / 'alpha_subset_sensitivity.csv'}")
    print(f"  {output_dir / 'agnews_imbalance_by_repeat.csv'}")
    print(f"  {output_dir / 'agnews_imbalance_summary.csv'}")
    print(f"  {output_dir / 'analysis_summary.json'}")

    if not args.skip_plots:
        print(f"  {output_dir / 'alpha_vs_gain.png'}")
        print(f"  {output_dir / 'alpha_subset_variability.png'}")
        print(f"  {output_dir / 'agnews_imbalance.png'}")

    print("\nPermitted interpretation:")
    print(
        "  1. The cross-task correlations describe the association "
        "between alpha and observed PivotLoRA gain."
    )
    print(
        "  2. Repeated subset sampling measures sensitivity to the "
        "available few-shot examples."
    )
    print(
        "  3. The imbalance analysis measures whether alpha remains "
        "qualitatively stable as class proportions become unequal."
    )
    print(
        "  4. Alpha=0.6 is an empirical high-alignment indicator for "
        "the evaluated setup, not a universal calibrated threshold."
    )
    print(
        "  5. This analysis does not establish that PivotLoRA training "
        "performance is invariant to class imbalance."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
