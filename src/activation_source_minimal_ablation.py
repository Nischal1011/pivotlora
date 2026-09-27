#!/usr/bin/env python3
"""
activation_source_minimal_ablation.py

Minimal, tightly controlled activation-source ablation for BERT-base
PivotLoRA:

    Target-position ([CLS]) PCA source
        versus
    All-token PCA source

Both conditions use the SAME source for query and value adapters. No
projection-specific or attention-weighted variants are included.

This script does not modify `train_classification.py` (the submitted harness) or
`activation_source_ablation.py` (the existing four-condition ablation). It
imports both unchanged and only restricts execution to the minimal grid:

    Tasks:   AG News, SST-2
    Shot:    N = 64
    Seeds:   42, 43, 44
    Sources: target_only, all_token

Everything else -- sampled examples (via `be.load_task_with_splits`),
classification-head initialization, rank r=8, alpha=16, q/v target modules,
PCA solver/seed/scaling/rank, eigenvalue-scaled B learning rates, pivot
epoch K=3, optimizer-state reset policy (`be.PivotCallback`, unchanged),
batch size 16, 15 epochs, and the evaluation split -- is inherited exactly
as `activation_source_ablation.run_projection_source_experiment` already
implements it; none of those values are redefined here.

Reuse policy: if a (task, source, seed) cell already exists in the
codebase's own `results_projection_source/projection_source_raw.csv` under
this exact protocol (same shot, learning rate, epoch count), that cell is
REUSED rather than retrained -- this generalizes the request's own
assumption ("if the existing target-only runs use the same samples and
protocol, this requires only N new runs") to any already-completed cell.
Every output row records whether it was reused or freshly trained. Pass
--force to ignore existing rows and retrain every cell from scratch.

Usage:
    cd bert
    python activation_source_minimal_ablation.py
    python activation_source_minimal_ablation.py --force
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

try:
    import train_classification as be
except ImportError as exc:
    raise ImportError(
        "Could not import train_classification.py. Run this script from the "
        "bert/ directory (or add it to PYTHONPATH)."
    ) from exc

try:
    import activation_source_ablation as ps
except ImportError as exc:
    raise ImportError(
        "Could not import activation_source_ablation.py. Run this script "
        "from the bert/ directory (or add it to PYTHONPATH)."
    ) from exc


# =====================================================================
# Minimal experiment grid (everything else comes from be.* / ps.* unchanged)
# =====================================================================

TASKS = ["agnews", "sst2"]
SHOT = 64
SEEDS = [42, 43, 44]
SOURCES = ["target_only", "all_token"]

OUTPUT_DIR = Path("results_activation_source_minimal")
RAW_CSV = OUTPUT_DIR / "activation_source_minimal_raw.csv"
REPORT_MD = OUTPUT_DIR / "activation_source_minimal_results.md"

RAW_FIELDS = [
    "task",
    "shot",
    "source",
    "seed",
    "lr",
    "epochs",
    "score",
    "metric",
    "wall_s",
    "peak_memory_mb",
    "pivot_done",
    "provenance",
]

TASK_DISPLAY = {
    "agnews": "AG News",
    "sst2": "SST-2",
}

SOURCE_DISPLAY = {
    "target_only": "Target position ([CLS])",
    "all_token": "All non-padding tokens",
}


# =====================================================================
# Reuse detection against the existing, unmodified ablation's own output
# =====================================================================

def find_existing_row(existing_rows, task, source, seed):
    for row in existing_rows:
        try:
            if (
                row["task"] == task
                and int(row["shot"]) == SHOT
                and row["source"] == source
                and int(row["seed"]) == seed
                and abs(float(row["lr"]) - be.LEARNING_RATE) < 1e-12
                and int(row["epochs"]) == be.EPOCHS
            ):
                return row
        except (KeyError, ValueError):
            continue
    return None


def diagnostics_for(task, seed, source):
    """
    Average num_vectors / explained_fraction (rank-8 PCA) across the 12
    BERT layers, read from the existing (unmodified) ablation's own
    diagnostics CSV -- the activation source bank is cached to disk and
    reused bit-for-bit regardless of whether the training run itself was
    reused or freshly executed.
    """
    if not ps.DIAGNOSTICS_CSV.exists():
        return None

    diag_source = "target" if source == "target_only" else "all"
    num_vectors = []
    explained = []

    with ps.DIAGNOSTICS_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["task"] == task
                and int(row["shot"]) == SHOT
                and int(row["seed"]) == seed
                and row["source"] == diag_source
            ):
                num_vectors.append(float(row["num_vectors"]))
                explained.append(float(row["explained_fraction"]))

    if not num_vectors:
        return None

    return {
        "num_vectors_mean": statistics.mean(num_vectors),
        "explained_fraction_mean": statistics.mean(explained),
        "n_layers": len(num_vectors),
    }


# =====================================================================
# Run / reuse the 12-cell grid
# =====================================================================

def collect_rows(force):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ps.ensure_directories()

    existing_rows = [] if force else ps.load_raw_results()

    rows = []
    for task in TASKS:
        for source in SOURCES:
            for seed in SEEDS:
                existing = find_existing_row(existing_rows, task, source, seed)

                if existing is not None:
                    print(
                        f"[REUSE] {task} {source} seed={seed}: "
                        f"score={existing['score']} "
                        f"(from {ps.RAW_CSV})"
                    )
                    row = {
                        "task": task,
                        "shot": SHOT,
                        "source": source,
                        "seed": seed,
                        "lr": be.LEARNING_RATE,
                        "epochs": be.EPOCHS,
                        "score": float(existing["score"]),
                        "metric": existing["metric"],
                        "wall_s": float(existing["wall_s"]),
                        "peak_memory_mb": existing.get("peak_memory_mb", ""),
                        "pivot_done": existing["pivot_done"],
                        "provenance": "reused_existing_artifact",
                    }
                else:
                    print(
                        f"[RUN] {task} {source} seed={seed}: "
                        f"no matching existing row -- training fresh..."
                    )
                    result, _diag = ps.run_projection_source_experiment(
                        task=task,
                        shot=SHOT,
                        seed=seed,
                        source_name=source,
                        learning_rate=be.LEARNING_RATE,
                        epochs=be.EPOCHS,
                        batch_size=be.BATCH_SIZE,
                        dataloader_workers=0,
                    )
                    row = {
                        "task": result["task"],
                        "shot": result["shot"],
                        "source": result["source"],
                        "seed": result["seed"],
                        "lr": result["lr"],
                        "epochs": result["epochs"],
                        "score": result["score"],
                        "metric": result["metric"],
                        "wall_s": result["wall_s"],
                        "peak_memory_mb": result["peak_memory_mb"],
                        "pivot_done": result["pivot_done"],
                        "provenance": "fresh_run_this_script",
                    }

                rows.append(row)

    with RAW_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote {len(rows)} rows to {RAW_CSV}")
    return rows


# =====================================================================
# Report generation
# =====================================================================

def mean_sd(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def build_report(rows):
    by_key = {}
    for row in rows:
        by_key[(row["task"], row["source"], int(row["seed"]))] = row

    lines = []
    lines.append("# Activation-Source Ablation: Target-Position vs. All-Token PivotLoRA")
    lines.append("")
    lines.append(
        "Minimal, tightly controlled ablation isolating the PCA activation "
        "source used to initialize BERT-base PivotLoRA's query/value "
        "adapters. Produced by `bert/activation_source_minimal_ablation.py`, "
        "which imports `train_classification.py` (the submitted harness) and "
        "`activation_source_ablation.py` unchanged -- no harness code was "
        "modified to run this experiment."
    )
    lines.append("")
    lines.append("## Fixed protocol (identical across both conditions)")
    lines.append("")
    lines.append("All values below are read directly from `train_classification.py` at run time, not redefined by this script:")
    lines.append("")
    lines.append(f"- Model: `{be.MODEL_NAME}`")
    lines.append(f"- Rank / alpha: r={be.RANK}, alpha={be.ALPHA}")
    lines.append("- Adapted modules: query and value projections")
    lines.append(f"- Learning rate: {be.LEARNING_RATE:.1e} (shared base rate; pre-pivot B rows use the harness's eigenvalue-scaled rate)")
    lines.append(f"- Epochs: {be.EPOCHS}")
    lines.append(f"- Batch size: {be.BATCH_SIZE}")
    lines.append(f"- Pivot epoch: K={be.PIVOT_EPOCH}, pivot fraction={be.PIVOT_K_FRACTION}, post-pivot B warmup={be.PIVOT_WARMUP_STEPS} steps")
    lines.append("- Optimizer-state handling: `be.PivotCallback` (unchanged) -- Adam moments reset for A, retained for B")
    lines.append("- Sampled examples: `be.load_task_with_splits(task, shot, seed)` (unchanged), identical for both sources at a given (task, seed)")
    lines.append("- Evaluation split: task's standard validation split, via `be.get_compute_metrics`")
    lines.append("- Sources differ only in the PCA activation matrix (X^target vs. X^all); the same source is used for both query and value adapters in each condition")
    lines.append("")

    lines.append("## Provenance")
    lines.append("")
    n_reused = sum(1 for r in rows if r["provenance"] == "reused_existing_artifact")
    n_fresh = sum(1 for r in rows if r["provenance"] == "fresh_run_this_script")
    lines.append(
        f"Of the {len(rows)} required (task, source, seed) cells, "
        f"**{n_reused} were reused** from the codebase's own "
        f"`results_projection_source/projection_source_raw.csv` (already run "
        f"under this exact protocol -- same shot, learning rate, and epoch "
        f"count) and **{n_fresh} were freshly trained** by this script."
    )
    lines.append("")
    lines.append("| Task | Source | Seed | Score | Provenance |")
    lines.append("|---|---|---:|---:|---|")
    for task in TASKS:
        for source in SOURCES:
            for seed in SEEDS:
                row = by_key[(task, source, seed)]
                lines.append(
                    f"| {TASK_DISPLAY[task]} | {SOURCE_DISPLAY[source]} | {seed} "
                    f"| {float(row['score']):.4f} | {row['provenance']} |"
                )
    lines.append("")

    lines.append("## Results")
    lines.append("")

    task_stats = {}
    for task in TASKS:
        target_scores = [float(by_key[(task, "target_only", s)]["score"]) for s in SEEDS]
        all_scores = [float(by_key[(task, "all_token", s)]["score"]) for s in SEEDS]
        paired_gains = [a - t for a, t in zip(all_scores, target_scores)]
        target_mean, target_sd = mean_sd(target_scores)
        all_mean, all_sd = mean_sd(all_scores)
        mean_gain = statistics.mean(paired_gains)
        wins = sum(1 for g in paired_gains if g > 0)
        task_stats[task] = {
            "target_scores": target_scores,
            "all_scores": all_scores,
            "paired_gains": paired_gains,
            "target_mean": target_mean,
            "target_sd": target_sd,
            "all_mean": all_mean,
            "all_sd": all_sd,
            "mean_gain": mean_gain,
            "wins": wins,
        }

    lines.append("### Table")
    lines.append("")
    header = "| Activation source | " + " | ".join(TASK_DISPLAY[t] for t in TASKS) + " |"
    sep = "|---|" + "---|" * len(TASKS)
    lines.append(header)
    lines.append(sep)
    row_target = "| Target position (`[CLS]`) | " + " | ".join(
        f"{task_stats[t]['target_mean']:.4f} ± {task_stats[t]['target_sd']:.4f}" for t in TASKS
    ) + " |"
    row_all = "| All non-padding tokens | " + " | ".join(
        f"{task_stats[t]['all_mean']:.4f} ± {task_stats[t]['all_sd']:.4f}" for t in TASKS
    ) + " |"
    lines.append(row_target)
    lines.append(row_all)
    row_gain = "| Paired gain of all-token | " + " | ".join(
        f"{task_stats[t]['mean_gain']:+.4f}" for t in TASKS
    ) + " |"
    row_wins = "| Seed-level wins (all-token) | " + " | ".join(
        f"{task_stats[t]['wins']}/3" for t in TASKS
    ) + " |"
    lines.append(row_gain)
    lines.append(row_wins)
    lines.append("")

    lines.append("### Per-seed paired differences (all-token minus target-only)")
    lines.append("")
    lines.append("| Task | Seed 42 | Seed 43 | Seed 44 | Mean paired diff |")
    lines.append("|---|---:|---:|---:|---:|")
    for task in TASKS:
        g = task_stats[task]["paired_gains"]
        lines.append(
            f"| {TASK_DISPLAY[task]} | {g[0]:+.4f} | {g[1]:+.4f} | {g[2]:+.4f} | {task_stats[task]['mean_gain']:+.4f} |"
        )
    lines.append("")

    lines.append("### PCA activation-source diagnostics (rank-8 PCA, averaged over 12 layers and 3 seeds)")
    lines.append("")
    lines.append("| Task | Source | Mean # PCA activation vectors | Mean top-8 explained variance |")
    lines.append("|---|---|---:|---:|")
    for task in TASKS:
        for source in SOURCES:
            diag_values = [diagnostics_for(task, s, source) for s in SEEDS]
            diag_values = [d for d in diag_values if d is not None]
            if diag_values:
                mean_vectors = statistics.mean(d["num_vectors_mean"] for d in diag_values)
                mean_explained = statistics.mean(d["explained_fraction_mean"] for d in diag_values)
                lines.append(
                    f"| {TASK_DISPLAY[task]} | {SOURCE_DISPLAY[source]} "
                    f"| {mean_vectors:.1f} | {mean_explained * 100:.2f}% |"
                )
            else:
                lines.append(
                    f"| {TASK_DISPLAY[task]} | {SOURCE_DISPLAY[source]} | n/a | n/a |"
                )
    lines.append("")
    lines.append(
        "`target_only` has exactly N=64 PCA activation vectors per layer (one `[CLS]` "
        "vector per sampled example); `all_token` has one vector per non-padding token "
        "across all N examples, so its vector count is much larger and varies slightly "
        "by tokenized sequence length."
    )
    lines.append("")

    lines.append("### LaTeX table")
    lines.append("")
    lines.append("```latex")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Activation-source ablation for BERT-base PivotLoRA at $N=64$."
    )
    lines.append(
        r"All configurations use identical sampled examples and training settings."
    )
    lines.append(
        r"Values are mean $\pm$ sample standard deviation over three paired seeds.}"
    )
    lines.append(r"\label{tab:activation-source}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}lcc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Activation source & AG~News & SST-2 \\")
    lines.append(r"\midrule")
    lines.append(
        r"Target position (\texttt{[CLS]}) & "
        f"{task_stats['agnews']['target_mean']:.3f}$\\pm${task_stats['agnews']['target_sd']:.3f} & "
        f"{task_stats['sst2']['target_mean']:.3f}$\\pm${task_stats['sst2']['target_sd']:.3f} \\\\"
    )
    lines.append(
        r"All non-padding tokens           & "
        f"{task_stats['agnews']['all_mean']:.3f}$\\pm${task_stats['agnews']['all_sd']:.3f} & "
        f"{task_stats['sst2']['all_mean']:.3f}$\\pm${task_stats['sst2']['all_sd']:.3f} \\\\"
    )
    lines.append(r"\midrule")
    lines.append(
        r"Paired gain of all-token         & "
        f"{task_stats['agnews']['mean_gain']:+.3f}             & "
        f"{task_stats['sst2']['mean_gain']:+.3f} \\\\"
    )
    lines.append(
        r"Seed-level wins                  & "
        f"{task_stats['agnews']['wins']}/3                 & "
        f"{task_stats['sst2']['wins']}/3 \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("```")
    lines.append("")

    lines.append("## What this experiment supports")
    lines.append("")
    any_all_wins = any(task_stats[t]["wins"] >= 2 for t in TASKS)
    any_target_wins = any(task_stats[t]["wins"] <= 1 for t in TASKS)
    if any_all_wins:
        winning_tasks = [TASK_DISPLAY[t] for t in TASKS if task_stats[t]["wins"] >= 2]
        lines.append(
            f"- All-token initialization outperforms target-token initialization in "
            f"at least one evaluated setting ({', '.join(winning_tasks)}), supporting: "
            f"\"All-token initialization can outperform target-token initialization in "
            f"individual settings.\""
        )
    else:
        lines.append(
            "- In this run, all-token initialization did not outperform target-token "
            "initialization in any evaluated setting (0/3 seed wins on both tasks)."
        )

    crossover = any(task_stats[t]["wins"] >= 2 for t in TASKS) and any(
        task_stats[t]["wins"] <= 1 for t in TASKS
    )
    if crossover:
        lines.append(
            "- Because at least one task favors all-token and at least one favors "
            "target-only (a genuine crossover), this result **also** supports the "
            "stronger claim: \"Neither source is uniformly preferable.\""
        )
    else:
        lines.append(
            "- This experiment does **not**, by itself, support \"Neither source is "
            "uniformly preferable\" -- both tasks point the same direction here. That "
            "stronger claim requires a tested setting where target-only wins and "
            "another where all-token wins; do not assume a crossover that was not "
            "observed."
        )
    lines.append("")
    lines.append(
        "No seed-level statistical significance is claimed anywhere above -- three "
        "seeds only support descriptive mean/SD and a paired win count, not a "
        "significance test."
    )
    lines.append("")

    return "\n".join(lines)


def main():
    global TASKS, SEEDS

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing matching rows in results_projection_source/ and retrain every cell.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=None,
        help="Restrict to a subset of tasks (spot-check convenience only; full report needs both).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        choices=SEEDS,
        default=None,
        help="Restrict to a subset of seeds (spot-check convenience only; full report needs all three).",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing the aggregate .md report (useful for a partial spot-check run).",
    )
    args = parser.parse_args()

    if args.tasks:
        TASKS = args.tasks
    if args.seeds:
        SEEDS = args.seeds

    print(f"Harness: {Path(be.__file__).resolve()}")
    print(f"Ablation module: {Path(ps.__file__).resolve()}")
    print(f"Tasks={TASKS} shot={SHOT} seeds={SEEDS} sources={SOURCES}")

    rows = collect_rows(force=args.force)

    if args.no_report:
        return

    report = build_report(rows)

    REPORT_MD.write_text(report, encoding="utf-8")
    print(f"\nWrote report to {REPORT_MD}")


if __name__ == "__main__":
    sys.exit(main())
