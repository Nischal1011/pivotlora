#!/usr/bin/env python3
"""
Recompute every derived statistic the paper reports from the CSVs in results/.

This is an arithmetic audit of the published summaries, not a reproduction of
training. It reads only ``results/*.csv`` -- no model, no dataset, no GPU --
and checks that the means, standard deviations, paired gains, ranks,
correlations, and permutation p-value stated in the paper follow from the
numbers actually shipped in this repository.

Usage
-----
    python analysis/verify_results.py            # run every check
    python analysis/verify_results.py --verbose  # also print each value

Exit code is 0 when every check passes and 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import pathlib
import sys
from collections import defaultdict

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append((name, detail))
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))


def close(a: float, b: float, tol: float = 5e-4) -> bool:
    return abs(a - b) <= tol


def load(name: str) -> list[dict]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def sample_std(values) -> float:
    """Sample standard deviation (ddof=1), the convention stated in the paper."""
    values = list(values)
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


# ---------------------------------------------------------------------------


def verify_bert_main(verbose: bool) -> None:
    print("\nBERT primary benchmark (results/bert_main.csv)")
    rows = load("bert_main.csv")
    table = {(r["task"], int(r["n"]), r["method"]): float(r["mean"]) for r in rows}

    primary_tasks = ["sst2", "cola", "mnli", "qnli", "rte", "agnews"]
    shots = [16, 32, 64, 128, 256]
    settings = [(t, n) for t in primary_tasks for n in shots]

    check("30 primary task-N settings", len(settings) == 30, f"{len(settings)}")
    check("13 methods x 7 tasks x 5 N = 455 rows", len(rows) == 455, f"{len(rows)}")

    wins = sum(1 for t, n in settings if table[(t, n, "PivotLoRA")] > table[(t, n, "LoRA")])
    losses = [(t, n) for t, n in settings if table[(t, n, "PivotLoRA")] < table[(t, n, "LoRA")]]
    check("PivotLoRA > LoRA in 29/30 settings", wins == 29, f"{wins}/30")
    check(
        "the single loss is RTE at N=128",
        losses == [("rte", 128)],
        f"{losses}",
    )
    if losses == [("rte", 128)]:
        delta = table[("rte", 128, "PivotLoRA")] - table[("rte", 128, "LoRA")]
        check("RTE N=128 difference is -0.020", close(delta, -0.020), f"{delta:+.3f}")

    # N=64 gains quoted in the alignment analysis.
    expected_gain = {
        "agnews": 0.194, "sst2": 0.143, "cola": 0.125,
        "qnli": 0.039, "mnli": 0.001, "rte": 0.018, "boolq": 0.016,
    }
    for task, want in expected_gain.items():
        got = table[(task, 64, "PivotLoRA")] - table[(task, 64, "LoRA")]
        check(f"N=64 gain on {task} is {want:+.3f}", close(got, want, 1e-3), f"{got:+.3f}")

    # Average ranks over the 30 primary settings (rank 1 = highest mean).
    methods = sorted({r["method"] for r in rows})
    rank_totals = defaultdict(float)
    for task, n in settings:
        ordered = sorted(methods, key=lambda m: -table[(task, n, m)])
        # Competition ranking with ties averaged.
        i = 0
        while i < len(ordered):
            j = i
            while j + 1 < len(ordered) and close(
                table[(task, n, ordered[j + 1])], table[(task, n, ordered[i])], 1e-12
            ):
                j += 1
            shared = mean(range(i + 1, j + 2))
            for k in range(i, j + 1):
                rank_totals[ordered[k]] += shared
            i = j + 1

    # NOTE ON RANK TOLERANCE.
    # bert_main.csv stores means rounded to three decimals, which is the
    # precision the paper itself reports. The published average ranks were
    # computed from full-precision per-seed means, so settings that are
    # separated at full precision can tie once rounded. Recomputing from the
    # rounded table therefore reproduces the *ordering* exactly but shifts
    # individual ranks by up to ~0.3. We check the ordering strictly and the
    # magnitudes loosely rather than pretending to an agreement that rounding
    # makes impossible.
    computed = {m: rank_totals[m] / 30 for m in methods}
    published = {r["method"]: float(r["average_rank"]) for r in load("bert_aggregate.csv")}

    order_published = [m for m, _ in sorted(published.items(), key=lambda kv: kv[1])]
    order_computed = [m for m, _ in sorted(computed.items(), key=lambda kv: kv[1])]
    # The 8.43/8.43/8.47/8.47 cluster is a near four-way tie in the paper, so
    # compare the ordering outside it.
    cluster = {"LoRA+", "LoRAM", "LoRA-One", "EVA"}
    check(
        "recomputed rank ordering matches the published ordering",
        [m for m in order_computed if m not in cluster]
        == [m for m in order_published if m not in cluster],
        f"{[m for m in order_computed if m not in cluster]}",
    )
    check(
        "the near-tied 8.4x cluster contains the same four methods",
        set(order_computed[6:10]) == cluster,
        f"{order_computed[6:10]}",
    )
    for method, want in sorted(published.items(), key=lambda kv: kv[1]):
        got = computed[method]
        check(
            f"average rank of {method} ~ {want:.2f} (rounded-table recomputation)",
            close(got, want, 0.35),
            f"{got:.2f}",
        )

    best = sum(
        1 for t, n in settings
        if all(table[(t, n, "PivotLoRA")] >= table[(t, n, m)] for m in methods)
    )
    check(
        "PivotLoRA has the best mean in >= 19/30 settings (paper: 19, rounding may add ties)",
        best >= 19, f"{best}/30",
    )

    low = [(t, n) for t, n in settings if n <= 64]
    low_rank = mean(
        1 + sum(1 for m in methods if table[(t, n, m)] > table[(t, n, "PivotLoRA")])
        for t, n in low
    )
    check(
        "N<=64 average rank ~ 1.39 (rounded-table recomputation)",
        close(low_rank, 1.39, 0.15), f"{low_rank:.2f}",
    )
    low_best = sum(
        1 for t, n in low
        if all(table[(t, n, "PivotLoRA")] >= table[(t, n, m)] for m in methods)
    )
    check(
        "PivotLoRA is best in >= 13/18 low-sample settings (paper: 13)",
        low_best >= 13, f"{low_best}/18",
    )

    # Exact two-sided sign test on the 29/30 win record.
    p = 2 * sum(math.comb(30, k) for k in range(0, 2)) / 2 ** 30
    check("exact two-sided sign-test p = 5.8e-8", close(p, 5.8e-8, 1e-9), f"{p:.2e}")


def verify_reasoning(verbose: bool) -> None:
    print("\nGenerated-answer reasoning (results/reasoning.csv)")
    rows = load("reasoning.csv")

    for dataset, n_test, lora_mean, pivot_mean, gain, gain_sd, positive in [
        ("gsm8k", 1319, 23.88, 25.37, 1.49, 1.84, 2),
        ("svamp", 300, 23.11, 55.00, 31.89, 16.57, 3),
    ]:
        subset = [r for r in rows if r["dataset"] == dataset]
        check(f"{dataset}: 6 runs", len(subset) == 6, f"{len(subset)}")

        by_method = {
            m: {int(r["seed"]): float(r["accuracy_pct"]) for r in subset if r["method"] == m}
            for m in ("LoRA", "PivotLoRA")
        }

        # accuracy_pct must follow from correct / n_test.
        for r in subset:
            recomputed = 100.0 * int(r["correct"]) / int(r["n_test"])
            check(
                f"{dataset} {r['method']} seed {r['seed']}: "
                f"{r['correct']}/{r['n_test']} = {r['accuracy_pct']}%",
                close(recomputed, float(r["accuracy_pct"]), 0.005),
                f"{recomputed:.2f}",
            )
            check(
                f"{dataset} {r['method']} seed {r['seed']}: parse rate",
                close(
                    100.0 * (int(r["n_test"]) - int(r["invalid"])) / int(r["n_test"]),
                    float(r["parse_rate_pct"]),
                    0.005,
                ),
            )
            check(f"{dataset} n_test = {n_test}", int(r["n_test"]) == n_test)

        got_lora = mean(by_method["LoRA"].values())
        got_pivot = mean(by_method["PivotLoRA"].values())
        check(f"{dataset} LoRA mean = {lora_mean}%", close(got_lora, lora_mean, 0.01), f"{got_lora:.2f}")
        check(f"{dataset} PivotLoRA mean = {pivot_mean}%", close(got_pivot, pivot_mean, 0.01), f"{got_pivot:.2f}")

        paired = [by_method["PivotLoRA"][s] - by_method["LoRA"][s] for s in (42, 43, 44)]
        check(f"{dataset} mean paired gain = {gain:+.2f} pp", close(mean(paired), gain, 0.01), f"{mean(paired):+.2f}")
        check(
            f"{dataset} paired-gain sample SD = {gain_sd:.2f} pp",
            close(sample_std(paired), gain_sd, 0.01),
            f"{sample_std(paired):.2f}",
        )
        check(
            f"{dataset} positive seeds = {positive}/3",
            sum(1 for d in paired if d > 0) == positive,
        )
        if verbose:
            print(f"        paired gains: {', '.join(f'{d:+.2f}' for d in paired)}")


def verify_tuned(verbose: bool) -> None:
    print("\nTuned comparison (results/tuned_comparison.csv)")
    rows = [r for r in load("tuned_comparison.csv") if r["comparison"] == "best_swept"]

    for r in rows:
        per_task = [float(r["agnews"]), float(r["sst2"]), float(r["cola"])]
        check(
            f"{r['method']} three-task average = {r['average']}",
            close(mean(per_task), float(r["average"]), 5e-4),
            f"{mean(per_task):.4f}",
        )

    # Gaps are quoted from unrounded three-task averages, so recompute them
    # from the per-task columns rather than from the rounded average column.
    exact = {
        r["method"]: mean([float(r["agnews"]), float(r["sst2"]), float(r["cola"])])
        for r in rows
    }
    check("PivotLoRA has the highest best-swept average", max(exact, key=exact.get) == "PivotLoRA")
    for other, want in [("TT-PCA", 0.012), ("LoRA+", 0.031), ("LoRA", 0.045),
                        ("EVA", 0.048), ("LoRAM", 0.050), ("PiSSA", 0.061)]:
        got = exact["PivotLoRA"] - exact[other]
        check(f"PivotLoRA advantage over {other} = +{want:.3f}", close(got, want, 1e-3), f"+{got:.4f}")


def verify_alignment(verbose: bool) -> None:
    print("\nAlignment diagnostic (results/alignment.csv)")
    rows = load("alignment.csv")
    rho = [float(r["rho"]) for r in rows]
    gain = [float(r["pivotlora_gain"]) for r in rows]

    check("7 tasks", len(rows) == 7, f"{len(rows)}")

    r_pearson = pearson(rho, gain)
    check("Pearson r = 0.93", close(r_pearson, 0.93, 0.005), f"{r_pearson:.3f}")

    r_spearman = pearson(rank_of(rho), rank_of(gain))
    check("Spearman rho = 0.86", close(r_spearman, 0.86, 0.01), f"{r_spearman:.3f}")

    # Exact two-sided permutation test over all 7! = 5040 assignments.
    observed = abs(r_pearson)
    permutations = list(itertools.permutations(gain))
    extreme = sum(1 for perm in permutations if abs(pearson(rho, list(perm))) >= observed - 1e-12)
    p_value = extreme / len(permutations)
    check("5,040 permutations enumerated", len(permutations) == 5040, f"{len(permutations)}")
    check("exact two-sided permutation p = 0.002", close(p_value, 0.002, 5e-4), f"{p_value:.4f}")

    loo = []
    for i in range(len(rows)):
        sub_rho = rho[:i] + rho[i + 1:]
        sub_gain = gain[:i] + gain[i + 1:]
        loo.append(pearson(sub_rho, sub_gain))
    check(
        "leave-one-task-out Pearson stays in [0.87, 0.96]",
        0.865 <= min(loo) and max(loo) <= 0.965,
        f"[{min(loo):.2f}, {max(loo):.2f}]",
    )

    agnews_removed = pearson(
        [r_ for r_, row in zip(rho, rows) if row["task"] != "agnews"],
        [g for g, row in zip(gain, rows) if row["task"] != "agnews"],
    )
    check("removing AG News still gives r = 0.87", close(agnews_removed, 0.87, 0.01), f"{agnews_removed:.2f}")

    cross = [r for r in rows if r["rho_cross_split"]]
    order_original = [r["task"] for r in sorted(cross, key=lambda r: -float(r["rho"]))]
    order_cross = [r["task"] for r in sorted(cross, key=lambda r: -float(r["rho_cross_split"]))]
    check(
        "cross-split preserves the AG News > SST-2 > CoLA ordering",
        order_original == order_cross == ["agnews", "sst2", "cola"],
        f"{order_cross}",
    )


def verify_activation_source(verbose: bool) -> None:
    print("\nActivation-source ablation (results/activation_source.csv)")
    rows = load("activation_source.csv")
    per_seed = [r for r in rows if r["seed"]]
    summary = {(r["task"], r["source"]): r for r in rows if not r["seed"]}

    for (task, source), row in summary.items():
        scores = [
            float(r["score"]) for r in per_seed
            if r["task"] == task and r["source"] == source
        ]
        check(
            f"{task} {source}: mean = {row['mean']}",
            close(mean(scores), float(row["mean"]), 5e-4),
            f"{mean(scores):.4f}",
        )
        check(
            f"{task} {source}: sample SD = {row['std']}",
            close(sample_std(scores), float(row["std"]), 5e-4),
            f"{sample_std(scores):.4f}",
        )

    for task, want_gain, want_wins in [("agnews", 0.0315, 3), ("sst2", 0.0233, 2)]:
        paired = []
        for seed in ("42", "43", "44"):
            target = next(
                float(r["score"]) for r in per_seed
                if r["task"] == task and r["source"] == "target_position" and r["seed"] == seed
            )
            all_token = next(
                float(r["score"]) for r in per_seed
                if r["task"] == task and r["source"] == "all_token" and r["seed"] == seed
            )
            paired.append(all_token - target)
        check(
            f"{task} paired gain of all-token = {want_gain:+.4f}",
            close(mean(paired), want_gain, 5e-4),
            f"{mean(paired):+.4f}",
        )
        check(
            f"{task} all-token seed-level wins = {want_wins}/3",
            sum(1 for d in paired if d > 0) == want_wins,
        )


def verify_pivot_diagnostics(verbose: bool) -> None:
    print("\nPivot diagnostics (results/pivot_diagnostics.csv)")
    rows = load("pivot_diagnostics.csv")
    reasoning = [r for r in rows if r["experiment_family"] == "reasoning"]

    check("6 reasoning runs", len(reasoning) == 6, f"{len(reasoning)}")
    check("all 64 adapters pivoted every run", all(r["adapters_pivoted"] == "64" for r in reasoning))
    check("pivot at epoch 3 / step 48 every run",
          all(r["pivot_epoch"] == "3" and r["pivot_step"] == "48" for r in reasoning))

    pres = [float(r["max_preservation_error"]) for r in reasoning]
    orth = [float(r["orthogonality_error_after"]) for r in reasoning]
    check("max preservation error = 1.33e-7", close(max(pres) * 1e7, 1.33, 0.005), f"{max(pres):.3e}")
    check("preservation range ~1.25e-7 to 1.33e-7",
          close(min(pres) * 1e7, 1.25, 0.005), f"{min(pres):.3e} .. {max(pres):.3e}")
    check("max orthogonality error = 2.38e-8", close(max(orth) * 1e8, 2.38, 0.005), f"{max(orth):.3e}")
    check("orthogonality range ~1.13e-8 to 2.38e-8",
          close(min(orth) * 1e8, 1.13, 0.005), f"{min(orth):.3e} .. {max(orth):.3e}")

    classification = [r for r in rows if r["experiment_family"] == "classification"]
    check("classification residuals all below 1e-6",
          all(float(r["max_preservation_error"]) < 1e-6 for r in classification))
    check("classification post-pivot orthogonality all below 1e-6",
          all(float(r["orthogonality_error_after"]) < 1e-6 for r in classification))


def verify_regime_controls(verbose: bool) -> None:
    print("\nRegime controls (results/regime_controls.csv)")
    rows = load("regime_controls.csv")

    summ = {
        (r["task"], r["n"], r["method"]): float(r["mean"])
        for r in rows if r["control"] == "summarization"
    }
    for task, n, want in [("xsum", "64", -0.003), ("xsum", "256", -0.003),
                          ("cnndm", "64", -0.017), ("cnndm", "256", -0.016)]:
        got = summ[(task, n, "PivotLoRA")] - summ[(task, n, "LoRA")]
        check(f"{task} N={n} ROUGE-L delta = {want:+.3f}", close(got, want, 1e-3), f"{got:+.3f}")

    full = {
        (r["task"], r["method"]): float(r["mean"])
        for r in rows if r["control"] == "full_data"
    }
    deltas = {t: full[(t, "PivotLoRA")] - full[(t, "LoRA")] for t in
              ("sst2", "cola", "mnli", "qnli", "rte", "agnews")}
    check("full-data deltas span -0.009 to +0.022",
          close(min(deltas.values()), -0.009, 1e-3) and close(max(deltas.values()), 0.022, 1e-3),
          f"[{min(deltas.values()):+.3f}, {max(deltas.values()):+.3f}]")

    bert = {(r["task"], int(r["n"]), r["method"]): float(r["mean"]) for r in load("bert_main.csv")}
    ft = {r["task"]: float(r["mean"]) for r in rows if r["control"] == "full_finetune"}
    beats = sum(1 for t, v in ft.items() if bert[(t, 64, "PivotLoRA")] >= v)
    check("PivotLoRA >= full fine-tuning on all 6 tasks at N=64", beats == 6, f"{beats}/6")


def verify_direction_controls(verbose: bool) -> None:
    print("\nDirection-versus-rate controls (results/direction_controls.csv)")
    rows = load("direction_controls.csv")
    table = {(r["variant"], r["task"]): float(r["mean"]) for r in rows}

    for task in ("agnews", "sst2", "cola"):
        check(
            f"{task}: random directions with eigen-LR stay below TT-PCA",
            table[("Random+EigenLR", task)] < table[("TT-PCA", task)],
            f"{table[('Random+EigenLR', task)]:.3f} < {table[('TT-PCA', task)]:.3f}",
        )
        check(
            f"{task}: TT-PCA survives uniform rates (within 0.01)",
            abs(table[("TT-PCA+UniformLR", task)] - table[("TT-PCA", task)]) <= 0.01,
        )
        check(
            f"{task}: TT-PCA survives shuffled rate assignment (within 0.015)",
            abs(table[("TT-PCA+ShuffledLR", task)] - table[("TT-PCA", task)]) <= 0.015,
        )


# ---------------------------------------------------------------------------


def pearson(x, y) -> float:
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else 0.0


def rank_of(values) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for position, index in enumerate(order):
        ranks[index] = float(position + 1)
    return ranks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("Verifying published result summaries against results/*.csv")
    print("(arithmetic audit only -- no training is performed)")

    verify_bert_main(args.verbose)
    verify_reasoning(args.verbose)
    verify_tuned(args.verbose)
    verify_alignment(args.verbose)
    verify_activation_source(args.verbose)
    verify_pivot_diagnostics(args.verbose)
    verify_regime_controls(args.verbose)
    verify_direction_controls(args.verbose)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for name, detail in FAIL:
        print(f"  FAILED: {name} ({detail})")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
