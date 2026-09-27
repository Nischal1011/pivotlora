import sys, os, csv, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_classification as main

# ── Ensure AG News is registered in TASK_INFO ─────────────────────────────────
if "agnews" not in main.TASK_INFO:
    main.TASK_INFO["agnews"] = {
        "dataset":     ("fancyzhx/ag_news",),
        "text_fields": ["text"],
        "num_labels":  4,
        "metric":      "accuracy",
        "is_regression": False,
    }

# ── Default sweep parameters ──────────────────────────────────────────────────
PIVOT_EPOCHS = [1, 2, 3, 5, 7, 10]   # 3 is the main-paper default
TASKS        = ["sst2", "cola", "agnews"]
SHOTS        = [16, 32, 64, 128, 256]
SEEDS        = [42, 43, 44]

RESULTS_CSV  = "ablation_pivot.csv"
CSV_FIELDS   = ["task", "shot", "pivot_epoch", "method", "seed", "score"]


def run_with_pivot(task, shot, seed, pivot_epoch):
    """Run pivotlora_lr with a specific pivot_epoch, restoring the original after."""
    orig = main.PIVOT_EPOCH
    main.PIVOT_EPOCH = pivot_epoch
    try:
        score, wall, peak = main.run_experiment(task, shot, seed, "pivotlora")
    finally:
        main.PIVOT_EPOCH = orig
    return score, wall, peak


def load_completed(path):
    completed = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                key = (row["task"], int(row["shot"]),
                       row["pivot_epoch"], row["method"], int(row["seed"]))
                try:
                    completed[key] = float(row["score"])
                except ValueError:
                    pass
        print(f"[RESUME] Loaded {len(completed)} completed rows from {path}")
    return completed


def run_ablation(tasks, shots, seeds):
    completed = load_completed(RESULTS_CSV)

    csv_fh = open(RESULTS_CSV, "a", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
    if os.path.getsize(RESULTS_CSV) == 0:
        writer.writeheader()

    # all_results[(task, shot, pivot_epoch_str, method)] -> list of scores
    all_results = {}

    def run_cell(task, shot, pivot_str, method, seed, run_fn):
        prefix = (task, shot, pivot_str, method)
        key    = (*prefix, seed)
        if prefix not in all_results:
            all_results[prefix] = []
        if key in completed:
            all_results[prefix].append(completed[key])
            print(f"  [SKIP] {method} pivot={pivot_str} {task}-{shot}-s{seed}: "
                  f"{completed[key]:.4f}")
            return
        try:
            s, wall, _ = run_fn()
            all_results[prefix].append(s)
            completed[key] = s
            writer.writerow({"task": task, "shot": shot, "pivot_epoch": pivot_str,
                             "method": method, "seed": seed, "score": s})
            csv_fh.flush()
            print(f"  [DONE] {method} pivot={pivot_str} {task}-{shot}-s{seed}: "
                  f"{s:.4f}  ({wall:.1f}s)")
        except Exception as e:
            print(f"  CRASH {method} pivot={pivot_str} {task}-{shot}-s{seed}: {e}")
            import traceback; traceback.print_exc()

    # ── random baseline ───────────────────────────────────────────────────────
    print("\n[PHASE 1] Running random baseline...")
    for task in tasks:
        for shot in shots:
            for seed in seeds:
                run_cell(task, shot, "N/A", "lora", seed,
                         lambda t=task, sh=shot, se=seed:
                             main.run_experiment(t, sh, se, "lora"))

    # ── pivotlora sweep ─────────────────────────────────────────────────────────
    print("\n[PHASE 2] Sweeping pivot epochs...")
    for pivot_epoch in PIVOT_EPOCHS:
        print(f"\n  --- pivot_epoch = {pivot_epoch} ---")
        for task in tasks:
            for shot in shots:
                for seed in seeds:
                    run_cell(task, shot, str(pivot_epoch), "pivotlora", seed,
                             lambda t=task, sh=shot, se=seed, pe=pivot_epoch:
                                 run_with_pivot(t, sh, se, pe))

    csv_fh.close()
    print_tables(all_results, tasks, shots)


def print_tables(all_results, tasks, shots):
    col_w = 15
    default_pe = main.PIVOT_EPOCH

    print("\n" + "="*110)
    print("PIVOT TIMING ABLATION — PivotLoRA (mean +/- std) vs random baseline")
    print("* marks default pivot_epoch=" + str(default_pe) + " used in main paper")
    print("="*110)

    def fmt_cell(scores):
        if not scores:
            return "CRASH"
        return f"{np.mean(scores):.3f}+/-{np.std(scores):.3f}"

    for task in tasks:
        info = main.TASK_INFO[task]
        print(f"\n  TASK: {task.upper()}  |  Metric: {info['metric'].upper()}")
        header = f"  {'Config':<26}" + "".join(f"{str(s)+'-shot':^{col_w}}" for s in shots)
        sep    = f"  {'─'*26}" + f"{'─'*col_w}" * len(shots)
        print(sep)
        print(header)
        print(sep)

        # random row
        row = f"  {'random (baseline)':<26}"
        for shot in shots:
            sc = all_results.get((task, shot, "N/A", "lora"), [])
            row += f"{fmt_cell(sc):^{col_w}}"
        print(row)
        print(f"  {'─'*26}" + f"{'─'*col_w}" * len(shots))

        # one row per pivot epoch
        for pe in PIVOT_EPOCHS:
            marker = "*" if pe == default_pe else " "
            label  = f"pivotlora pivot={pe}{marker}"
            row    = f"  {label:<26}"
            for shot in shots:
                sc = all_results.get((task, shot, str(pe), "pivotlora"), [])
                row += f"{fmt_cell(sc):^{col_w}}"
            print(row)

        print(sep)

    # Delta table
    print("\n  DELTA vs random (positive = better than random):")
    print(f"  {'─'*26}" + f"{'─'*col_w}" * len(shots))
    for task in tasks:
        for pe in PIVOT_EPOCHS:
            marker = "*" if pe == default_pe else " "
            label  = f"p={pe}{marker} {task}"
            row    = f"  {label:<26}"
            for shot in shots:
                base = all_results.get((task, shot, "N/A", "lora"), [])
                warm = all_results.get((task, shot, str(pe), "pivotlora"), [])
                if not base or not warm:
                    cell = "N/A"
                else:
                    delta = np.mean(warm) - np.mean(base)
                    cell = f"{delta:+.3f}"
                row += f"{cell:^{col_w}}"
            print(row)

    print(f"\nResults saved to: {RESULTS_CSV}")
    print("="*110)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pivot epoch ablation for PivotLoRA")
    parser.add_argument("--seeds",        type=int,  default=None,
                        help="Use first N seeds (default: 3 -> [42,43,44])")
    parser.add_argument("--tasks",        nargs="+", default=None,
                        help="Tasks (default: sst2 cola agnews)")
    parser.add_argument("--shots",        nargs="+", type=int, default=None,
                        help="Shot counts (default: 16 32 64 128 256)")
    parser.add_argument("--pivot_epochs", nargs="+", type=int, default=None,
                        help="Pivot epochs to sweep (default: 1 2 3 5 7 10)")
    args = parser.parse_args()

    seeds = SEEDS[: args.seeds] if args.seeds else SEEDS
    tasks = args.tasks          or TASKS
    shots = args.shots          or SHOTS
    if args.pivot_epochs:
        PIVOT_EPOCHS[:] = args.pivot_epochs

    main.SEED_LIST[:] = seeds

    print(f"[CONFIG] pivot_epochs = {PIVOT_EPOCHS}")
    print(f"[CONFIG] tasks={tasks}  shots={shots}  seeds={seeds}")
    run_ablation(tasks, shots, seeds)