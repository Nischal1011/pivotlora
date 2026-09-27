import sys, os, csv, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_classification as main

# ── Ensure AG News is registered ──────────────────────────────────────────────
if "agnews" not in main.TASK_INFO:
    main.TASK_INFO["agnews"] = {
        "dataset":     ("fancyzhx/ag_news",),
        "text_fields": ["text"],
        "num_labels":  4,
        "metric":      "accuracy",
        "is_regression": False,
        "task_type":   "cls",
    }

# ── Monkey-patch PivotCallback to support skip_reset mode ───────────────────
_original_on_epoch_end = main.PivotCallback.on_epoch_end

def _patched_on_epoch_end(self, args, state, control, model=None, optimizer=None, **kwargs):
    """Wrapper that optionally skips the optimizer state reset."""
    # Store the original pivot behavior
    skip_reset = getattr(self, '_skip_reset', False)

    if skip_reset:
        # Run the pivot but intercept the optimizer reset
      
        opt = optimizer or self._optimizer

        # Call the original, which does pivot + reset
        _original_on_epoch_end(self, args, state, control, model=model, optimizer=optimizer, **kwargs)

    else:
        _original_on_epoch_end(self, args, state, control, model=model, optimizer=optimizer, **kwargs)


class PivotLoRANoResetCallback(main.PivotCallback):
    """
    PivotLoRA pivot that does everything EXCEPT reset the optimizer state for A.
    The subspace correction, B re-projection, and B warmup all happen normally.
    Only the Adam state reset for lora_A is skipped.
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
        print(f"\n[PivotLoRA-NoReset] === PIVOT at epoch {epoch_f:.3f} (target={self.pivot_epoch}) ===")
        k = max(1, int(main.RANK * self.k_fraction))
        pivoted = 0

        for name, m in model.named_modules():
            if not isinstance(m, main.LoRALayer) or m.A_init_snapshot is None:
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
                A_new = Q_aug[:, :main.RANK]

                # B-preserving re-projection (same as original)
                proj  = A_new.T @ A_current
                B_old = m.lora_B_matrix
                B_new = proj @ B_old

                m.lora_A.data.copy_(A_new)
                if m._split_b:
                    for ri in range(m.rank):
                        m.lora_B_rows[ri].data.copy_(B_new[ri:ri+1, :])
                else:
                    m.lora_B.data.copy_(B_new)

                # *** KEY DIFFERENCE: NO optimizer state reset for A ***
                # The original PivotCallback zeros exp_avg and exp_avg_sq here.
                # We skip that entirely.

                print(f"  {name}: norm={delta_norm:.3f} S[:3]={S[:3].tolist()} k={k} (NO RESET)")
                pivoted += 1

        print(f"[PivotLoRA-NoReset] Pivot done: {pivoted} layers. Starting {self.warmup_steps}-step B warmup. (A optimizer state KEPT)")
        self.pivot_done = True
        self.warmup_remaining = self.warmup_steps
        if opt is not None:
            self._set_b_lr(opt, self.base_lr)


import torch  # needed for the callback

# ── Default parameters ────────────────────────────────────────────────────────
TASKS       = ["sst2", "cola", "agnews"]
SHOTS       = [16, 32, 64, 128, 256]
SEEDS       = [42, 43, 44]
METHODS     = ["eigenlora", "pivotlora", "pivot_noreset"]

RESULTS_CSV = "ablation_reset.csv"
CSV_FIELDS  = ["task", "shot", "method", "seed", "score"]


def run_pivot_noreset(task, shot, seed):
    """
    Run PivotLoRA but with the NoReset callback instead of the standard one.
    We do this by temporarily monkey-patching the callback creation in run_experiment.
    
    Simpler approach: just call run_experiment with pivotlora_lr, but swap the callback.
    """
    # We need to intercept the callback creation. The cleanest way is to
    # directly replicate the relevant parts of run_experiment for this variant.
    # But that's fragile. Instead, we'll use a flag approach.

    # Save original callback class
    original_callback_class = main.PivotCallback

    # Replace with our no-reset version
    main.PivotCallback = PivotLoRANoResetCallback

    try:
        score, wall, peak = main.run_experiment(task, shot, seed, "pivotlora")
    finally:
        # Restore original
        main.PivotCallback = original_callback_class

    return score, wall, peak


def load_completed(path):
    completed = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                key = (row["task"], int(row["shot"]), row["method"], int(row["seed"]))
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
    if os.path.getsize(RESULTS_CSV) == 0 if os.path.exists(RESULTS_CSV) else True:
        writer.writeheader()

    all_results = {}

    def run_cell(task, shot, method, seed):
        key = (task, shot, method, seed)
        prefix = (task, shot, method)
        if prefix not in all_results:
            all_results[prefix] = []
        if key in completed:
            all_results[prefix].append(completed[key])
            print(f"  [SKIP] {method} {task}-{shot}-s{seed}: {completed[key]:.4f}")
            return
        try:
            if method == "pivot_noreset":
                s, wall, _ = run_pivot_noreset(task, shot, seed)
            else:
                s, wall, _ = main.run_experiment(task, shot, seed, method)
            all_results[prefix].append(s)
            completed[key] = s
            writer.writerow({"task": task, "shot": shot,
                             "method": method, "seed": seed, "score": s})
            csv_fh.flush()
            print(f"  [DONE] {method} {task}-{shot}-s{seed}: {s:.4f}  ({wall:.1f}s)")
        except Exception as e:
            print(f"  CRASH {method} {task}-{shot}-s{seed}: {e}")
            import traceback; traceback.print_exc()

    for task in tasks:
        for shot in shots:
            print(f"\n>>> {task.upper()} {shot}-shot <<<")
            for method in METHODS:
                for seed in seeds:
                    run_cell(task, shot, method, seed)

    csv_fh.close()
    print_results(all_results, tasks, shots)


def print_results(all_results, tasks, shots):
    col_w = 18

    print("\n" + "="*110)
    print("OPTIMIZER RESET ABLATION")
    print("  pca_lr         = PCA init, no pivot (baseline)")
    print("  pivotlora_lr     = full PivotLoRA (pivot + optimizer reset)")
    print("  pivot_noreset = PivotLoRA pivot WITHOUT optimizer reset")
    print("="*110)

    def fmt(scores):
        if not scores:
            return "N/A"
        return f"{np.mean(scores):.4f}±{np.std(scores):.4f}"

    for task in tasks:
        info = main.TASK_INFO[task]
        print(f"\n  TASK: {task.upper()} | Metric: {info['metric'].upper()}")
        print(f"  {'Method':<22}", end="")
        for s in shots:
            print(f"{'%d-shot' % s:^{col_w}}", end="")
        print()
        print(f"  {'─'*22}" + f"{'─'*col_w}" * len(shots))

        for method in METHODS:
            print(f"  {method:<22}", end="")
            for shot in shots:
                sc = all_results.get((task, shot, method), [])
                print(f"{fmt(sc):^{col_w}}", end="")
            print()

        # Interpretation row
        print(f"\n  Interpretation:")
        print(f"  {'─'*22}" + f"{'─'*col_w}" * len(shots))
        print(f"  {'pivot contribution':<22}", end="")
        for shot in shots:
            pca = all_results.get((task, shot, "eigenlora"), [])
            warm = all_results.get((task, shot, "pivotlora"), [])
            if pca and warm:
                d = np.mean(warm) - np.mean(pca)
                print(f"{d:+.4f}           ", end="")
            else:
                print(f"{'N/A':^{col_w}}", end="")
        print("  (pivotlora - pca_lr)")

        print(f"  {'reset contribution':<22}", end="")
        for shot in shots:
            noreset = all_results.get((task, shot, "pivot_noreset"), [])
            warm = all_results.get((task, shot, "pivotlora"), [])
            if noreset and warm:
                d = np.mean(warm) - np.mean(noreset)
                print(f"{d:+.4f}           ", end="")
            else:
                print(f"{'N/A':^{col_w}}", end="")
        print("  (pivotlora - noreset)")

        print(f"  {'subspace contribution':<22}", end="")
        for shot in shots:
            pca = all_results.get((task, shot, "eigenlora"), [])
            noreset = all_results.get((task, shot, "pivot_noreset"), [])
            if pca and noreset:
                d = np.mean(noreset) - np.mean(pca)
                print(f"{d:+.4f}           ", end="")
            else:
                print(f"{'N/A':^{col_w}}", end="")
        print("  (noreset - pca_lr)")

    print("\n" + "="*110)
    print("If 'reset contribution' >> 'subspace contribution': optimizer reset is the key mechanism")
    print("If 'subspace contribution' >> 'reset contribution': subspace correction is key")
    print("If both contribute: the mechanisms are complementary")
    print("="*110)
    print(f"\nResults saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimizer reset ablation for PivotLoRA")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Use first N seeds (default: 3)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Tasks (default: sst2 cola agnews)")
    parser.add_argument("--shots", nargs="+", type=int, default=None,
                        help="Shot counts (default: 16 32 64 128 256)")
    args = parser.parse_args()

    seeds = SEEDS[:args.seeds] if args.seeds else SEEDS
    tasks = args.tasks or TASKS
    shots = args.shots or SHOTS

    main.SEED_LIST[:] = seeds

    print(f"[CONFIG] tasks={tasks}  shots={shots}  seeds={seeds}")
    print(f"[CONFIG] methods={METHODS}")
    run_ablation(tasks, shots, seeds)