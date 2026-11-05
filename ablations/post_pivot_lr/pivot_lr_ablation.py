"""Post-pivot B learning-rate ablation for classification PivotLoRA (BERT-base).

All pivot variants share the PCA initialization, eigenvalue-scaled B rates before
the pivot, the QR pivot at K=3, and the Adam-state reset for A. They differ only in
the B learning rate after the pivot. The Hugging Face Trainer builds a linear-decay
scheduler from each group's initial rate and rewrites group["lr"] every step, so
persistent changes are applied after each scheduler step (on_step_end) as
target_rate * decay, where decay is read from the A group.

Variants
  ttpca           no pivot (method "eigenlora")
  as_run          current code: B dips to eta_A and ramps to 16*eta_A over 10 steps,
                  then the scheduler restores the eigenvalue-scaled rates
  no_switch       pivot only; B keeps the eigenvalue-scaled rates
  uniform         after the pivot B uses 16*eta_A (with the normal decay), no ramp
  uniform_ramp    10-step ramp, then 16*eta_A (with the normal decay) to the end
"""
import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))
import train_classification as main  # noqa: E402

BasePivot = main.PivotCallback


def _decay(opt, a_lr0):
    """Current decay factor of the shared scheduler, read from the A group."""
    for g in opt.param_groups:
        if not g.get("_is_B_group", False):
            return g["lr"] / a_lr0
    return 1.0


class NoSwitch(BasePivot):
    def _set_b_lr(self, opt, lr):
        pass

    def on_step_end(self, *a, **k):
        return


class Uniform(BasePivot):
    ramp = False

    def on_epoch_end(self, args, state, control, model=None, optimizer=None, **kw):
        was = self.pivot_done
        super().on_epoch_end(args, state, control, model=model, optimizer=optimizer, **kw)
        if self.pivot_done and not was:
            self._steps_after = 0
            self._apply(optimizer or self._optimizer)

    def _set_b_lr(self, opt, lr):
        pass  # handled in _apply

    def _apply(self, opt):
        if opt is None:
            return
        d = _decay(opt, self.base_lr)
        target = self.lr_ratio * self.base_lr
        if self.ramp and self._steps_after < self.warmup_steps:
            frac = (self._steps_after + 1) / self.warmup_steps
            target = self.base_lr + frac * (self.lr_ratio - 1) * self.base_lr
        for g in opt.param_groups:
            if g.get("_is_B_group", False):
                g["lr"] = target * d

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kw):
        if not self.pivot_done:
            return
        self._steps_after += 1
        self._apply(optimizer or self._optimizer)


class UniformRamp(Uniform):
    ramp = True


VARIANTS = {
    "as_run": BasePivot,
    "no_switch": NoSwitch,
    "uniform": Uniform,
    "uniform_ramp": UniformRamp,
}
ORDER = ["ttpca", "as_run", "no_switch", "uniform", "uniform_ramp"]
FIELDS = ["task", "shot", "variant", "seed", "score"]


def run_one(task, shot, seed, variant):
    if variant == "ttpca":
        main.PivotCallback = BasePivot
        return main.run_experiment(task, shot, seed, "eigenlora")[0]
    main.PivotCallback = VARIANTS[variant]
    try:
        return main.run_experiment(task, shot, seed, "pivotlora")[0]
    finally:
        main.PivotCallback = BasePivot


def main_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["sst2", "cola", "agnews"])
    p.add_argument("--shots", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--variants", nargs="+", default=ORDER)
    p.add_argument("--out", default="pivot_lr_ablation.csv")
    a = p.parse_args()

    done = set()
    if os.path.exists(a.out):
        with open(a.out, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["task"], int(r["shot"]), r["variant"], int(r["seed"])))
    new = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for task in a.tasks:
            for shot in a.shots:
                for seed in a.seeds:
                    for v in a.variants:
                        if (task, shot, v, seed) in done:
                            continue
                        score = run_one(task, shot, seed, v)
                        w.writerow({"task": task, "shot": shot, "variant": v,
                                    "seed": seed, "score": f"{score:.6f}"})
                        fh.flush()
                        print(f"[ABL] {task} N={shot} s{seed} {v}: {score:.4f}", flush=True)


if __name__ == "__main__":
    main_cli()
