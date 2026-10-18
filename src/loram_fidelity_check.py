"""
LoRAM fidelity check.

In train_classification.py, LoRALayer multiplies every supplied A initialization by
sqrt(5) / sqrt(d_in). For LoRAM, whose initialization is a random orthonormal
basis scaled by log(1 + sigma_max(W)), this shrinks the intended magnitude by
about 12x on BERT-base. "loram_faithful" divides LoRAM's A by the same factor
before it reaches LoRALayer, so the adapter starts at LoRAM's intended
magnitude. Everything else is the unchanged primary harness.

Runs LoRA, the harness LoRAM, and faithful LoRAM at N=64 on AG News, SST-2, and
CoLA with seeds 42-44, writing to results_loram_fidelity/results.csv (relative to src/).

    python -u loram_fidelity_check.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUTPUT_DIR = os.path.join(HERE, "results_loram_fidelity")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(OUTPUT_DIR)

import numpy as np

import train_classification as harness

harness_loram_init = harness.compute_loram_init
harness_run_experiment = harness.run_experiment


def faithful_loram_init(weight, rank):
    A_init, B_init = harness_loram_init(weight, rank)
    d_in = weight.shape[1]
    layer_scale = np.sqrt(5.0) / np.sqrt(d_in)
    return A_init / layer_scale, B_init


def run_experiment(task, shot, seed, method):
    if method != "loram_faithful":
        return harness_run_experiment(task, shot, seed, method)

    harness.compute_loram_init = faithful_loram_init
    try:
        return harness_run_experiment(task, shot, seed, "loram")
    finally:
        harness.compute_loram_init = harness_loram_init


harness.run_experiment = run_experiment

if __name__ == "__main__":
    harness.SEED_LIST[:] = [42, 43, 44]
    harness.run_study(
        tasks=["agnews", "sst2", "cola"],
        shots=[64],
        methods=["lora", "loram", "loram_faithful"],
    )
