from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import math
import os
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# =============================================================================
# PATHS AND HARNESS IMPORT
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# train_classification.py must be in this directory.
HARNESS_DIR = SCRIPT_DIR
HARNESS_MODULE = "train_classification"

RESULTS_DIR = SCRIPT_DIR / "results_jcmz"
RAW_RESULTS_CSV = RESULTS_DIR / "lr_sweep_jcmz_main_harness.csv"
FAILURES_CSV = RESULTS_DIR / "failures.csv"
MANIFEST_JSON = RESULTS_DIR / "manifest.json"

TUNED_BY_TASK_CSV = RESULTS_DIR / "tuned_by_task.csv"
TUNED_SUMMARY_CSV = RESULTS_DIR / "tuned_summary.csv"
ALL_CONFIG_SUMMARY_CSV = RESULTS_DIR / "all_config_summary.csv"
LATEX_TABLE_TXT = RESULTS_DIR / "tuned_table.tex"
PIVOT_REPORT_JSON = RESULTS_DIR / "pivot_verification.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

try:
    harness = importlib.import_module(HARNESS_MODULE)
except Exception as exc:
    raise SystemExit(
        f"\nCould not import harness module {HARNESS_MODULE!r} "
        f"from {str(HARNESS_DIR)!r}.\n\n"
        f"Expected file:\n"
        f"  {HARNESS_DIR / 'train_classification.py'}\n\n"
        f"Original error: {type(exc).__name__}: {exc}\n"
    ) from exc


# =============================================================================
# SWEEP CONFIGURATION
# =============================================================================

TASKS = ["agnews", "sst2", "cola"]
SEEDS = [42, 43, 44]
SHOT = 64
EPOCHS = 15

LEARNING_RATES = [
    5e-5,
    2e-4,
    8e-4,
]

# Four ratios produce four independently tuned LoRA+ configurations.
LORA_PLUS_RATIOS = [
    4.0,
    8.0,
    16.0,
    32.0,
]

ANCHOR_LEARNING_RATE = 2e-4

# Public name -> harness METHOD_CONFIGS key.
METHOD_KEYS = {
    "LoRA": "lora",
    "LoRA+": "lora_plus",
    "EVA": "eva",
    "LoRAM": "loram",
    "TT-PCA": "eigenlora",
    "PivotLoRA": "pivotlora",
}

METHOD_ORDER = [
    "LoRA",
    "LoRA+",
    "EVA",
    "LoRAM",
    "TT-PCA",
    "PivotLoRA",
]

TASK_DISPLAY_NAMES = {
    "agnews": "AG News",
    "sst2": "SST-2",
    "cola": "CoLA",
}

TASK_METRICS = {
    "agnews": "accuracy",
    "sst2": "accuracy",
    "cola": "matthews",
}

RAW_FIELDS = [
    "run_id",
    "timestamp_utc",
    "protocol",
    "task",
    "metric",
    "shot",
    "seed",
    "method",
    "method_key",
    "lora_plus_ratio",
    "learning_rate",
    "epochs",
    "score",
    "wall_seconds",
    "peak_memory_mb",
    "status",
    "harness_module",
    "harness_file",
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class RunSpec:
    protocol: str
    task: str
    shot: int
    seed: int
    method: str
    method_key: str
    learning_rate: float
    epochs: int
    lora_plus_ratio: Optional[float] = None

    @property
    def run_id(self) -> str:
        ratio = (
            "none"
            if self.lora_plus_ratio is None
            else format_float(self.lora_plus_ratio)
        )

        return "__".join(
            [
                self.protocol,
                self.task,
                f"shot-{self.shot}",
                f"seed-{self.seed}",
                safe_name(self.method),
                f"key-{self.method_key}",
                f"lr-{format_float(self.learning_rate)}",
                f"ratio-{ratio}",
                f"epochs-{self.epochs}",
            ]
        )


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str) -> str:
    return (
        value.lower()
        .replace("+", "plus")
        .replace("/", "-")
        .replace(" ", "-")
    )


def format_float(value: float) -> str:
    return f"{float(value):.12g}"


def parse_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None

    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan", "null"}:
        return None

    return float(text)


def ensure_csv(path: Path, fieldnames: Sequence[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()


def append_csv_row(
    path: Path,
    fieldnames: Sequence[str],
    row: Dict[str, object],
) -> None:
    ensure_csv(path, fieldnames)

    normalized = {field: row.get(field, "") for field in fieldnames}

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(normalized)
        file.flush()
        os.fsync(file.fileno())


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []

    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def completed_run_ids() -> set:
    completed = set()

    for row in read_csv_rows(RAW_RESULTS_CSV):
        if row.get("status") == "ok" and row.get("run_id"):
            completed.add(row["run_id"])

    return completed


def cleanup_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def validate_selection(
    requested: Optional[Sequence[str]],
    available: Sequence[str],
    label: str,
) -> List[str]:
    if not requested:
        return list(available)

    available_lookup = {x.lower(): x for x in available}
    selected = []

    for item in requested:
        key = item.lower()

        if key not in available_lookup:
            raise ValueError(
                f"Unknown {label} {item!r}. "
                f"Available: {', '.join(available)}"
            )

        value = available_lookup[key]

        if value not in selected:
            selected.append(value)

    return selected


# =============================================================================
# HARNESS VALIDATION
# =============================================================================

def validate_harness() -> None:
    required_attributes = [
        "TASK_INFO",
        "METHOD_CONFIGS",
        "LEARNING_RATE",
        "EPOCHS",
        "SEED_LIST",
        "run_experiment",
        "PivotCallback",
        "LoRALayer",
    ]

    missing = [
        name for name in required_attributes
        if not hasattr(harness, name)
    ]

    if missing:
        raise RuntimeError(
            "train_classification.py is missing required attributes:\n  "
            + "\n  ".join(missing)
        )

    missing_tasks = [
        task for task in TASKS
        if task not in harness.TASK_INFO
    ]

    if missing_tasks:
        raise RuntimeError(
            "Missing task keys in train_classification.TASK_INFO:\n  "
            + "\n  ".join(missing_tasks)
        )

    missing_methods = [
        method_key
        for method_key in METHOD_KEYS.values()
        if method_key not in harness.METHOD_CONFIGS
    ]

    if missing_methods:
        raise RuntimeError(
            "Missing method keys in train_classification.METHOD_CONFIGS:\n  "
            + "\n  ".join(missing_methods)
        )


def run_probe() -> None:
    print("=" * 78)
    print("JCmz HARNESS PROBE")
    print("=" * 78)

    expected_file = HARNESS_DIR / "train_classification.py"
    imported_file = Path(harness.__file__).resolve()

    print(f"Script directory : {SCRIPT_DIR}")
    print(f"Harness directory: {HARNESS_DIR}")
    print(f"Harness module   : {HARNESS_MODULE}")
    print(f"Expected file    : {expected_file}")
    print(f"File exists      : {expected_file.is_file()}")
    print(f"Imported from    : {imported_file}")
    print()

    validate_harness()

    print("Harness globals")
    print("-" * 78)
    print(f"DEVICE           : {getattr(harness, 'DEVICE', '<missing>')}")
    print(f"MODEL_NAME       : {getattr(harness, 'MODEL_NAME', '<missing>')}")
    print(f"RANK             : {getattr(harness, 'RANK', '<missing>')}")
    print(f"ALPHA            : {getattr(harness, 'ALPHA', '<missing>')}")
    print(f"BATCH_SIZE       : {getattr(harness, 'BATCH_SIZE', '<missing>')}")
    print(f"LEARNING_RATE    : {getattr(harness, 'LEARNING_RATE', '<missing>')}")
    print(f"EPOCHS           : {getattr(harness, 'EPOCHS', '<missing>')}")
    print(f"SEED_LIST        : {getattr(harness, 'SEED_LIST', '<missing>')}")
    print()

    print("Method mapping")
    print("-" * 78)

    for display_name in METHOD_ORDER:
        method_key = METHOD_KEYS[display_name]
        cfg = harness.METHOD_CONFIGS[method_key]

        print(
            f"{display_name:<12} -> {method_key:<12} "
            f"subspace={cfg.get('subspace_type')} "
            f"ratio={cfg.get('lr_ratio')} "
            f"pivot={cfg.get('use_pivot', False)}"
        )

    print()
    print("Task mapping")
    print("-" * 78)

    for task in TASKS:
        info = harness.TASK_INFO[task]
        print(
            f"{task:<12} metric={info.get('metric')} "
            f"labels={info.get('num_labels')} "
            f"dataset={info.get('dataset')}"
        )

    print()
    print("Sweep")
    print("-" * 78)
    print(f"Tasks            : {TASKS}")
    print(f"Shot             : {SHOT}")
    print(f"Seeds            : {SEEDS}")
    print(f"Learning rates   : {LEARNING_RATES}")
    print(f"LoRA+ ratios     : {LORA_PLUS_RATIOS}")
    print(f"Epochs           : {EPOCHS}")
    print(f"Raw output       : {RAW_RESULTS_CSV}")

    # 5 configurations that do not vary ratio plus 4 LoRA+ ratios.
    configs_per_lr = 5 + len(LORA_PLUS_RATIOS)
    total = (
        len(TASKS)
        * len(SEEDS)
        * len(LEARNING_RATES)
        * configs_per_lr
    )

    print(f"Full run count   : {total}")
    print()
    print("PROBE PASSED")
    print("=" * 78)


# =============================================================================
# MANIFEST
# =============================================================================

def write_manifest() -> None:
    harness_path = Path(harness.__file__).resolve()

    payload = {
        "created_utc": utc_now(),
        "driver_file": str(Path(__file__).resolve()),
        "harness_module": HARNESS_MODULE,
        "harness_file": str(harness_path),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(getattr(harness, "DEVICE", "unknown")),
        "model_name": getattr(harness, "MODEL_NAME", None),
        "rank": getattr(harness, "RANK", None),
        "alpha": getattr(harness, "ALPHA", None),
        "batch_size": getattr(harness, "BATCH_SIZE", None),
        "tasks": TASKS,
        "shot": SHOT,
        "seeds": SEEDS,
        "learning_rates": LEARNING_RATES,
        "lora_plus_ratios": LORA_PLUS_RATIOS,
        "epochs": EPOCHS,
        "method_keys": METHOD_KEYS,
    }

    try:
        import transformers
        payload["transformers"] = transformers.__version__
    except Exception:
        payload["transformers"] = None

    try:
        import datasets
        payload["datasets"] = datasets.__version__
    except Exception:
        payload["datasets"] = None

    try:
        import peft
        payload["peft"] = peft.__version__
    except Exception:
        payload["peft"] = None

    with MANIFEST_JSON.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


# =============================================================================
# RUN CONSTRUCTION
# =============================================================================

def build_full_specs(
    tasks: Sequence[str],
    seeds: Sequence[int],
    methods: Sequence[str],
) -> List[RunSpec]:
    specs = []

    for task in tasks:
        for seed in seeds:
            for learning_rate in LEARNING_RATES:
                for method in methods:
                    method_key = METHOD_KEYS[method]

                    if method == "LoRA+":
                        for ratio in LORA_PLUS_RATIOS:
                            specs.append(
                                RunSpec(
                                    protocol="full",
                                    task=task,
                                    shot=SHOT,
                                    seed=seed,
                                    method=method,
                                    method_key=method_key,
                                    learning_rate=learning_rate,
                                    epochs=EPOCHS,
                                    lora_plus_ratio=ratio,
                                )
                            )
                    else:
                        specs.append(
                            RunSpec(
                                protocol="full",
                                task=task,
                                shot=SHOT,
                                seed=seed,
                                method=method,
                                method_key=method_key,
                                learning_rate=learning_rate,
                                epochs=EPOCHS,
                                lora_plus_ratio=None,
                            )
                        )

    return specs


def build_anchor_specs(
    tasks: Sequence[str],
    seeds: Sequence[int],
) -> List[RunSpec]:
    specs = []

    # These rows use protocol="full" so a later --full invocation reuses them.
    for task in tasks:
        for seed in seeds:
            for method in ["LoRA", "PivotLoRA"]:
                specs.append(
                    RunSpec(
                        protocol="full",
                        task=task,
                        shot=SHOT,
                        seed=seed,
                        method=method,
                        method_key=METHOD_KEYS[method],
                        learning_rate=ANCHOR_LEARNING_RATE,
                        epochs=EPOCHS,
                    )
                )

    return specs


# =============================================================================
# HARNESS EXECUTION
# =============================================================================

def call_harness(spec: RunSpec) -> Tuple[float, float, float]:
    """
    Execute one experiment while temporarily overriding harness globals.

    train_classification.run_experiment reads LEARNING_RATE and EPOCHS directly from
    the imported module, so they must be changed before invocation.
    """
    validate_harness()

    if spec.method_key not in harness.METHOD_CONFIGS:
        raise KeyError(
            f"Method key {spec.method_key!r} is missing from "
            "train_classification.METHOD_CONFIGS."
        )

    old_lr = harness.LEARNING_RATE
    old_epochs = harness.EPOCHS
    old_seed_list = list(harness.SEED_LIST)
    old_method_cfg = dict(harness.METHOD_CONFIGS[spec.method_key])

    try:
        harness.LEARNING_RATE = float(spec.learning_rate)
        harness.EPOCHS = int(spec.epochs)
        harness.SEED_LIST[:] = [int(spec.seed)]

        # LoRA+ ratio is stored in METHOD_CONFIGS.
        if spec.method == "LoRA+":
            if spec.lora_plus_ratio is None:
                raise ValueError("LoRA+ requires lora_plus_ratio.")

            harness.METHOD_CONFIGS[spec.method_key]["lr_ratio"] = float(
                spec.lora_plus_ratio
            )

        print()
        print("#" * 88)
        print(f"RUN ID     : {spec.run_id}")
        print(f"TASK       : {spec.task}")
        print(f"METHOD     : {spec.method} -> {spec.method_key}")
        print(f"SEED       : {spec.seed}")
        print(f"SHOT       : {spec.shot}")
        print(f"BASE LR    : {spec.learning_rate:.3e}")
        print(f"EPOCHS     : {spec.epochs}")

        if spec.lora_plus_ratio is not None:
            print(f"LORA+ RATIO: {spec.lora_plus_ratio:g}")
            print(
                f"LORA+ B LR : "
                f"{spec.learning_rate * spec.lora_plus_ratio:.3e}"
            )

        print("#" * 88)

        cleanup_memory()

        result = harness.run_experiment(
            task=spec.task,
            shot=spec.shot,
            seed=spec.seed,
            method=spec.method_key,
        )

        if isinstance(result, tuple):
            score = float(result[0])
            wall_seconds = (
                float(result[1])
                if len(result) >= 2
                else float("nan")
            )
            peak_memory_mb = (
                float(result[2])
                if len(result) >= 3
                else float("nan")
            )
        else:
            score = float(result)
            wall_seconds = float("nan")
            peak_memory_mb = float("nan")

        if not math.isfinite(score):
            raise RuntimeError(f"Non-finite score returned: {score}")

        return score, wall_seconds, peak_memory_mb

    finally:
        harness.LEARNING_RATE = old_lr
        harness.EPOCHS = old_epochs
        harness.SEED_LIST[:] = old_seed_list

        harness.METHOD_CONFIGS[spec.method_key].clear()
        harness.METHOD_CONFIGS[spec.method_key].update(old_method_cfg)

        cleanup_memory()


def execute_specs(
    specs: Sequence[RunSpec],
    force: bool = False,
    stop_on_error: bool = False,
) -> None:
    ensure_csv(RAW_RESULTS_CSV, RAW_FIELDS)
    write_manifest()

    completed = completed_run_ids()
    total = len(specs)

    print(f"\nScheduled runs: {total}")
    print(f"Cached successful runs: {len(completed)}")
    print(f"Output: {RAW_RESULTS_CSV}\n")

    completed_this_call = 0
    skipped = 0
    failed = 0

    for index, spec in enumerate(specs, start=1):
        print()
        print("=" * 88)
        print(f"SWEEP PROGRESS: {index}/{total}")
        print("=" * 88)

        if spec.run_id in completed and not force:
            print(f"[SKIP] Already completed: {spec.run_id}")
            skipped += 1
            continue

        started = time.perf_counter()

        try:
            score, wall_seconds, peak_memory_mb = call_harness(spec)

            total_driver_seconds = time.perf_counter() - started

            row = {
                "run_id": spec.run_id,
                "timestamp_utc": utc_now(),
                "protocol": spec.protocol,
                "task": spec.task,
                "metric": TASK_METRICS[spec.task],
                "shot": spec.shot,
                "seed": spec.seed,
                "method": spec.method,
                "method_key": spec.method_key,
                "lora_plus_ratio": (
                    ""
                    if spec.lora_plus_ratio is None
                    else format_float(spec.lora_plus_ratio)
                ),
                "learning_rate": format_float(spec.learning_rate),
                "epochs": spec.epochs,
                "score": format_float(score),
                "wall_seconds": format_float(
                    wall_seconds
                    if math.isfinite(wall_seconds)
                    else total_driver_seconds
                ),
                "peak_memory_mb": (
                    format_float(peak_memory_mb)
                    if math.isfinite(peak_memory_mb)
                    else ""
                ),
                "status": "ok",
                "harness_module": HARNESS_MODULE,
                "harness_file": str(Path(harness.__file__).resolve()),
            }

            append_csv_row(RAW_RESULTS_CSV, RAW_FIELDS, row)
            completed.add(spec.run_id)
            completed_this_call += 1

            print(
                f"\n[DONE] {spec.run_id}\n"
                f"       score={score:.6f}"
            )

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            raise

        except Exception as exc:
            failed += 1

            print(f"\n[FAILED] {spec.run_id}")
            print(f"{type(exc).__name__}: {exc}")
            traceback.print_exc()

            failure_fields = [
                "timestamp_utc",
                "run_id",
                "task",
                "seed",
                "method",
                "learning_rate",
                "lora_plus_ratio",
                "error_type",
                "error",
            ]

            append_csv_row(
                FAILURES_CSV,
                failure_fields,
                {
                    "timestamp_utc": utc_now(),
                    "run_id": spec.run_id,
                    "task": spec.task,
                    "seed": spec.seed,
                    "method": spec.method,
                    "learning_rate": spec.learning_rate,
                    "lora_plus_ratio": (
                        ""
                        if spec.lora_plus_ratio is None
                        else spec.lora_plus_ratio
                    ),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

            if stop_on_error:
                raise

    print()
    print("=" * 88)
    print("EXECUTION SUMMARY")
    print("=" * 88)
    print(f"Completed now : {completed_this_call}")
    print(f"Skipped       : {skipped}")
    print(f"Failed        : {failed}")
    print(f"Raw results   : {RAW_RESULTS_CSV}")
    print("=" * 88)


# =============================================================================
# PIVOT VERIFICATION
# =============================================================================

def run_verify_pivot(force: bool = False) -> None:
    """
    Execute a PivotLoRA run and instrument PivotCallback.on_epoch_end.

    Verification passes only if:
      1. the pivot callback fires;
      2. pivot_done becomes True; and
      3. at least one LoRALayer changes at the pivot.
    """
    validate_harness()

    original_on_epoch_end = harness.PivotCallback.on_epoch_end

    verification = {
        "started_utc": utc_now(),
        "callback_calls": 0,
        "pivot_events": 0,
        "changed_layers": 0,
        "pivot_done": False,
        "epochs_seen": [],
        "passed": False,
    }

    def instrumented_on_epoch_end(
        self,
        args,
        state,
        control,
        model=None,
        optimizer=None,
        **kwargs,
    ):
        verification["callback_calls"] += 1

        epoch = None if state.epoch is None else float(state.epoch)
        verification["epochs_seen"].append(epoch)

        before = {}
        if model is not None:
            for name, module in model.named_modules():
                if isinstance(module, harness.LoRALayer):
                    before[name] = module.lora_A.detach().cpu().clone()

        was_done = bool(self.pivot_done)

        result = original_on_epoch_end(
            self,
            args,
            state,
            control,
            model=model,
            optimizer=optimizer,
            **kwargs,
        )

        now_done = bool(self.pivot_done)

        if now_done and not was_done:
            verification["pivot_events"] += 1
            verification["pivot_done"] = True

            changed = 0

            if model is not None:
                for name, module in model.named_modules():
                    if name not in before:
                        continue

                    after = module.lora_A.detach().cpu()
                    difference = torch.linalg.vector_norm(
                        after - before[name]
                    ).item()

                    if difference > 1e-8:
                        changed += 1

            verification["changed_layers"] += changed

        return result

    harness.PivotCallback.on_epoch_end = instrumented_on_epoch_end

    spec = RunSpec(
        protocol="verify-pivot",
        task="sst2",
        shot=SHOT,
        seed=42,
        method="PivotLoRA",
        method_key=METHOD_KEYS["PivotLoRA"],
        learning_rate=ANCHOR_LEARNING_RATE,
        # Pivot target is epoch 3, so four epochs safely include the event.
        epochs=max(4, int(getattr(harness, "PIVOT_EPOCH", 3)) + 1),
    )

    try:
        score, wall_seconds, peak_memory_mb = call_harness(spec)

        verification["score"] = score
        verification["wall_seconds"] = wall_seconds
        verification["peak_memory_mb"] = peak_memory_mb

    finally:
        harness.PivotCallback.on_epoch_end = original_on_epoch_end

    verification["finished_utc"] = utc_now()
    verification["passed"] = bool(
        verification["pivot_events"] >= 1
        and verification["pivot_done"]
        and verification["changed_layers"] >= 1
    )

    with PIVOT_REPORT_JSON.open("w", encoding="utf-8") as file:
        json.dump(verification, file, indent=2)

    print()
    print("=" * 78)
    print("PIVOT VERIFICATION")
    print("=" * 78)
    print(json.dumps(verification, indent=2))
    print(f"Report: {PIVOT_REPORT_JSON}")

    if not verification["passed"]:
        raise RuntimeError(
            "Pivot verification failed. The callback did not demonstrate an "
            "actual change to at least one LoRA A matrix."
        )

    print("PIVOT VERIFICATION PASSED")
    print("=" * 78)


# =============================================================================
# AGGREGATION
# =============================================================================

def load_successful_full_rows() -> List[Dict[str, object]]:
    parsed = []

    for row in read_csv_rows(RAW_RESULTS_CSV):
        if row.get("status") != "ok":
            continue

        if row.get("protocol") != "full":
            continue

        try:
            parsed.append(
                {
                    "run_id": row["run_id"],
                    "task": row["task"],
                    "metric": row["metric"],
                    "shot": int(row["shot"]),
                    "seed": int(row["seed"]),
                    "method": row["method"],
                    "method_key": row["method_key"],
                    "lora_plus_ratio": parse_optional_float(
                        row.get("lora_plus_ratio")
                    ),
                    "learning_rate": float(row["learning_rate"]),
                    "epochs": int(row["epochs"]),
                    "score": float(row["score"]),
                    "wall_seconds": (
                        float(row["wall_seconds"])
                        if row.get("wall_seconds")
                        else float("nan")
                    ),
                    "peak_memory_mb": (
                        float(row["peak_memory_mb"])
                        if row.get("peak_memory_mb")
                        else float("nan")
                    ),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            print(f"[WARN] Ignoring malformed row {row}: {exc}")

    # If --force created duplicates, retain the last successful row per run_id.
    deduplicated = {}

    for row in parsed:
        deduplicated[row["run_id"]] = row

    return list(deduplicated.values())


def config_key(row: Dict[str, object]) -> Tuple:
    return (
        row["task"],
        row["method"],
        row["learning_rate"],
        row["lora_plus_ratio"],
    )


def aggregate_results(require_complete: bool = True) -> None:
    rows = load_successful_full_rows()

    if not rows:
        raise RuntimeError(
            f"No successful full-protocol rows found in {RAW_RESULTS_CSV}"
        )

    expected_seeds = set(SEEDS)
    grouped: Dict[Tuple, List[Dict[str, object]]] = {}

    for row in rows:
        grouped.setdefault(config_key(row), []).append(row)

    summaries = []

    for key, group in grouped.items():
        task, method, learning_rate, ratio = key

        # Keep at most one row per seed.
        by_seed = {int(row["seed"]): row for row in group}
        observed_seeds = set(by_seed)

        complete = expected_seeds.issubset(observed_seeds)

        if require_complete and not complete:
            print(
                f"[INCOMPLETE] {task} {method} "
                f"lr={learning_rate:.3e} ratio={ratio}: "
                f"seeds={sorted(observed_seeds)}"
            )
            continue

        selected_rows = [
            by_seed[seed]
            for seed in sorted(observed_seeds)
            if seed in expected_seeds
        ]

        scores = [float(row["score"]) for row in selected_rows]
        walls = [
            float(row["wall_seconds"])
            for row in selected_rows
            if math.isfinite(float(row["wall_seconds"]))
        ]
        peaks = [
            float(row["peak_memory_mb"])
            for row in selected_rows
            if math.isfinite(float(row["peak_memory_mb"]))
        ]

        mean_score = statistics.fmean(scores)
        std_score = statistics.stdev(scores) if len(scores) >= 2 else 0.0

        summaries.append(
            {
                "task": task,
                "task_display": TASK_DISPLAY_NAMES[task],
                "metric": TASK_METRICS[task],
                "method": method,
                "learning_rate": learning_rate,
                "lora_plus_ratio": ratio,
                "n_seeds": len(scores),
                "seeds": ",".join(
                    str(row["seed"]) for row in selected_rows
                ),
                "mean_score": mean_score,
                "std_score": std_score,
                "mean_wall_seconds": (
                    statistics.fmean(walls) if walls else float("nan")
                ),
                "mean_peak_memory_mb": (
                    statistics.fmean(peaks) if peaks else float("nan")
                ),
                "complete": complete,
            }
        )

    if not summaries:
        raise RuntimeError(
            "No complete configurations were available for aggregation."
        )

    all_config_fields = [
        "task",
        "task_display",
        "metric",
        "method",
        "learning_rate",
        "lora_plus_ratio",
        "n_seeds",
        "seeds",
        "mean_score",
        "std_score",
        "mean_wall_seconds",
        "mean_peak_memory_mb",
        "complete",
    ]

    with ALL_CONFIG_SUMMARY_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=all_config_fields)
        writer.writeheader()

        for row in sorted(
            summaries,
            key=lambda x: (
                TASKS.index(x["task"]),
                METHOD_ORDER.index(x["method"]),
                x["learning_rate"],
                -1 if x["lora_plus_ratio"] is None
                else x["lora_plus_ratio"],
            ),
        ):
            writer.writerow(row)

    # Method-specific tuning: select the configuration with the largest
    # three-seed mean separately for each task and method.
    tuned_rows = []

    for task in TASKS:
        for method in METHOD_ORDER:
            candidates = [
                row for row in summaries
                if row["task"] == task and row["method"] == method
            ]

            if not candidates:
                print(f"[MISSING] No complete result: {task} {method}")
                continue

            # Deterministic tie-breaking:
            # 1. higher score;
            # 2. lower base LR;
            # 3. lower ratio.
            best = sorted(
                candidates,
                key=lambda row: (
                    -row["mean_score"],
                    row["learning_rate"],
                    -1 if row["lora_plus_ratio"] is None
                    else row["lora_plus_ratio"],
                ),
            )[0]

            tuned_rows.append(dict(best))

    tuned_fields = all_config_fields

    with TUNED_BY_TASK_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=tuned_fields)
        writer.writeheader()

        for row in tuned_rows:
            writer.writerow(row)

    # Do not average accuracy and MCC into a single performance claim.
    # This file reports per-task values in separate columns.
    summary_fields = [
        "method",
        "agnews_mean",
        "agnews_std",
        "agnews_lr",
        "agnews_ratio",
        "sst2_mean",
        "sst2_std",
        "sst2_lr",
        "sst2_ratio",
        "cola_mean",
        "cola_std",
        "cola_lr",
        "cola_ratio",
    ]

    by_task_method = {
        (row["task"], row["method"]): row
        for row in tuned_rows
    }

    with TUNED_SUMMARY_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()

        for method in METHOD_ORDER:
            output = {"method": method}

            for task in TASKS:
                row = by_task_method.get((task, method))
                prefix = task

                if row is None:
                    output[f"{prefix}_mean"] = ""
                    output[f"{prefix}_std"] = ""
                    output[f"{prefix}_lr"] = ""
                    output[f"{prefix}_ratio"] = ""
                else:
                    output[f"{prefix}_mean"] = row["mean_score"]
                    output[f"{prefix}_std"] = row["std_score"]
                    output[f"{prefix}_lr"] = row["learning_rate"]
                    output[f"{prefix}_ratio"] = (
                        ""
                        if row["lora_plus_ratio"] is None
                        else row["lora_plus_ratio"]
                    )

            writer.writerow(output)

    write_latex_table(by_task_method)
    print_tuned_table(by_task_method)

    print()
    print("Generated:")
    print(f"  {ALL_CONFIG_SUMMARY_CSV}")
    print(f"  {TUNED_BY_TASK_CSV}")
    print(f"  {TUNED_SUMMARY_CSV}")
    print(f"  {LATEX_TABLE_TXT}")


def print_tuned_table(by_task_method: Dict[Tuple, Dict]) -> None:
    print()
    print("=" * 104)
    print("TUNED PROTOCOL — COMPLETE THREE-SEED CELLS ONLY")
    print("=" * 104)

    header = (
        f"{'Method':<14}"
        f"{'AG News accuracy':>28}"
        f"{'SST-2 accuracy':>28}"
        f"{'CoLA MCC':>28}"
    )
    print(header)
    print("-" * 104)

    for method in METHOD_ORDER:
        cells = []

        for task in TASKS:
            row = by_task_method.get((task, method))

            if row is None:
                cells.append("MISSING")
                continue

            ratio_text = (
                ""
                if row["lora_plus_ratio"] is None
                else f",r={row['lora_plus_ratio']:g}"
            )

            cells.append(
                f"{row['mean_score']:.4f}±{row['std_score']:.4f} "
                f"(lr={row['learning_rate']:.0e}{ratio_text})"
            )

        print(
            f"{method:<14}"
            f"{cells[0]:>28}"
            f"{cells[1]:>28}"
            f"{cells[2]:>28}"
        )

    print("=" * 104)


def latex_escape(text: str) -> str:
    return (
        text.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def write_latex_table(by_task_method: Dict[Tuple, Dict]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & AG News Acc. & SST-2 Acc. & CoLA MCC \\",
        r"\midrule",
    ]

    for method in METHOD_ORDER:
        cells = []

        for task in TASKS:
            row = by_task_method.get((task, method))

            if row is None:
                cells.append("--")
            else:
                cells.append(
                    f"{row['mean_score']:.4f} $\\pm$ "
                    f"{row['std_score']:.4f}"
                )

        lines.append(
            f"{latex_escape(method)} & "
            f"{cells[0]} & {cells[1]} & {cells[2]} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Method-specific learning-rate tuning at 64 shots. "
            r"Each cell reports mean $\pm$ standard deviation over seeds "
            r"42, 43, and 44. Accuracy and Matthews correlation are reported "
            r"separately and are not averaged across tasks.}",
            r"\label{tab:jcmz-lr-sweep}",
            r"\end{table}",
        ]
    )

    LATEX_TABLE_TXT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="JCmz LR sweep using train_classification.py."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--probe",
        action="store_true",
        help="Validate the harness import, task keys, and method keys.",
    )
    mode.add_argument(
        "--verify-pivot",
        action="store_true",
        help="Run and instrument one PivotLoRA pivot event.",
    )
    mode.add_argument(
        "--anchor",
        action="store_true",
        help="Run LoRA and PivotLoRA at the central LR for all tasks/seeds.",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Run the full 243-run sweep.",
    )
    mode.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate completed three-seed cells.",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help=f"Task subset. Available: {', '.join(TASKS)}",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seed subset. Default: 42 43 44",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=f"Method subset. Available: {', '.join(METHOD_ORDER)}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun cells even if a successful cached row exists.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if a run fails.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Aggregate configurations that do not contain all three seeds.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.probe:
        run_probe()
        return

    validate_harness()

    if args.verify_pivot:
        run_verify_pivot(force=args.force)
        return

    selected_tasks = validate_selection(
        args.tasks,
        TASKS,
        "task",
    )

    selected_methods = validate_selection(
        args.methods,
        METHOD_ORDER,
        "method",
    )

    selected_seeds = list(args.seeds) if args.seeds else list(SEEDS)

    invalid_seeds = [
        seed for seed in selected_seeds
        if seed not in SEEDS
    ]

    if invalid_seeds:
        raise ValueError(
            f"This protocol uses seeds {SEEDS}; invalid seed selection: "
            f"{invalid_seeds}"
        )

    if args.anchor:
        specs = build_anchor_specs(
            tasks=selected_tasks,
            seeds=selected_seeds,
        )

        execute_specs(
            specs,
            force=args.force,
            stop_on_error=args.stop_on_error,
        )
        return

    if args.full:
        specs = build_full_specs(
            tasks=selected_tasks,
            seeds=selected_seeds,
            methods=selected_methods,
        )

        execute_specs(
            specs,
            force=args.force,
            stop_on_error=args.stop_on_error,
        )
        return

    if args.aggregate:
        aggregate_results(
            require_complete=not args.allow_incomplete
        )
        return


if __name__ == "__main__":
    main()
