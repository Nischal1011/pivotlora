import os, sys, shutil, csv, time
import numpy as np
import torch
from collections import defaultdict

from train_classification import (
    load_task_with_splits, get_block_subspaces, inject_model,
    LoRALayer, RANK, ALPHA, LEARNING_RATE, EPOCHS, BATCH_SIZE, DEVICE,
)
from transformers import (
    AutoModelForSequenceClassification, TrainingArguments, Trainer, set_seed,
)
from sklearn.metrics import accuracy_score, matthews_corrcoef

SEEDS = [42, 43, 44]
TASKS = {
    "agnews": {"metric": "accuracy", "num_labels": 4},
    "sst2":   {"metric": "accuracy", "num_labels": 2},
    "cola":   {"metric": "matthews", "num_labels": 2},
}
SHOT = 64
RESULTS_CSV = "results_direction_vs_lr.csv"

ALL_METHODS = [
    "lora",
    "eigenlora",
    "random_eigen_lr",
    "pca_uniform_lr",
    "pca_shuffled_lr",
    "pca_frozen_a",
    "random_frozen_a",
]


def run_one(task, seed, method):
    print(f"\n{'='*70}")
    print(f"Task={task} Shot={SHOT} Seed={seed} Method={method}")
    print(f"{'='*70}")

    task_info = TASKS[task]
    train_ds, val_ds, info = load_task_with_splits(task, SHOT, seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=info["num_labels"]
    ).to(DEVICE)

    block_subspaces, eigenvalues_dict = get_block_subspaces(
        model, train_ds, RANK, info["num_labels"], seed,
        subspace_type="pca_eigen",
    )

    if method == "lora":
        inject_model(model, RANK, block_subspaces=None,
                     use_lda=False, apply_rotation=False,
                     eigenvalues_dict=None)

    elif method == "eigenlora":
        inject_model(model, RANK, block_subspaces=block_subspaces,
                     use_lda=True, apply_rotation=True,
                     eigenvalues_dict=eigenvalues_dict)

    elif method == "random_eigen_lr":
        inject_model(model, RANK, block_subspaces=None,
                     use_lda=False, apply_rotation=False,
                     eigenvalues_dict=eigenvalues_dict)

    elif method == "pca_uniform_lr":
        inject_model(model, RANK, block_subspaces=block_subspaces,
                     use_lda=True, apply_rotation=True,
                     eigenvalues_dict=None)

    elif method == "pca_shuffled_lr":
        shuffled_eig = {}
        rng_shuf = np.random.default_rng(seed + 9999)
        for idx in eigenvalues_dict:
            eigs = eigenvalues_dict[idx].copy()
            rng_shuf.shuffle(eigs)
            shuffled_eig[idx] = eigs
        inject_model(model, RANK, block_subspaces=block_subspaces,
                     use_lda=True, apply_rotation=True,
                     eigenvalues_dict=shuffled_eig)

    elif method == "pca_frozen_a":
        inject_model(model, RANK, block_subspaces=block_subspaces,
                     use_lda=True, apply_rotation=True,
                     eigenvalues_dict=None)
        for m in model.modules():
            if isinstance(m, LoRALayer):
                m.lora_A.requires_grad = False

    elif method == "random_frozen_a":
        inject_model(model, RANK, block_subspaces=None,
                     use_lda=False, apply_rotation=False,
                     eigenvalues_dict=None)
        for m in model.modules():
            if isinstance(m, LoRALayer):
                m.lora_A.requires_grad = False

    # ── Optimizer ─────────────────────────────────────────────
    if method in ("eigenlora", "random_eigen_lr", "pca_shuffled_lr"):
        A_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and "lora_A" in n]
        cls_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and "classifier" in n]
        param_groups = []
        if A_params:
            param_groups.append(
                {"params": A_params, "lr": LEARNING_RATE, "weight_decay": 0.01})
        for _, m in model.named_modules():
            if not isinstance(m, LoRALayer):
                continue
            if m._split_b:
                for ri, row in enumerate(m.lora_B_rows):
                    scale_i = m.b_lr_scales[ri].item()
                    param_groups.append({
                        "params": [row],
                        "lr": scale_i * 16 * LEARNING_RATE,
                        "weight_decay": 0.01,
                    })
            else:
                param_groups.append({
                    "params": [m.lora_B],
                    "lr": 16 * LEARNING_RATE,
                    "weight_decay": 0.01,
                })
        if cls_params:
            param_groups.append(
                {"params": cls_params, "lr": LEARNING_RATE, "weight_decay": 0.01})
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    elif method in ("pca_uniform_lr", "pca_frozen_a", "random_frozen_a"):
        A_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and "lora_A" in n]
        B_params = []
        for _, m in model.named_modules():
            if not isinstance(m, LoRALayer):
                continue
            if m._split_b:
                for row in m.lora_B_rows:
                    if row.requires_grad:
                        B_params.append(row)
            elif m.lora_B is not None and m.lora_B.requires_grad:
                B_params.append(m.lora_B)
        cls_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and "classifier" in n]
        param_groups = []
        if A_params:
            param_groups.append(
                {"params": A_params, "lr": LEARNING_RATE, "weight_decay": 0.01})
        if B_params:
            param_groups.append(
                {"params": B_params, "lr": 16 * LEARNING_RATE, "weight_decay": 0.01})
        if cls_params:
            param_groups.append(
                {"params": cls_params, "lr": LEARNING_RATE, "weight_decay": 0.01})
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    elif method == "lora":
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=LEARNING_RATE,
            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,} | A frozen: {'frozen' in method}")

    _ta_kwargs = dict(
        output_dir=f"tmp/ablation_{task}_{method}_{seed}",
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        save_strategy="no",
        logging_steps=999,
        report_to="none",
        seed=seed,
        dataloader_num_workers=4,
        dataloader_persistent_workers=True,
        fp16=True,
        tf32=False,
        disable_tqdm=True,
    )
    try:
        training_args = TrainingArguments(eval_strategy="epoch", **_ta_kwargs)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy="epoch", **_ta_kwargs)

    metric_name = task_info["metric"]
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        if metric_name == "matthews":
            return {"matthews": matthews_corrcoef(p.label_ids, preds)}
        return {"accuracy": accuracy_score(p.label_ids, preds)}

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
    )

    t0 = time.perf_counter()
    trainer.train()
    wall = time.perf_counter() - t0

    metric_key = f"eval_{metric_name}"
    best = max((log.get(metric_key, -999) for log in trainer.state.log_history), default=0)
    print(f"  [RESULT] {method}: {best:.4f} | {wall:.1f}s")
    return best


if __name__ == "__main__":
    # Warmup
    print("WARMUP RUN...")
    try:
        run_one("sst2", 42, "lora")
    except:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    completed = {}
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV) as f:
            for line in f:
                if line.startswith("task"):
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 4:
                    completed[(parts[0], parts[1], int(parts[2]))] = float(parts[3])

    need_header = not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0
    fh = open(RESULTS_CSV, "a")
    if need_header:
        fh.write("task,method,seed,score\n")

    results = {}
    for task in TASKS:
        for method in ALL_METHODS:
            scores = []
            for seed in SEEDS:
                key = (task, method, seed)
                if key in completed:
                    scores.append(completed[key])
                    print(f"  [SKIP] {task}-{method}-s{seed}: {completed[key]:.4f}")
                    continue
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                score = run_one(task, seed, method)
                scores.append(score)
                fh.write(f"{task},{method},{seed},{score:.6f}\n")
                fh.flush()
            if scores:
                m, s = np.mean(scores), np.std(scores)
                results[(task, method)] = (m, s)
                print(f">>> {task} {method}: {m:.4f}+/-{s:.4f}")
    fh.close()

    # ═══════════════════════════════════════════════════════════
    labels = {
        "lora":           "Random + Uniform 1x",
        "eigenlora":           "PCA + Eigen-LR 16x (EigenLoRA)",
        "random_eigen_lr":  "Random + Eigen-LR 16x",
        "pca_uniform_lr":   "PCA + Uniform 16x",
        "pca_shuffled_lr":  "PCA + Shuffled-Eigen 16x",
        "pca_frozen_a":     "PCA + Frozen-A",
        "random_frozen_a":  "Random + Frozen-A",
    }

    print(f"\n\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    for task in TASKS:
        metric = TASKS[task]["metric"]
        print(f"\n  {task.upper()} ({metric}):")
        print(f"  {'Method':<38} {'Mean':>8} {'Std':>8}")
        print(f"  {'-'*56}")
        for method in ALL_METHODS:
            if (task, method) in results:
                m, s = results[(task, method)]
                print(f"  {labels[method]:<38} {m:>8.3f} {s:>8.3f}")

    print(f"\n\n{'='*80}")
    print("2x2 FACTORIAL: Direction x LR")
    print(f"{'='*80}")
    for task in TASKS:
        r = results.get((task, "lora"), (0,0))[0]
        r_eig = results.get((task, "random_eigen_lr"), (0,0))[0]
        p_uni = results.get((task, "pca_uniform_lr"), (0,0))[0]
        p_eig = results.get((task, "eigenlora"), (0,0))[0]
        print(f"\n  {task.upper()}:")
        print(f"  {'':>15} {'Uniform':>12} {'Eigen-LR':>12}")
        print(f"  {'Random':<15} {r:>12.3f} {r_eig:>12.3f}")
        print(f"  {'PCA':<15} {p_uni:>12.3f} {p_eig:>12.3f}")
        if all(v > 0 for v in [r, r_eig, p_uni, p_eig]):
            dir_eff = ((p_uni - r) + (p_eig - r_eig)) / 2
            lr_eff = ((r_eig - r) + (p_eig - p_uni)) / 2
            print(f"  Direction effect: {dir_eff:+.3f}")
            print(f"  LR effect:       {lr_eff:+.3f}")
            print(f"  Ratio:           {abs(dir_eff)/max(abs(lr_eff),0.001):.1f}x")

    print(f"\n\n{'='*80}")
    print("FROZEN-A: Does direction matter without A training?")
    print(f"{'='*80}")
    for task in TASKS:
        r_fz = results.get((task, "random_frozen_a"), (0,0))[0]
        p_fz = results.get((task, "pca_frozen_a"), (0,0))[0]
        print(f"  {task.upper()}: Random-frozen={r_fz:.3f}, PCA-frozen={p_fz:.3f}, "
              f"gap={p_fz-r_fz:+.3f}")

    print(f"\n\n{'='*80}")
    print("SHUFFLED: Does eigenvalue-LR matching matter?")
    print(f"{'='*80}")
    for task in TASKS:
        p_eig = results.get((task, "eigenlora"), (0,0))[0]
        p_shuf = results.get((task, "pca_shuffled_lr"), (0,0))[0]
        p_uni = results.get((task, "pca_uniform_lr"), (0,0))[0]
        print(f"  {task.upper()}: Correct={p_eig:.3f}, Shuffled={p_shuf:.3f}, "
              f"Uniform={p_uni:.3f}")