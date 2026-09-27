import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import csv
import os

def load_few_shot_data(task_name, n_shot, seed=42):
    """Load n_shot examples per class with stratified sampling."""
    rng = np.random.RandomState(seed)
    
    if task_name == "sst2":
        ds = load_dataset("nyu-mll/glue", "sst2", split="train")
        val_ds = load_dataset("nyu-mll/glue", "sst2", split="validation")
        text_key = "sentence"
    elif task_name == "fancyzhx/ag_news":
        ds = load_dataset("fancyzhx/ag_news", split="train")
        val_ds = load_dataset("fancyzhx/ag_news", split="test")
        text_key = "text"
    elif task_name == "cola":
        ds = load_dataset("nyu-mll/glue", "cola", split="train")
        val_ds = load_dataset("nyu-mll/glue", "cola", split="validation")
        text_key = "sentence"
    elif task_name == "mnli":
        ds = load_dataset("nyu-mll/glue", "mnli", split="train")
        val_ds = load_dataset("nyu-mll/glue", "mnli", split="validation_matched")
        text_key = "premise"  # will concat with hypothesis
    elif task_name == "qnli":
        ds = load_dataset("nyu-mll/glue", "qnli", split="train")
        val_ds = load_dataset("nyu-mll/glue", "qnli", split="validation")
        text_key = "question"  # will concat with sentence
    elif task_name == "rte":
        ds = load_dataset("nyu-mll/glue", "rte", split="train")
        val_ds = load_dataset("nyu-mll/glue", "rte", split="validation")
        text_key = "sentence1"  # will concat with sentence2
    elif task_name == "boolq":
        ds = load_dataset("google/boolq", split="train")
        val_ds = load_dataset("google/boolq", split="validation")
        text_key = "question"  # will concat with passage
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    labels = np.array(ds["label"])
    unique_labels = np.unique(labels)
    n_per_class = n_shot // len(unique_labels)
    
    selected_idx = []
    for label in unique_labels:
        label_idx = np.where(labels == label)[0]
        chosen = rng.choice(label_idx, size=min(n_per_class, len(label_idx)), replace=False)
        selected_idx.extend(chosen)
    
    rng.shuffle(selected_idx)
    train_subset = ds.select(selected_idx)
    
    return train_subset, val_ds, text_key


def get_texts(dataset, task_name, text_key):
    """Extract text (handle pair tasks)."""
    if task_name == "mnli":
        return [f"{p} [SEP] {h}" for p, h in zip(dataset["premise"], dataset["hypothesis"])]
    elif task_name == "qnli":
        return [f"{q} [SEP] {s}" for q, s in zip(dataset["question"], dataset["sentence"])]
    elif task_name == "rte":
        return [f"{s1} [SEP] {s2}" for s1, s2 in zip(dataset["sentence1"], dataset["sentence2"])]
    elif task_name == "boolq":
        return [f"{q} [SEP] {p[:256]}" for q, p in zip(dataset["question"], dataset["passage"])]
    else:
        return list(dataset[text_key])


def train_full_finetune(task_name, n_shot, seed=42, device="cpu"):
    """
    Full fine-tuning of BERT-base on few-shot data.
    Same hyperparameters as LoRA experiments where possible:
    - 15 epochs, batch size 16, lr 2e-5 (standard for full FT)
    - AdamW, weight decay 0.01
    """
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Load data
    train_ds, val_ds, text_key = load_few_shot_data(task_name, n_shot, seed)
    
    n_labels = len(set(train_ds["label"]))
    
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    train_texts = get_texts(train_ds, task_name, text_key)
    val_texts = get_texts(val_ds, task_name, text_key)
    
    train_enc = tokenizer(train_texts, padding=True, truncation=True, 
                          max_length=128, return_tensors="pt")
    val_enc = tokenizer(val_texts, padding=True, truncation=True, 
                        max_length=128, return_tensors="pt")
    
    train_labels = torch.tensor(train_ds["label"], dtype=torch.long)
    val_labels = torch.tensor(val_ds["label"], dtype=torch.long)
    
    train_dataset = TensorDataset(
        train_enc["input_ids"], train_enc["attention_mask"], train_labels
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Model — full fine-tuning (all params trainable)
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=n_labels
    ).to(device)
    
    # Count params
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Optimizer — standard full FT learning rate (2e-5, lower than LoRA's 2e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    n_epochs = 15
    total_steps = n_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    # Train
    model.train()
    for epoch in range(n_epochs):
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    batch_size = 128
    
    with torch.no_grad():
        for i in range(0, len(val_enc["input_ids"]), batch_size):
            input_ids = val_enc["input_ids"][i:i+batch_size].to(device)
            attention_mask = val_enc["attention_mask"][i:i+batch_size].to(device)
            labels = val_labels[i:i+batch_size].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    
    # Also check train accuracy (to confirm overfitting)
    model.eval()
    train_correct = 0
    with torch.no_grad():
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
    train_acc = train_correct / len(train_dataset)
    
    return {
        "task": task_name,
        "n_shot": n_shot,
        "seed": seed,
        "train_acc": train_acc,
        "val_acc": accuracy,
        "trainable_params": total_params,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Expected runtime: {'~5 min' if device == 'cuda' else '~20 min'}")
    
    tasks = ["sst2", "cola", "fancyzhx/ag_news", "mnli", "qnli", "rte"]
    shots = [64, 256]
    seeds = [42, 43, 44]
    
    results = []
    
    for task in tasks:
        for n_shot in shots:
            accs = []
            for seed in seeds:
                print(f"\n{'='*50}")
                print(f"Task: {task}, n={n_shot}, seed={seed}")
                print(f"{'='*50}")
                
                result = train_full_finetune(task, n_shot, seed, device)
                results.append(result)
                accs.append(result["val_acc"])
                
                print(f"  Train acc: {result['train_acc']:.3f}")
                print(f"  Val acc:   {result['val_acc']:.3f}")
                print(f"  Params:    {result['trainable_params']:,}")
            
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            print(f"\n  >>> {task} {n_shot}-shot: {mean_acc:.3f} ± {std_acc:.3f}")
    
    # Save CSV
    os.makedirs("results", exist_ok=True)
    with open("results/full_finetune_baseline.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Full Fine-Tuning vs LoRA vs PivotLoRA")
    print("=" * 70)
    print(f"{'Task':>10s} {'Shot':>5s} {'FullFT':>10s} {'LoRA':>10s} {'PivotLoRA':>10s}")
    print("-" * 50)
    
    # Reference LoRA/PivotLoRA values
    lora_ref = {
        ("sst2", 64): .652, ("sst2", 256): .844,
        ("cola", 64): .163, ("cola", 256): .407,
        ("fancyzhx/ag_news", 64): .625, ("fancyzhx/ag_news", 256): .855,
        ("mnli", 64): .366, ("mnli", 256): .392,
        ("qnli", 64): .555, ("qnli", 256): .662,
        ("rte", 64): .551, ("rte", 256): .549,
    }
    pivot_ref = {
        ("sst2", 64): .795, ("sst2", 256): .850,
        ("cola", 64): .288, ("cola", 256): .411,
        ("fancyzhx/ag_news", 64): .818, ("fancyzhx/ag_news", 256): .878,
        ("mnli", 64): .367, ("mnli", 256): .454,
        ("qnli", 64): .594, ("qnli", 256): .691,
        ("rte", 64): .569, ("rte", 256): .563,
    }
    
    for task in tasks:
        for n_shot in shots:
            task_results = [r for r in results 
                          if r["task"] == task and r["n_shot"] == n_shot]
            mean_acc = np.mean([r["val_acc"] for r in task_results])
            std_acc = np.std([r["val_acc"] for r in task_results])
            
            lora_val = lora_ref.get((task, n_shot), "?")
            pivot_val = pivot_ref.get((task, n_shot), "?")
            
            print(f"{task:>10s} {n_shot:>5d}  "
                  f"{mean_acc:.3f}±{std_acc:.3f}  "
                  f"{lora_val:>8s}       {pivot_val:>8s}")
    
    print("\n  Expected: Full FT achieves ~100% train acc but poor val acc")
    print("  at 64-shot due to overfitting 110M params on 64 examples.")
    print("  LoRA's rank constraint provides essential regularization.")