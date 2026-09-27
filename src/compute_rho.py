import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LEN = 128
SHOTS = [16, 32, 64, 128, 256]
SEEDS = [42, 43, 44]
RANK = 8

TASKS = {
    "SST-2":   {"dataset": ("nyu-mll/glue", "sst2"), "text_fields": ["sentence"], "num_labels": 2},
    "AG News": {"dataset": ("fancyzhx/ag_news",),     "text_fields": ["text"],     "num_labels": 4},
    "MNLI":    {"dataset": ("nyu-mll/glue", "mnli"), "text_fields": ["premise", "hypothesis"], "num_labels": 3},
    "RTE":     {"dataset": ("nyu-mll/glue", "rte"),  "text_fields": ["sentence1", "sentence2"], "num_labels": 2},
    "QNLI":    {"dataset": ("nyu-mll/glue", "qnli"), "text_fields": ["question", "sentence"], "num_labels": 2},
    "BoolQ":   {"dataset": ("google/boolq",), "text_fields": ["question", "passage"], "num_labels": 2, "label_key": "answer"},
}


def get_cls_embeddings(model, tokenizer, texts, batch_size=32):
    model.eval()
    all_cls = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, truncation=True, max_length=MAX_SEQ_LEN,
                          padding="max_length", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_cls.append(cls)
    return np.concatenate(all_cls, axis=0)


def pca_class_alignment(X, y, rank=8):
    classes = np.unique(y)
    overall_mean = X.mean(axis=0)
    
    pca = PCA(n_components=min(rank, X.shape[0]-1, X.shape[1]))
    pca.fit(X)
    V_r = pca.components_.T

    sb_projected = 0.0
    sb_total = 0.0
    for c in classes:
        mask = (y == c)
        n_c = mask.sum()
        class_mean = X[mask].mean(axis=0)
        diff = class_mean - overall_mean
        sb_total += n_c * np.dot(diff, diff)
        projected = V_r.T @ diff
        sb_projected += n_c * np.dot(projected, projected)
    
    if sb_total < 1e-10:
        return 0.0
    return sb_projected / sb_total


def fisher_discriminant_ratio(X, y):
    classes = np.unique(y)
    overall_mean = X.mean(axis=0)
    
    sb_trace = 0.0
    for c in classes:
        mask = (y == c)
        n_c = mask.sum()
        class_mean = X[mask].mean(axis=0)
        diff = class_mean - overall_mean
        sb_trace += n_c * np.dot(diff, diff)
    
    sw_trace = 0.0
    for c in classes:
        mask = (y == c)
        class_mean = X[mask].mean(axis=0)
        diffs = X[mask] - class_mean
        sw_trace += np.sum(diffs ** 2)
    
    if sw_trace < 1e-10:
        return float('inf')
    return sb_trace / sw_trace


def sample_few_shot(dataset, n_shot, num_labels, seed, label_key="label"):
    rng = np.random.default_rng(seed)
    if label_key == "answer":
        labels = np.array(dataset["answer"], dtype=int)
    else:
        labels = np.array(dataset["label"])
    
    indices = []
    for c in range(num_labels):
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        n_per = n_shot // num_labels
        indices.extend(c_idx[:n_per].tolist())
    rng.shuffle(indices)
    return dataset.select(indices), labels[indices]


def get_texts(ds, info):
    fields = info["text_fields"]
    if len(fields) == 1:
        return list(ds[fields[0]])
    elif fields[0] == "question" and fields[1] == "passage":
        return [f"{q} [SEP] {p[:200]}" for q, p in zip(ds[fields[0]], ds[fields[1]])]
    else:
        return [f"{a} {b}" for a, b in zip(ds[fields[0]], ds[fields[1]])]


def main():
    print(f"Device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    
    all_results = {}
    
    for task_name, info in TASKS.items():
        print(f"\nLoading {task_name}...")
        ds = load_dataset(*info["dataset"])
        train = ds["train"]
        label_key = info.get("label_key", "label")
        
        task_alpha = {}
        task_fisher = {}
        
        for n_shot in SHOTS:
            alphas = []
            fishers = []
            for seed in SEEDS:
                subset, labels = sample_few_shot(train, n_shot, info["num_labels"], seed, label_key)
                texts = get_texts(subset, info)
                cls_emb = get_cls_embeddings(model, tokenizer, texts)
                
                alpha = pca_class_alignment(cls_emb, labels, rank=RANK)
                fisher = fisher_discriminant_ratio(cls_emb, labels)
                alphas.append(alpha)
                fishers.append(fisher)
            
            task_alpha[n_shot] = (np.mean(alphas), np.std(alphas))
            task_fisher[n_shot] = (np.mean(fishers), np.std(fishers))
        
        all_results[task_name] = {"alpha": task_alpha, "fisher": task_fisher}
    
    # Print results
    print("\n" + "=" * 80)
    print("FISHER DISCRIMINANT RATIO (tr(S_B)/tr(S_W)), mean over 3 seeds")
    print("=" * 80)
    print(f"{'Task':>8s}", end="")
    for s in SHOTS:
        print(f"{'n='+str(s):>10s}", end="")
    print()
    print("-" * 60)
    for task_name in TASKS:
        print(f"{task_name:>8s}", end="")
        for s in SHOTS:
            m, _ = all_results[task_name]["fisher"][s]
            print(f"{m:>10.2f}", end="")
        print()
    
    print("\n" + "=" * 80)
    print("PCA-CLASS ALIGNMENT α (Definition 1), mean over 3 seeds")
    print("=" * 80)
    print(f"{'Task':>8s}", end="")
    for s in SHOTS:
        print(f"{'n='+str(s):>10s}", end="")
    print()
    print("-" * 60)
    for task_name in TASKS:
        print(f"{task_name:>8s}", end="")
        for s in SHOTS:
            m, _ = all_results[task_name]["alpha"][s]
            print(f"{m:>10.3f}", end="")
        print()
    
    # Compare with paper values
    paper_alpha = {
        "SST-2":   [0.803, 0.653, 0.673, 0.495, 0.582],
        "AG News": [0.911, 0.821, 0.843, 0.876, 0.901],
        "MNLI":    [0.760, 0.536, 0.423, 0.310, 0.351],
        "RTE":     [0.705, 0.478, 0.299, 0.345, 0.280],
        "QNLI":    [0.756, 0.560, 0.453, 0.424, 0.437],
    }
    paper_fisher = {
        "SST-2":   [0.07, 0.04, 0.02, 0.01, 0.01],
        "AG News": [0.37, 0.18, 0.14, 0.11, 0.10],
        "MNLI":    [0.16, 0.07, 0.03, 0.02, 0.01],
        "RTE":     [0.07, 0.03, 0.01, 0.01, 0.00],
        "QNLI":    [0.07, 0.04, 0.02, 0.01, 0.01],
    }
    
    print("\n" + "=" * 80)
    print("VERIFICATION vs PAPER VALUES")
    print("=" * 80)
    mismatches = 0
    for task_name in paper_alpha:
        for i, s in enumerate(SHOTS):
            got_a = all_results[task_name]["alpha"][s][0]
            exp_a = paper_alpha[task_name][i]
            got_f = all_results[task_name]["fisher"][s][0]
            exp_f = paper_fisher[task_name][i]
            
            a_ok = abs(got_a - exp_a) < 0.005
            f_ok = abs(got_f - exp_f) < 0.005
            
            if not a_ok:
                print(f"  ❌ {task_name} n={s}: α got {got_a:.3f}, paper says {exp_a:.3f}")
                mismatches += 1
            if not f_ok:
                print(f"  ❌ {task_name} n={s}: Fisher got {got_f:.2f}, paper says {exp_f:.2f}")
                mismatches += 1
    
    if mismatches == 0:
        print("  ✅ ALL 50 values match the paper (5 tasks × 5 shots × 2 metrics)")
    else:
        print(f"  ⚠️  {mismatches} mismatches found")
    
    # Monotonicity check at 64-shot
    print("\n" + "=" * 80)
    print("MONOTONICITY CHECK: α at 64-shot vs PivotLoRA gain")
    print("=" * 80)
    gains = {
        "AG News": 0.193, "SST-2": 0.143, "QNLI": 0.039,
        "MNLI": 0.001, "RTE": 0.018, "BoolQ": 0.016,
    }
    pairs = []
    for task_name in gains:
        a = all_results[task_name]["alpha"][64][0]
        g = gains[task_name]
        pairs.append((task_name, a, g))
        print(f"  {task_name:>8s}: α = {a:.3f}, gain = +{g*100:.1f}pp")
    
    # Sort by α descending
    pairs.sort(key=lambda x: -x[1])
    print("\nSorted by α (should roughly match gain order):")
    for task, a, g in pairs:
        print(f"  {task:>8s}: α = {a:.3f}, gain = +{g*100:.1f}pp")
    
    # Check if gain order matches α order (allowing small inversions)
    alpha_order = [p[0] for p in pairs]
    gain_order = [p[0] for p in sorted(pairs, key=lambda x: -x[2])]
    print(f"\n  α order:    {alpha_order}")
    print(f"  Gain order: {gain_order}")
    
    # Pearson correlation
    alphas_all = [p[1] for p in pairs]
    gains_all = [p[2] for p in pairs]
    r = np.corrcoef(alphas_all, gains_all)[0, 1]
    print(f"  Pearson r (6 tasks at 64-shot): {r:.3f}")


if __name__ == "__main__":
    main()