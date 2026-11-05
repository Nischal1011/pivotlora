"""GPT-2 summarization: sequence-mean vs response-position activation source.

Variants (everything else identical to src/summarization_{xsum,cnndm}.py):
  lora          vanilla LoRA
  pivot_seqmean PivotLoRA as published: PCA of the mean adapter input over all
                positions of the padded sequence (document, summary, padding)
  pivot_resp_unilr  pivot_resp with one learning rate (2e-4) for both factors, no
                eigenvalue scaling and no 16x B rate (the reasoning recipe)
  pivot_resp    PivotLoRA with response-position pooling: PCA of the mean adapter
                input over the positions t whose next token t+1 is supervised
                (labels[t+1] != -100), the rule used for the reasoning runs
"""
import os
import sys
import csv
import argparse
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))


def make_response_subspaces(mod):
    """Return a drop-in replacement for mod.get_gpt2_subspaces."""

    def get_gpt2_subspaces(model, dataset, rank, seed):
        print("[SUBSPACE-RESP] Response-position pooling...")
        model.eval().to(mod.DEVICE)
        acts = {"c_proj": defaultdict(list), "c_fc": defaultdict(list)}
        holder = {}

        def make_hook(store, idx):
            def hook(module, inputs):
                x = inputs[0].detach().float()                    # (B, T, D)
                m = holder["mask"].to(x.device).float()           # (B, T)
                pooled = (x * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
                store[idx].append(pooled.cpu())
            return hook

        handles = []
        for i, block in enumerate(model.transformer.h):
            handles.append(block.attn.c_proj.register_forward_pre_hook(make_hook(acts["c_proj"], i)))
            handles.append(block.mlp.c_fc.register_forward_pre_hook(make_hook(acts["c_fc"], i)))

        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
        n_pos = []
        with torch.no_grad():
            for batch in loader:
                labels = batch["labels"]
                mask = torch.zeros_like(labels, dtype=torch.bool)
                mask[:, :-1] = labels[:, 1:] != -100              # t predicts t+1
                holder["mask"] = mask
                n_pos.extend(mask.sum(1).tolist())
                model(**{k: v.to(mod.DEVICE) for k, v in batch.items() if k != "labels"})
        for h in handles:
            h.remove()
        print(f"[SUBSPACE-RESP] response positions per example: "
              f"mean {sum(n_pos) / len(n_pos):.1f}, min {min(n_pos)}")

        subspaces, eig_dict = defaultdict(dict), defaultdict(dict)
        for target, store in acts.items():
            for idx, lst in store.items():
                X = torch.cat(lst, dim=0).numpy()
                Q, eigs = mod.compute_pca_eigen(X, rank)
                subspaces[idx][target] = Q.to(mod.DEVICE)
                eig_dict[idx][target] = eigs
        return dict(subspaces), dict(eig_dict)

    return get_gpt2_subspaces


def run_one(dataset, shot, seed, variant):
    if dataset == "xsum":
        import summarization_xsum as mod
        runner = mod.run_gen_experiment
    else:
        import summarization_cnndm as mod
        runner = mod.run_experiment
    original = mod.get_gpt2_subspaces
    try:
        if variant == "lora":
            rouge, _ = runner(shot, seed, "lora")
        elif variant == "pivot_seqmean":
            rouge, _ = runner(shot, seed, "pivotlora")
        elif variant == "pivot_seqmean_unilr":
            orig_inject, orig_eigen_opt = mod.inject_gpt2, mod.build_eigen_optimizer

            def inject_no_eigen2(*a, eigenvalues_dict=None, **k):
                return orig_inject(*a, **k)

            mod.inject_gpt2 = inject_no_eigen2
            mod.build_eigen_optimizer = mod.build_uniform_optimizer
            try:
                rouge, _ = runner(shot, seed, "pivotlora")
            finally:
                mod.inject_gpt2, mod.build_eigen_optimizer = orig_inject, orig_eigen_opt
        elif variant == "pivot_resp":
            mod.get_gpt2_subspaces = make_response_subspaces(mod)
            rouge, _ = runner(shot, seed, "pivotlora")
        else:  # pivot_resp_unilr: response pooling + one learning rate for both factors
            mod.get_gpt2_subspaces = make_response_subspaces(mod)
            orig_inject, orig_eigen_opt = mod.inject_gpt2, mod.build_eigen_optimizer

            def inject_no_eigen(*a, eigenvalues_dict=None, **k):
                return orig_inject(*a, **k)

            mod.inject_gpt2 = inject_no_eigen
            mod.build_eigen_optimizer = mod.build_uniform_optimizer
            try:
                rouge, _ = runner(shot, seed, "pivotlora")
            finally:
                mod.inject_gpt2, mod.build_eigen_optimizer = orig_inject, orig_eigen_opt
    finally:
        mod.get_gpt2_subspaces = original
    return rouge


FIELDS = ["dataset", "shot", "variant", "seed", "rouge1", "rouge2", "rougeL"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["xsum", "cnndm"])
    p.add_argument("--shots", nargs="+", type=int, default=[64, 256])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--variants", nargs="+", default=["lora", "pivot_seqmean", "pivot_resp"])
    p.add_argument("--out", default="gpt2_response_ablation.csv")
    a = p.parse_args()

    done = set()
    if os.path.exists(a.out):
        with open(a.out, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["dataset"], int(r["shot"]), r["variant"], int(r["seed"])))
    new = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for ds in a.datasets:
            for shot in a.shots:
                for seed in a.seeds:
                    for v in a.variants:
                        if (ds, shot, v, seed) in done:
                            continue
                        r = run_one(ds, shot, seed, v)
                        w.writerow({"dataset": ds, "shot": shot, "variant": v, "seed": seed,
                                    "rouge1": f"{r['rouge1']:.6f}", "rouge2": f"{r['rouge2']:.6f}",
                                    "rougeL": f"{r['rougeL']:.6f}"})
                        fh.flush()
                        print(f"[ABL] {ds} N={shot} s{seed} {v}: RL={r['rougeL']:.4f}", flush=True)


if __name__ == "__main__":
    main()
