# PivotLoRA

### When Does Subspace Direction Matter for LoRA? Regime Analysis of the Magnitude Principle in Few-Shot Adaptation

Nischal Subedi, Cencheng Shen, Peng Zhao — **NeurIPS 2026**

LoRA constrains the weight update to a low-rank subspace, so the initial row
span of the input factor is a real design choice rather than an arbitrary one.
This work shows that the value of that choice is **regime-dependent**: it is
largest when the representation dimension is high relative to the labeled set
size, and when the estimated activation subspace is aligned with supervised
task structure.

**PivotLoRA** operationalizes that view with two changes to standard LoRA:

1. **Prediction-relevant PCA initialization.** One frozen forward pass collects
   activations at the positions the supervision actually reads — `[CLS]` for
   BERT classification, the sequence representation for decoder classification,
   and the hidden states that predict supervised response tokens for
   completion-only reasoning. The top-`r` PCA directions of those activations
   initialize `A`, with `B = 0`, so `ΔW = 0` and the pretrained function is
   unchanged at initialization.

2. **One output-preserving orthogonal pivot.** After `K = 3` epochs, a reduced
   QR decomposition replaces the drifted input factor with an orthonormal basis
   for the *same* row span and re-encodes `B` so that `B_new A_new = B A`
   exactly. The represented update — and the model function — is preserved; only
   the coordinate parameterization changes, which removes the anisotropy from
   the Gram matrix `A Aᵀ` that shapes the `B`-factor gradient.

The paper also derives an **alignment coefficient ρ**, computable from a single
frozen forward pass, that tracks shared-protocol gains across classification
tasks (Pearson r = 0.93, exact permutation p = 0.002 over seven tasks).

**Headline results.** Across 30 BERT-base few-shot settings and 13 methods,
PivotLoRA has the best average rank and beats vanilla LoRA in 29 of 30
settings. Decoder-scale classification reproduces the low-shot pattern on
LLaMA-2-7B and Llama-3.1-8B. A response-position extension gives positive mean
paired gains on GSM8K (+1.49 pp) and SVAMP (+31.89 pp), while full-data
classification and mean-pooled summarization show little or no directional
advantage.

---

## Reproducibility note

> Few-shot fine-tuning results may vary across hardware and software
> environments. The repository provides the implementation, primary
> configurations, sampled-example manifests, and result summaries used in the
> paper; exact numerical reproduction is not guaranteed on different systems.

No model or dataset `revision` is pinned, so reconstruction also depends on Hub
state at download time. Package versions were recorded at run time only for the
reasoning experiments (PyTorch 2.9.0+cu128, CUDA 12.8); see
[`requirements.txt`](requirements.txt).

[`results/`](results/) holds the summaries matching the paper's tables, with a
`source` column on every row. [`outputs/`](outputs/) holds the run artifacts
behind them: per-seed CSVs, protocol manifests, pivot diagnostics, the SVAMP
predictions, and the reasoning run logs. Checkpoints are not included.

---

## Installation

Python 3.10 or 3.11.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
# Match the CUDA build to your driver; cu128 is what the reasoning runs used.
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

CPU-only is enough to run everything under `tests/` and `analysis/` — plain
`pip install torch` will do.

### Main dependencies

| Package | Pin | Role |
|---|---|---|
| `torch` | 2.9.0+cu128 | recorded at run time for the reasoning experiments |
| `transformers` | 5.14.1 | models, `Trainer`, the pivot callbacks |
| `datasets` | 5.0.1 | GLUE, AG News, GSM8K, SVAMP, XSum, CNN/DailyMail |
| `peft` | 0.20.0 | AdaLoRA and DoRA baselines only; PivotLoRA is implemented directly |
| `numpy` / `scipy` / `scikit-learn` | 2.4.6 / 1.17.1 / 1.9.0 | PCA, QR, metrics |
| `rouge-score`, `nltk`, `evaluate` | — | summarization controls only |

### Model access

BERT and GPT-2 download from the Hub without credentials. The LLaMA
experiments need gated-checkpoint access — authenticate with
`huggingface-cli login` or `HF_TOKEN` outside this repository.

---

## Quick start: verify without training

Three checks, all offline, all seconds:

```bash
# 1. The method's algebra: B_new A_new == B A and A_new A_newᵀ == I,
#    plus PCA initialization and optimizer-state handling.
pytest tests/ -v

# 2. The published arithmetic: means, standard deviations, paired gains,
#    ranks, correlations, and the exact permutation p-value, all recomputed
#    from results/*.csv.
python analysis/verify_results.py

# 3. Everything compiles and imports, with no missing local modules.
python -m compileall -q src
python src/lr_sweep.py --probe                       # sweep grid, no training
python src/activation_source_ablation.py --probe     # ablation grid, no training
```

`analysis/verify_results.py` runs 137 checks covering every headline number.

---

## Running the experiments

All commands are run from the repository root.

### BERT classification (the primary controlled benchmark)

```bash
# Full benchmark: 6 tasks x 5 sample sizes x 3 seeds x 13 methods.
python src/train_classification.py

# A single cell.
python src/train_classification.py --tasks sst2 --shots 64 --seeds 3 \
                                   --methods lora pivotlora

# Smoke test (short run, one seed).
python src/train_classification.py --smoke
```

Config: [`configs/bert_classification.yaml`](configs/bert_classification.yaml) ·
Results: [`results/bert_main.csv`](results/bert_main.csv),
[`results/bert_aggregate.csv`](results/bert_aggregate.csv)

### Reasoning adaptation (GSM8K and SVAMP)

```bash
# Full paired study: both methods, all three seeds, complete test split.
python src/train_reasoning_gsm8k.py
python src/train_reasoning_svamp.py

# First K seeds only (--seeds 1 runs seed 42; --seeds 2 runs 42 and 43).
python src/train_reasoning_gsm8k.py --seeds 1

# One method only.
python src/train_reasoning_gsm8k.py --methods pivotlora

# Smoke test: both methods, seed 42, 4 epochs, 50-example eval.
# Still reaches the epoch-3 pivot, so it exercises the full code path.
python src/train_reasoning_gsm8k.py --smoke
```

Configs: [`configs/gsm8k.yaml`](configs/gsm8k.yaml),
[`configs/svamp.yaml`](configs/svamp.yaml) ·
Results: [`results/reasoning.csv`](results/reasoning.csv),
[`results/pivot_diagnostics.csv`](results/pivot_diagnostics.csv) ·
Run outputs: [`outputs/reasoning/`](outputs/reasoning/) ·
Manifests: [`manifests/gsm8k/`](manifests/gsm8k/),
[`manifests/svamp/`](manifests/svamp/)

### Computing the alignment coefficient ρ

ρ needs no gradients and no training — one frozen forward pass, then a trace
ratio:

```
Σ_task = (1/N) Σ_c n_c (μ_c − μ̄)(μ_c − μ̄)ᵀ
ρ      = Tr(Uᵀ Σ_task U) / Tr(Σ_task)
```

where `U` holds the top-`r` PCA directions of the same prediction-position
representations.

```bash
# ρ for every task and sample size (CPU is fine).
python src/compute_rho.py

# Repeated-subset and class-imbalance uncertainty analysis around ρ.
python src/alignment_rho.py --output-dir results_alignment --num-subsets 10

# Recompute the Pearson, Spearman, and exact permutation statistics from the
# seven published values -- no model inference at all.
python analysis/verify_results.py
```

Results: [`results/alignment.csv`](results/alignment.csv)

### Decoder-scale classification

Both reported backbones run this one script with identical settings — only the
checkpoint changes, selected with `--model`:

```bash
# Llama-3.1-8B (the default), SST-2 and AG News.
python src/train_decoder_classification.py --model llama3.1-8b

# LLaMA-2-7B, same protocol.
python src/train_decoder_classification.py --model llama2-7b

# BoolQ was additionally run on Llama-3.1-8B.
python src/train_decoder_classification.py --model llama3.1-8b --tasks boolq

# A single cell.
python src/train_decoder_classification.py --model llama2-7b --tasks sst2 --shots 64
```

`--model` also accepts any Hugging Face model id. Each backbone writes its own
`results_decoder_<checkpoint>.csv`, since the resume key
`(task, shot, method, seed)` does not include the model. Override with
`--results-csv`.

Config: [`configs/decoder_classification.yaml`](configs/decoder_classification.yaml) ·
Results: [`results/decoder_classification.csv`](results/decoder_classification.csv)

### Ablations and controls

```bash
python src/activation_source_ablation.py            # target-position vs. all-token
python src/activation_source_minimal_ablation.py    # the two-condition grid
python src/pivot_epoch_ablation.py                  # pivot timing K
python src/optimizer_state_ablation.py              # optimizer-state reset policy
python src/direction_vs_lr_ablation.py              # direction vs. eigenvalue-scaled LR
python src/lr_sweep.py                              # learning-rate sensitivity
python src/full_finetune_baseline.py                # full fine-tuning reference
python src/summarization_xsum.py                    # sequence-mean control
python src/summarization_cnndm.py                   # sequence-mean control
python src/validate_spiked_covariance.py            # synthetic theory validation
```

> **Heads up:** `compute_rho.py`, `validate_spiked_covariance.py`,
> `full_finetune_baseline.py`, and `direction_vs_lr_ablation.py` take no
> command-line arguments and begin work as soon as they are invoked — passing
> `--help` starts a run rather than printing usage. Every other script uses
> `argparse`; `lr_sweep.py` and `activation_source_ablation.py` additionally
> accept `--probe` (validate the configuration, train nothing) and
> `--aggregate` (summarize existing outputs).

---

## Paper results → files

| Paper result | Config | Summary | Run output |
|---|---|---|---|
| Table 1 — BERT-base main (SST-2, AG News) | `configs/bert_classification.yaml` | `results/bert_main.csv` | [`outputs/bert/main_bert_results.csv`](outputs/bert/main_bert_results.csv) |
| Appendix — complete BERT results, 7 tasks × 13 methods × 5 N | `configs/bert_classification.yaml` | `results/bert_main.csv` | [`outputs/bert/main_bert_results.csv`](outputs/bert/main_bert_results.csv) |
| Appendix — full aggregate rankings (avg. rank 1.83, 29/30 vs. LoRA) | — | `results/bert_aggregate.csv` | — |
| Table 2 — decoder-scale at N=64 | `configs/decoder_classification.yaml` | `results/decoder_classification.csv` | [`outputs/decoder/`](outputs/decoder/) |
| Appendix — LLaMA-2-7B / Llama-3.1-8B / BoolQ tables | `configs/decoder_classification.yaml` | `results/decoder_classification.csv` | [`outputs/decoder/`](outputs/decoder/) |
| §4.4 + Appendix — GSM8K and SVAMP per-seed | `configs/gsm8k.yaml`, `configs/svamp.yaml` | `results/reasoning.csv` | [`outputs/reasoning/`](outputs/reasoning/) |
| Appendix — reasoning pivot diagnostics (1.33e-7 / 2.38e-8) | `configs/gsm8k.yaml`, `configs/svamp.yaml` | `results/pivot_diagnostics.csv` | [`outputs/reasoning/pivot_diagnostics/`](outputs/reasoning/pivot_diagnostics/) |
| Appendix — classification pivot diagnostics | `configs/bert_classification.yaml` | `results/pivot_diagnostics.csv` | — |
| Appendix — learning-rate sensitivity (PivotLoRA .667) | `configs/lr_sweep.yaml` | `results/tuned_comparison.csv` | [`outputs/bert/lr_sweep_agnews.csv`](outputs/bert/lr_sweep_agnews.csv) |
| Figure 3 + Appendix — alignment ρ vs. gain | — | `results/alignment.csv` | [`outputs/bert/alignment/`](outputs/bert/alignment/) |
| Appendix — activation-source ablation (E.12) | `configs/activation_source_ablation.yaml` | `results/activation_source.csv` | [`outputs/bert/activation_source_raw.csv`](outputs/bert/activation_source_raw.csv) |
| Appendix — direction vs. learning-rate controls | `configs/lr_sweep.yaml` | `results/direction_controls.csv` | [`outputs/bert/direction_vs_lr.csv`](outputs/bert/direction_vs_lr.csv) |
| Appendix — summarization, full-data, full fine-tuning | — | `results/regime_controls.csv` | [`outputs/summarization/`](outputs/summarization/), [`outputs/bert/full_data_audit.csv`](outputs/bert/full_data_audit.csv), [`outputs/bert/full_finetune_baseline.csv`](outputs/bert/full_finetune_baseline.csv) |
| Appendix — five-seed confirmation | — | `results/robustness_five_seed.csv` | — |
| Appendix — initialization overhead and cost | — | `results/compute.csv` | — |

`results/` holds the tidied summaries matching the paper's tables;
[`outputs/`](outputs/) holds what the runs actually produced — per-seed CSVs,
protocol manifests, per-adapter pivot diagnostics, every SVAMP prediction, and
the full reasoning run logs. See [`results/README.md`](results/README.md) and
[`outputs/README.md`](outputs/README.md) for per-file schemas.

---

## Repository layout

```
PivotLoRA/
├── README.md
├── LICENSE                               MIT
├── requirements.txt
│
├── src/
│   ├── pivotlora.py                      CORE: PCA init + output-preserving QR pivot
│   ├── activation_collection.py          CORE: target-position & response-position pooling
│   ├── evaluation.py                     CORE: numeric-answer parser and scorer
│   │
│   ├── train_classification.py           BERT-base benchmark (all 13 methods)
│   ├── train_decoder_classification.py   LLaMA-2-7B / Llama-3.1-8B classification
│   ├── train_reasoning_gsm8k.py          GSM8K, LoRA vs. response-position PivotLoRA
│   ├── train_reasoning_svamp.py          SVAMP, same frozen protocol
│   │
│   ├── compute_rho.py                    alignment coefficient ρ
│   ├── alignment_rho.py                  ρ uncertainty and class-imbalance analysis
│   ├── activation_source_ablation.py     target-position vs. all-token
│   ├── activation_source_minimal_ablation.py
│   ├── pivot_epoch_ablation.py           pivot timing K
│   ├── optimizer_state_ablation.py       optimizer-state reset policy
│   ├── direction_vs_lr_ablation.py       direction vs. eigenvalue-scaled rates
│   ├── lr_sweep.py                       learning-rate sensitivity
│   ├── full_finetune_baseline.py         full fine-tuning reference
│   ├── summarization_xsum.py             GPT-2 sequence-mean control
│   ├── summarization_cnndm.py            GPT-2 sequence-mean control
│   └── validate_spiked_covariance.py     synthetic theory validation
│
├── configs/                              6 representative configurations
├── manifests/                            exact sampled-example indices
│   ├── bert/                             90 files: 6 tasks × 5 N × 3 seeds
│   ├── gsm8k/                            3 files: seeds 42, 43, 44
│   └── svamp/                            3 files: indices *and* stable SVAMP IDs
├── results/                              result summaries + schema notes
├── outputs/                              run artifacts behind those summaries
│   ├── reasoning/                        per-seed CSVs, manifests, pivot
│   │                                     diagnostics, SVAMP predictions, logs
│   ├── bert/                             benchmark, ablations, alignment
│   ├── decoder/                          Llama-3.1-8B classification
│   └── summarization/                    GPT-2 ROUGE-L controls
├── analysis/verify_results.py            137 offline arithmetic checks
└── tests/
    ├── test_pivot.py                     pivot algebra, PCA init, optimizer state
    └── test_answer_parser.py             numeric evaluator and the causal shift
```

### Reading the method

`src/pivotlora.py` and `src/activation_collection.py` are dependency-light
extractions so the method can be read without a training loop. The harnesses
contain the same algebra inline, wrapped in the `Trainer` callbacks that
produced the reported numbers.

One convention note: the paper writes `ΔW = (α/r) · B A` with `A ∈ ℝ^{r×d}`
(directions in **rows**), while the code stores the transpose, `A ∈ ℝ^{d×r}`
(directions in **columns**). `pivotlora.py` supports both via
`rows_are_directions`, and `tests/test_pivot.py` checks they agree.

---

## Sampled-example manifests

The experiments use very small training sets, so which examples were drawn
matters as much as the seed. `manifests/` records the exact selection for every
principal run.

- **BERT** (`manifests/bert/<task>_n<N>_seed<seed>.json`) — positional indices
  into the Hugging Face train split, plus per-class counts. Sampling is
  approximately class-balanced: for each label, shuffle that label's indices
  with `numpy.random.default_rng(seed)` and take `floor(N/C)`, then shuffle the
  concatenation. This is why MNLI yields 63 examples at `N = 64` and 15 at
  `N = 16`. Every method at a given `(task, N, seed)` uses the same subset.
- **GSM8K** (`manifests/gsm8k/gsm8k_seed_<seed>.json`) — positional indices from
  `default_rng(seed).choice(7473, 128, replace=False)`. GSM8K rows carry no
  stable ID field, so indices are the only available handle.
- **SVAMP** (`manifests/svamp/svamp_seed_<seed>.json`) — both positional indices
  **and** the stable `chal-*` IDs. The ID sets were regenerated from the
  sampling code and cross-checked against the `selected_train_IDs` lines in the
  original run logs; all three seeds match exactly.

Manifests were reconstructed from the sampling code. Dataset text is not
redistributed — the indices and IDs rebuild each subset from the public
datasets.

---

## Hardware used originally

| Experiment family | Hardware | Precision |
|---|---|---|
| BERT-base classification | 1 × NVIDIA T4, 16 GB | FP16 |
| GPT-2 summarization controls | 1 × NVIDIA T4, 16 GB | FP16 |
| LLaMA-2-7B classification | 1 × NVIDIA L4, 24 GB | BF16 (~14.6 GB peak) |
| Llama-3.1-8B classification | 1 × NVIDIA L4, 24 GB | BF16 |
| Llama-3.1-8B GSM8K / SVAMP reasoning | 1 × NVIDIA GeForce RTX 5090, 31.8 GB, CUDA 12.8, PyTorch 2.9.0+cu128 | BF16 base, FP32 adapters |

Overhead is small: on the BERT measurement PivotLoRA adds 4.0 s (+5.8%) to a
~69 s run and 2 MB of peak VRAM, mostly from the per-rank optimizer groups
rather than the QR pivot. Initialization and the pivot each take under a
second.

---

## Key hyperparameters at a glance

| | BERT classification | Decoder classification | GSM8K / SVAMP reasoning |
|---|---|---|---|
| Rank `r` | 8 | 8 | 8 |
| LoRA `α` | 16 | 16 | 16 |
| Targets | query, value (24 adapters) | `q_proj`, `v_proj` (64) | `q_proj`, `v_proj` (64) |
| Base LR `η_A` | 2e-4 | 2e-4 | 2e-4 |
| Epochs | 15 | 15 | 15 |
| Batch size | 16 | 4 × 4 accum = 16 | 4 × 2 accum = 8 |
| Sequence length | 128 | 128 / 192 (BoolQ) | 512 |
| Pivot epoch `K` | 3 | 3 | 3 (global step 48) |
| Optimizer reset | `A` only, `B` retained | `A` only | **both `A` and `B`** |
| Differential / eigen-scaled LR | yes, `λ_ratio = 16` | yes | **no** |
| Post-pivot `B` warm-up | 10 steps | 10 steps | **none** |
| Basis scale | 1.0 | 1.0 | **√5 ≈ 2.2361** |
| Seeds | 42, 43, 44 | 42, 43, 44 | 42, 43, 44 |

The reasoning column differs deliberately on the bolded rows: those
experiments vary initialization, basis scale, and pivot-state handling jointly,
so they evaluate the complete response-position procedure rather than subspace
direction alone. The `√5` basis scale is not norm-matched to Kaiming-uniform
initialization.

---

## Scope of the results

- ρ is defined from discrete class means; it does not apply to open-ended
  generation and is not used to predict the GSM8K or SVAMP gains.
- Aggregate rankings hold under the shared benchmark protocol, not under
  per-baseline tuning. `results/tuned_comparison.csv` is the sensitivity
  analysis.
- The reasoning results use three paired seeds per dataset. GSM8K has one
  negative seed and the SVAMP gain ranges from +19.33 to +50.67 pp.
- The activation-source ablation and the direction-versus-rate controls ran in
  separate environments, so compare within each experiment rather than across
  them.
- `results/bert_aggregate.csv` ranks come from full-precision per-seed means;
  recomputing from the three-decimal means in `bert_main.csv` reproduces the
  ordering but shifts individual ranks slightly.

Script changes made when packaging: `bert_experiment.py` →
`train_classification.py` and `projection_source_ablation.py` →
`activation_source_ablation.py` (with dependent imports updated); dataset ids
namespaced to `nyu-mll/glue`, `fancyzhx/ag_news`, `aps/super_glue` for
`datasets` 5.x compatibility; `--model` and `--results-csv` added to
`train_decoder_classification.py`.

---

## Citation

```bibtex
@inproceedings{subedi2026pivotlora,
  title     = {When Does Subspace Direction Matter for {LoRA}? Regime Analysis
               of the Magnitude Principle in Few-Shot Adaptation},
  author    = {Subedi, Nischal and Shen, Cencheng and Zhao, Peng},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  url       = {https://github.com/Nischal1011/pivotlora}
}
```

## License

Code in this repository: [MIT](LICENSE). The paper itself is distributed under
CC BY 4.0 through NeurIPS.
