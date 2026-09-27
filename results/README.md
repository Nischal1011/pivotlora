# Result summaries

These files hold the numbers reported in the paper. Each row's `source` column
names where it came from:

- `paper_table_<label>` — the corresponding camera-ready table.
- `run_csv+paper_table_<label>` — present in a run artifact and matching the
  paper table.
- a file path — read directly from that run artifact.

Run `python analysis/verify_results.py` from the repository root to recompute
every derived statistic from these CSVs — 137 checks, no model or GPU needed.
Standard deviations are sample standard deviations (ddof = 1) throughout.

---

## Files

### `bert_main.csv` — primary BERT-base benchmark

`task, n, metric, method, mean, std, num_seeds`

455 rows: 7 tasks × 13 methods × 5 sample sizes. The six primary tasks (SST-2,
CoLA, MNLI, QNLI, RTE, AG News) give 30 task–N settings; BoolQ is the auxiliary
task used in the alignment analysis. CoLA is Matthews correlation, the rest
accuracy. `TT-PCA` and `TT-LDA` are the paper's own ablations.

Supports the 29-of-30 win rate over LoRA (the exception is RTE at N=128,
−0.020) and the N=64 gains used in the alignment analysis.

### `bert_aggregate.csv` — aggregate rankings

`method, average_rank, settings_better_than_pivotlora, num_settings, source`

Average rank over the 30 primary settings. PivotLoRA: 1.83, best mean in 19/30,
29/30 versus LoRA, exact two-sided sign test p = 5.8e-8.

Ranks were computed from full-precision per-seed means. `bert_main.csv` stores
three-decimal means, so recomputing from it reproduces the ordering but shifts
individual values slightly; `analysis/verify_results.py` handles this.

### `decoder_classification.csv` — LLaMA-2-7B and Llama-3.1-8B

`model, task, n, metric, method, mean, std, num_seeds, source`

SST-2 and AG News at N ∈ {64, 256} for both backbones, plus BoolQ for
Llama-3.1-8B. EVA was evaluated only at N=64.

PivotLoRA has the highest mean on all five N=64 settings. At N=256 the
differences shrink and it is not uniformly best.

### `reasoning.csv` — GSM8K and SVAMP

`dataset, model, method, seed, n_train, n_test, correct, accuracy_pct, invalid, parse_rate_pct, source`

All twelve runs (2 datasets × 2 methods × 3 seeds) on the complete test split:
1,319 for GSM8K, 300 for SVAMP.

| | LoRA mean | PivotLoRA mean | Mean paired gain | Gain SD | Positive seeds |
|---|---|---|---|---|---|
| GSM8K | 23.88% | 25.37% | **+1.49 pp** | 1.84 | 2/3 |
| SVAMP | 23.11% | 55.00% | **+31.89 pp** | 16.57 | 3/3 |

These are complete-procedure results: initialization, basis scale (√5), and
pivot-state handling vary jointly, so they do not isolate subspace direction.
GSM8K seed 42 is −0.61 pp; the SVAMP gains are +25.67, +19.33, +50.67 pp.

### `tuned_comparison.csv` — learning-rate sensitivity

`comparison, method, agnews, sst2, cola, average, source`

`best_swept` is each method's best three-seed mean over the base-LR grid
{5e-5, 2e-4, 8e-4}, with LoRA+ additionally selected over
η_B/η_A ∈ {4, 8, 16, 32}. `shared_anchor` is the average at η_A = 2e-4 with
each method's prescribed factor-wise allocation.

PivotLoRA .667 > TT-PCA .655 > LoRA+ .636 > LoRA .622 > EVA .618 > LoRAM .617 >
PiSSA .606. Advantage figures in the paper come from unrounded three-task
averages. The raw 243-run sweep is not released.

### `alignment.csv` — the ρ diagnostic (Figure 3)

`task, rho, rho_cross_split, fisher_ratio, pivotlora_gain, metric, source`

Seven tasks. Pearson r = 0.93, Spearman ρ = 0.86, exact two-sided permutation
p = 0.002 over all 7! = 5,040 assignments, leave-one-task-out Pearson in
[0.87, 0.96]. Removing AG News still gives r = 0.87.

`rho_cross_split` is the same-sample-bias check on three tasks: the PCA
subspace is estimated on one half of the N=64 set and Σ_task on the other, then
the halves are exchanged and averaged. Absolute values drop, but the
AG News > SST-2 > CoLA ordering holds.

CoLA gains are Matthews-correlation points, the rest accuracy points. The
association is descriptive.

### `activation_source.csv` — target-position vs. all-token

`task, n, source, seed, score, mean, std, num_seeds, paired_gain_vs_target, seed_level_wins, num_vectors_per_layer, top8_explained_variance_ratio, source_file`

Per-seed scores (rows with a `seed`) and the paper's summary rows (rows
without). All-token initialization has the higher mean on both tasks: +3.15 pp
on AG News (3/3 seed wins), +2.33 pp on SST-2 (2/3). Paired gains are computed
before rounding, so they differ slightly from the rounded means.

This ran as an independent matched ablation in a separate environment — the
paired within-seed differences are the meaningful quantity. The diagnostics
columns show the alignment–estimation trade-off: target-position PCA uses 64
vectors per layer with 55.9%/67.8% top-8 explained variance, all-token PCA uses
3,397.7/830.7 vectors with 15.1%/21.2%.

### `direction_controls.csv` — direction vs. learning-rate allocation

`experiment_group, variant, task, n, mean, std, num_seeds, source`

A random orthonormal basis with the same eigenvalue-scaled per-rank rates does
not become competitive (AG News .444, SST-2 .571, CoLA .028 versus TT-PCA's
.751, .769, .268), while TT-PCA holds up when the rates are made uniform or
permuted. Within this sweep the directions explain more than the
eigenvalue-to-coordinate assignment.

Rerun in a separate environment, so the control's own LoRA and TT-PCA rows —
not `bert_main.csv` — are the reference for the three variants.

### `pivot_diagnostics.csv` — numerical verification of the pivot

`experiment_family, task_or_dataset, seed, adapters_pivoted, pivot_epoch, pivot_step, mean_preservation_error, max_preservation_error, mean_span_residual, orthogonality_error_before, orthogonality_error_after, mean_alignment_change, optimizer_state_reset, source`

Reasoning rows are per-seed values from the run artifacts; all 64 adapters
transform at epoch 3 / global step 48 in every run. Maximum relative
preservation error 1.33e-7 (range 1.25e-7–1.33e-7); maximum post-pivot
orthogonality error 2.38e-8 (range 1.13e-8–2.38e-8).

Classification rows aggregate three seeds and 72 adapter instances per task.
`orthogonality_error_before = 2.809` shows how far the input factor drifts
before the pivot restores it to below 1e-6.

The two families use different optimizer-state policies — classification resets
`A` only, reasoning resets both.

### `regime_controls.csv` — where the advantage does not appear

`control, model, task, n, metric, method, mean, std, num_seeds, source`

- `summarization` — GPT-2 ROUGE-L on XSum and CNN/DailyMail under sequence-mean
  pooling. PivotLoRA is within 0.003 of LoRA on XSum, 0.016–0.017 lower on
  CNN/DailyMail.
- `full_data` — BERT on the full training split, seed 42. Differences span
  −0.009 to +0.022.
- `full_finetune` — full fine-tuning at N=64 (~110M trainable parameters).
  PivotLoRA matches or exceeds it on all six tasks with ~0.30M adapter
  parameters, roughly 370× fewer.

### `robustness_five_seed.csv` — five-seed confirmation

`task, n, method, mean, std, num_seeds, mean_gain_vs_lora, bootstrap_ci_low, bootstrap_ci_high, source`

Seeds 42–46 on seven settings with 95% bootstrap CIs on the mean gain. Run in a
separate environment from the primary three-seed tables.

### `compute.csv` — runtime and memory

`model, task, n, method, wall_time_seconds, wall_time_is_approximate, peak_memory_mb, peak_memory_gb, overhead_vs_lora_seconds, overhead_vs_lora_pct, hardware, source`

BERT figures average two seeds on one T4. PivotLoRA adds 4.0 s (+5.8%) and
2 MB, mostly from the per-rank optimizer groups rather than the QR pivot —
TT-LDA, which uses a uniform differential rate, costs −0.1 s. LLaMA figures are
approximate and reported in GB, as the paper reports them.
