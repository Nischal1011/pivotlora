# Run outputs

Artifacts written by the experiment scripts. `results/` holds the tidied
summaries that match the paper's tables; this directory holds what the runs
actually produced.

Superseded runs, smoke tests, and the raw 243-run learning-rate sweep are not
included.

## `reasoning/` — GSM8K and StrategyQA

| File | Contents |
|---|---|
| `{gsm8k,strategyqa}_results.csv` | per-run accuracy, correct counts, parse rate, format compliance, wall time; includes the base-model reference |
| `{gsm8k,strategyqa}_protocol_manifest.json` | the protocol as recorded at run time, including the prompt and completion templates |
| `pivot_diagnostics/{dataset}_seed_{42,43,44}.json` | per-adapter preservation and orthogonality errors for all 64 adapters |
| `predictions/{dataset}_{base,lora,pivotlora}*.jsonl` | every test prediction, one record per example, with format-compliance flags |
| `logs/` | full run logs, including the hardware banner and sampled IDs. `gsm8k_full_run_attempt1_gpu_error.log` is a first attempt that stopped on a CUDA error while the GPU was shared with another process; the study resumed from the completed runs. `strategyqa_screening_run.log` is the base + LoRA screening run, whose rows the full study reuses. |
| `superseded_v1/` | the originally released reasoning outputs, produced with the `"Answer: "` prompt boundary described in the main README. Kept for transparency; not used in the paper. |

These reproduce the paper's reasoning tables exactly. Local paths in the logs
are replaced with `<workdir>`.

## `bert/`

| File | Contents |
|---|---|
| `main_bert_results.csv` | per-seed scores, 6 tasks × 5 N × 3 seeds × 13 methods |
| `ablation_pivot_epoch.csv` | pivot timing K |
| `ablation_optimizer_state.csv` | optimizer-state reset policy |
| `direction_vs_lr.csv` | direction versus eigenvalue-scaled rates |
| `lr_sweep_agnews.csv` | AG News learning-rate sensitivity |
| `full_finetune_baseline.csv` | full fine-tuning at N=64 |
| `full_data_audit.csv`, `full_data_audit_environment.json` | full-training-split reference plus the one captured environment snapshot |
| `activation_source_{raw,summary,diagnostics}.csv` | target-position vs. all-token, with vector counts and explained-variance ratios |
| `activation_source_minimal_raw.csv` | the two-condition grid behind `results/activation_source.csv` |
| `alignment/` | ρ per seed and subset, bootstrap summaries, class-imbalance analysis, and the plots |

`main_bert_results.csv` comes from a re-execution of the benchmark and tracks
the paper's tables closely rather than exactly; `results/bert_main.csv` holds
the reported values.

## `decoder/` and `summarization/`

`decoder/llama31_results.csv` is the per-seed Llama-3.1-8B classification
output. `summarization/{xsum,cnndm}.csv` are the GPT-2 ROUGE-L controls.
