# Run outputs

Artifacts written by the experiment scripts. `results/` holds the tidied
summaries that match the paper's tables; this directory holds what the runs
actually produced.

Superseded runs, smoke tests, and the raw 243-run learning-rate sweep are not
included.

## `reasoning/` — GSM8K and SVAMP

| File | Contents |
|---|---|
| `gsm8k_results.csv`, `svamp_results.csv` | per-seed accuracy, correct counts, wall time |
| `gsm8k_protocol_manifest.json`, `svamp_protocol_manifest.json` | the frozen protocol as recorded at run time |
| `pivot_diagnostics/{gsm8k,svamp}_seed_{42,43,44}.json` | per-adapter preservation and orthogonality errors for all 64 adapters |
| `predictions/svamp_{lora,pivotlora}_seed_{42,43,44}.jsonl` | every SVAMP test prediction, one record per example |
| `logs/{gsm8k,svamp}_full_run.log` | full run logs, including the hardware banner and SVAMP's sampled IDs |

These reproduce the paper's reasoning tables exactly. The GSM8K log predates
the `gsm8k_rebuttal.py` → `run_gsm8k.py` rename and still prints a
`REBUTTAL RESULT:` banner; it is otherwise the reported artifact, left
unedited. GSM8K has no prediction JSONL — the evaluator records accuracy and an
invalid count but never wrote per-example records.

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
