# Camera-ready ablations

Run from the repository root.

| Folder | Paper | Command |
|---|---|---|
| `post_pivot_lr/` | Appendix F.10, post-pivot learning-rate policy (BERT-base; SST-2, CoLA, AG News; N in {16,...,256}; 3 seeds) | `python ablations/post_pivot_lr/pivot_lr_ablation.py --out pivot_lr_ablation.csv` |
| `gpt2_summarization/` | Appendix E.4, summarization control (GPT-2; XSum, CNN/DailyMail; N in {64, 256}; 3 seeds) | `python ablations/gpt2_summarization/gpt2_response_ablation.py --variants lora pivot_seqmean pivot_resp pivot_seqmean_unilr pivot_resp_unilr --out gpt2_response_ablation.csv` |

Each folder contains the script and the per-seed results used in the paper. Both scripts resume from an existing output CSV.
