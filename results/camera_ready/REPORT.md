# HackerSignal camera-ready audit — report (2026-10-28)

Worktree `_worktrees/exploit-text-graph/camera-ready-audit` (branch `audit/camera-ready` from e0c1153; nothing
committed or pushed). Every number below comes from a run on the released data recorded in
`results/camera_ready/results.csv` (row-level provenance in `provenance.json`). None is copied from the paper.

## Matched

- **Data.** All 15 split and corpus files match the HF release (commit fc56ae0) and the July tarball byte-for-byte.
  Table 3 row counts are exact; the corpus has 340,536 rows. Task 3 is CVE-disjoint (0/0/0) and its test set is a
  subset of Task 1's (1,898/1,898). Task 2 dates: train ≤2019-12-31, val 2020–2022, test ≥2023-01-04.
- **Tables 4 and 6** (the July values put into the tex on 10/27): all 48 cells reproduce within 0.001 in fp32 on
  the full test sets. The masking ladder matches July within 0.0023 MRR (65 cells).
- **Text claims.** The Task 1 paired Δ 0.003 [−0.013, 0.019] and the "≈0.02" CI half-width hold.
- **Published BoW rows** reproduce exactly with the repo's 2,000-character cut (LR .772, SVM .799; DT .708 lies
  within .702 ± .007).
- **Hybrid.** The code does what Appendix D says: 0.7·dense + 0.3·BM25 over the BM25 top-100.

## Changed

- **Table 5** (protocol per Ben: BoW 2,000-char cut; corrected recurrent models; transformers on the 15K
  subsample with the epoch picked on validation; 3 seeds).
  - The published RNN .142 is a padding bug: the code reads `h_n` after up to 300 pad steps. Corrected: .436 ± .031.
    GRU, LSTM and BiLSTM are unaffected by the bug.
  - The best model changes from BiLSTM (.871 published, now .856) to CySecBERT .898 / SecureBERT .892, which do
    not differ significantly.
- **Task 3 paired Δ:** 0.005 → **0.004** [−0.012, 0.020].
- **Appendix C** says there is no train/test CVE overlap "for any task"; Task 1 has 9 shared CVEs.
- **Table 3 caption** says "test 2024+" for Task 1; 92 of its test queries link to pre-2024 CVEs.
- **Appendix D** omits several protocol details:
  - the 384-character input cut for the dense encoders;
  - the 2,000-character cut for BoW;
  - for the transformers: the 1,000-character cut, the unweighted loss and validation-based epoch selection.
  - The vocabulary is 175,062 tokens, not ~19K.
  - The compute paragraph and checklist describe no run that produced the numbers.
- **`all_results.json`**: its Task 2 rows were estimates (Macro-F1 + 0.036 / + 0.032) marked `verified: true`, and
  the radar figure plotted them. `scripts/plot_baseline_radar.py` now reads `results.csv` only and fails on a
  missing value.

## Could not reproduce / not run

- DarkBERT: the model is gated; Ben has requested access, and it is recorded as unavailable.
- Appendix C's MinHash near-duplicate check was not re-run; exact duplicates are 0. The "41/25" figures in the
  brief appear in no repo file.
- Tables 1–2 (corpus statistics) and the "per-source audit risk scores" are not baseline runs (no code computes the
  scores), so they were not regenerated.
- **B6 (bi-encoder fine-tuned on the Task 3 training split): done, folded into `results.csv`.** Branch
  `exp/b6-finetuned-biencoder` (171b7a6); 56,833 pre-2022 (query, gold description) pairs, one epoch, cached MNRL
  (batch 64 in mini-batches of 16, scale 20, no duplicate CVEs per batch), lr 2e-5, 10% warmup, fp32, seeds 42–44,
  Apple M5 Pro; no model selection on validation. The B6 harness reproduces the audit's zero-shot rows (BGE 0.725673,
  MiniLM 0.722305). Fine-tuned rows use their own bootstrap stream, so no existing CI changed (diff: additions only).

  | model | zero-shot MRR@10 | fine-tuned (mean ± SD, 3 seeds) | 95% CI | paired Δ [95% CI] | R@1 |
  |---|---|---|---|---|---|
  | BGE-base-v1.5 | 0.726 | 0.757 ± 0.005 | [0.740, 0.774] | 0.031 [0.019, 0.044] | 0.667 → 0.693 |
  | MiniLM-L6-v2 | 0.722 | 0.750 ± 0.002 | [0.732, 0.767] | 0.028 [0.016, 0.039] | 0.670 → 0.688 |

  Per seed (test MRR@10): BGE 0.7533 / 0.7556 / 0.7624; MiniLM 0.7493 / 0.7522 / 0.7479.
  Run history: the first launch ran out of memory at step 637 (plain MNRL, batch 64), which led to the cached loss
  and eager attention. BGE seed 43 had one MPS command-buffer error at step 103 and trained on to a normal loss
  (0.1237). Its evaluation then ran out of GPU memory (two macOS VMs held about 20 GB) and the mini rebooted at 14:48.
  171b7a6 evaluates from the saved weights with default attention; the rerun resumed at the final step, so that row
  records loss 0. provenance.json carries the run3 loss and time.
- **Full-64K transformer runs (appendix only): done on the GSU cluster, NVIDIA L40S.** Array 4777484, 9 tasks, all
  exit 0 (collected 2026-10-28 16:45 UTC; rows in `raw/task2_cluster_tf_full.jsonl`, sha256 275ade36…16d4, verified
  against the cluster output). Same code commit (6ee2e1f) and data hashes as the mini runs; best epoch by validation;
  mean ± SD (ddof=1) over seeds 42–44. Per-query predictions stay on the cluster, so these rows have no bootstrap CI.
  The two mini (MPS) SecBERT full_64413 runs are superseded and marked so in `results.csv`.

  | model | macro-F1 (64K, L40S) | 15K subsample (M5 Pro) | info_disc. | mem_corr. | auth. |
  |---|---|---|---|---|---|
  | SecBERT | 0.899 ± 0.001 | 0.852 ± 0.007 | 0.753 | 0.904 | 0.878 |
  | SecureBERT | 0.903 ± 0.006 | 0.892 ± 0.008 | 0.805 | 0.899 | 0.878 |
  | CySecBERT | 0.901 ± 0.011 | 0.898 ± 0.003 | 0.809 | 0.877 | 0.879 |

  More training data lifts all three to about 0.90 and erases the 15K gap between SecBERT and the other two.
  The SDs overlap, so no transformer leads on the full training set.

## Camera-ready values (generated from results.csv)

**Table 4 — Task 1 (1,990 queries)**

| model | R@1 | R@5 | R@10 | MRR@10 |
|---|---|---|---|---|
| BM25 | 0.595 | 0.688 | 0.715 | 0.636 |
| MiniLM-L6-v2 | 0.655 | 0.776 | 0.809 | 0.709 |
| mpnet-base-v2 | 0.626 | 0.756 | 0.792 | 0.682 |
| BGE-base-v1.5 | 0.653 | 0.789 | 0.823 | **0.714** |
| E5-base-v2 | 0.649 | **0.789** | **0.824** | 0.710 |
| Hybrid BM25+mpnet | **0.679** | 0.753 | 0.768 | 0.711 |

**Table 6 — Task 3 (1,898 queries)**

| model | R@1 | R@5 | R@10 | MRR@10 |
|---|---|---|---|---|
| BM25 | 0.620 | 0.713 | 0.740 | 0.660 |
| MiniLM-L6-v2 | 0.670 | 0.788 | 0.820 | 0.722 |
| mpnet-base-v2 | 0.647 | 0.777 | 0.809 | 0.703 |
| BGE-base-v1.5 | 0.667 | 0.798 | 0.828 | 0.726 |
| E5-base-v2 | 0.663 | **0.799** | **0.830** | 0.721 |
| Hybrid BM25+mpnet | **0.698** | 0.772 | 0.786 | **0.730** |

These differ from the tex's current July (fp16) cells by at most 0.002, and the Task 3 R@5 bold moves from BGE to
E5 (0.7987 vs 0.7982; a tie in practice). The proposed diff switches the tables to these audit-backed values.

**Table 5 — Task 2 (1,735 test samples; macro-F1 mean ± SD over seeds 42/43/44; per-class F1 seed mean)**

| | model | macro-F1 | info_disc. | mem_corr. | auth. |
|---|---|---|---|---|---|
| BoW | Decision Tree | .702 ± .007 | .45 | .64 | .60 |
| | TF-IDF + LR | .772 ± .000 | .66 | .69 | .79 |
| | SVM (linear) | .799 ± .000 | .70 | .68 | .82 |
| Recurrent | RNN | .436 ± .031 | .30 | .42 | .44 |
| | GRU | .858 ± .001 | .70 | .84 | .81 |
| | LSTM | .824 ± .013 | .63 | .77 | .78 |
| | BiLSTM | .856 ± .003 | .69 | .84 | .81 |
| Transformer | SecBERT | .852 ± .007 | .67 | .82 | .83 |
| | SecureBERT | .892 ± .008 | **.80** | .90 | .85 |
| | CySecBERT | **.898 ± .003** | .79 | **.90** | **.86** |

Paired deltas (bootstrap 95% CI):

| task | comparison | Δ | 95% CI |
|---|---|---|---|
| Task 1 | BGE − Hybrid (MRR) | 0.003 | [−0.013, 0.019] |
| Task 3 | Hybrid − BGE (MRR) | 0.004 | [−0.012, 0.020] |
| Task 2 | CySecBERT − SecureBERT (macro-F1) | 0.006 | [−0.004, 0.016] |
| Task 2 | GRU − LSTM (macro-F1) | 0.029 | [0.016, 0.043] |

## Proposed wording for the data findings

- **Table 3 caption:** "Task 3 splits by CVE publication year (train <2022, val 2022–2023, test 2024+) and Task 1
  uses the same test queries by source date (92 of its 1,990 test queries link to CVEs published before 2024), both
  over the full 340K NVD corpus; …"
- **Table 3 footnote:** "zero CVE overlap between Task 3 train/val/test gold sets."
- **Appendix C:** "Finally, we check gold-CVE overlap between splits. For Task 3, CVE-disjointness is enforced by
  construction … Task 1 does not impose disjointness; 9 of its 1,876 test CVEs (9 of 1,990 test queries) also occur
  in training."

The full proposed tex change (37 edits) is in `results/camera_ready/proposed_paper/` (diff, tex, pdf). It compiles,
and the main-text pagination is unchanged. **It has not been applied.**

## Needs Ben's decision

1. Go-ahead to apply the proposed diff to the paper (and port it to the repo's `paper/` copy), then push the README
   "Baseline results" section.
2. **Year span:** the abstract says 1990–2026, Table 1 says 1991–26, and the released Task 2 train split contains
   1988-10-01 timestamps. `data/stats` comes from an older 8.74M-row build, so it can't settle the span; it needs a
   min/max over the released 7.45M corpus.
3. **Camera-ready header:** the tex is still the anonymous submission build. It needs `\usepackage[eandd, final]`
   and the author block (Ampel, Samtani), with affiliations I don't have.
4. Whether to keep the Appendix D sentence on full-text encoding (BGE 0.744 / E5 0.741 on Task 1) and BGE with its
   query instruction (0.725 / 0.737), or move it to a sensitivity table.
5. Bib entries for SecureBERT (SecureComm 2022, published 2023) and CySecBERT (ACM TOPS 27(2), 2024) follow the
   planner session's Crossref check.
6. (Done: full-64K runs from the GSU cluster, NVIDIA L40S, reported as a separate appendix table.)
6b. (Done: B6 in the Task 3 text and Appendix D Table 10. Ben, 10/28 20:30: keep it out of the abstract, and the two
    page-fit fixes are approved (duplicate Conclusion sentence cut; radar figure at 0.88\linewidth).)
6c. Abstract re-sync: the OpenReview abstract and the public poster page (neurips.cc/virtual/2026/poster/139194)
    still show the submitted abstract (64 sources, 1990–2026, anonymous HF URL). Poster moved to Atlanta Poster
    Session 1, Wed Dec 9, 10:00 AM–1:00 PM ET; no repo or paper text names the session.
7. The planner's Table 5 draft has two rounding slips against results.csv: GRU SD is .001 (0.00149), not .002;
   SecBERT macro-F1 is .852 (0.85249), not .853. Every other value agrees.
