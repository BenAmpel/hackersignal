# Discrepancies: paper vs code vs data (camera-ready audit, 2026-10-27/28)

One row per issue. "tex" = `reference/paper/HackerSignal_NeurIPS/HackerSignal_neurips2026.tex` as of 10/27 (its line
numbers). "July" = `reference/july_rerun/HackerSignal_corrected_rerun.ipynb` (cell numbers). Repo code at e0c1153.
All impact numbers come from this audit's runs (`results.csv`; mean ± SD over seeds 42/43/44 where stated).

Severity: **A** changes a published number or claim; **B** changes a protocol description; **C** hygiene.

## Data and splits

| # | sev | where | paper says | data / code does | impact |
|---|---|---|---|---|---|
| D1 | A | tex:576 (App. C) | "no CVE identifier appears in both the train and test for any task" | Task 1: 9 gold CVEs in train ∩ test (26 train∩val, 10 val∩test). Task 3: 0/0/0, as claimed. | False for Task 1; restrict the claim to Task 3. 9 of 1,990 Task 1 test rows affected. |
| D2 | A | tex:201 (Table 3 caption) | Tasks 1 and 3: "train <2022, test 2024+" | Task 3 holds by gold-CVE year (train 1999–2021, val 2022–23, test 2024–26). Task 1 test has 92 rows (all exploitdb) with gold CVEs from 2015–2023; 1 train row has a 2023 CVE; 78% of Task 1 test rows have an empty `timestamp`. | Caption overstates Task 1; see the proposed wording in the report. |
| D3 | B | test.jsonl | one gold CVE per query | Multi-CVE exploits appear as several rows with the same id and text: Task 1 5 ids → 13 rows, Task 3 3 ids → 9 rows. Under single-gold scoring at most one can be rank 1. | Merging them raises MRR by .002–.003 for every model; ranking unchanged (`grouped_multi_gold` rows). |
| D4 | B | Table 3 footnote "Same corpus" | one corpus | Task 1 and Task 3 corpus files differ in SHA-256 only because the key is `id` vs `cve_id`; ids, texts and order are identical. | None on results. |
| D5 | B | tex:576 | MinHash ≥0.8: zero train–test pairs | MinHash not re-run. Exact normalised test–train duplicates: 0 (Tasks 1–3). Same first 100 chars: 12 (Tasks 1, 3); same first line: 23. The "41/25" duplicate figures named in the brief appear in no repo file. | Exact-level claim holds; near-dup claim not re-verified. |
| D6 | C | data/audits/*.json | audit artefacts for the benchmark | `split_overlap_audit.json`, `task3_leakage_audit.json`, `quality_by_task.json` describe the v1 five-task benchmark (task3 = severity), not the v2 splits. | Paper points readers at stale audit files. |
| D7 | C | HF dataset | — | All 15 v2 split/corpus files at HF commit fc56ae0 match the local release byte-for-byte; HF still ships `benchmark_v2/results_task2.json` (old binary task). | Remove or relabel that file on HF. |
| D8 | B | Task 2 data | — | Train median 70 words vs test 212; 32% of test rows exceed the 300-token RNN window (13% of train). `dtl_exploits` supplies 25,962 train rows and 0 test rows. | Worth one sentence in the paper. |

## Retrieval (Tasks 1 and 3)

| # | sev | where | paper says | code does | impact |
|---|---|---|---|---|---|
| R1 | A | `run_retrieval_baselines.py` vs `run_baselines_v2.py` | one protocol | Two implementations. `run_retrieval_baselines.py:74,85,150,158` cut to `text[:384]` chars and rerank BM25 top-100 (`:131`). `run_baselines_v2.py:132,138,183,188` cut to `text[:512]` and fuse over the **full corpus** (`:193–214`). | Audit follows App. D + `run_retrieval_baselines.py`. Tables 4/6 reproduce within .001 on all 48 cells. |
| R2 | A | tex:589 | BGE/E5 rows in Tables 4/6; E5 with "query:"/"passage:" | No committed script ever ran BGE or E5. | Now reproduced from code. E5 prefixes immaterial (Task 1 MRR .7097 with vs .7101 without). BGE query instruction is **material**: MRR .7141 → .7252 (Task 1), .7257 → .7374 (Task 3); the paper calls BGE "instruction-tuned" but runs it without one. |
| R3 | B | tex:589 | "max sequence length 512 tokens" | `run_retrieval_baselines.py:64` never sets `max_seq_length` (defaults MiniLM 256, mpnet 384, BGE/E5 512); July sets 512. Text is cut to 384 **characters** first, which App. D never mentions. | **Material.** Full text up to 512 tokens changes Task 1 MRR by MiniLM −.025, mpnet +.033, BGE +.030, E5 +.032 (Task 3: BGE .750, E5 .748). Larger than the CI half-width (.02). |
| R4 | B | tex:589, 602 | batch 128; "5K-document chunks on CPU with multi-process pooling"; 16 GB machine; "no CUDA GPUs" | Code: batch 512, 50K chunks, MPS (`run_retrieval_baselines.py:56,62,69`). July: batch 512 fp16 on an L40S (CUDA). This audit: batch 128 fp32 on MPS, Apple M5 Pro 48 GB. | Compute paragraph describes no run that produced the numbers. |
| R5 | C | `run_retrieval_baselines.py:110` | (submitted text said RRF k=60; fixed 10/27) | Weighted sum 0.7·(cos+1)/2 + 0.3·BM25/max over BM25 top-100; docstring still says "RRF-style". | Docstring only; App. D matches the code and this audit. |
| R6 | B | `run_*` `mrr()` | "MRR" | Truncated at rank 10. | Label as MRR@10; `mrr_full` gives untruncated MRR. |
| R7 | C | `run_*:50,98,111,149` | — | `np.argsort(s)[-k:][::-1]` breaks ties toward the higher index; July `argpartition` is arbitrary. Audit ranks the gold exactly (lower index first) and stores optimistic/pessimistic bounds. | BM25 R@1 .5940 (July) vs .5950 = 2 tied queries; MRR identical. |
| R8 | C | July cell 21 | — | July encodes in fp16. | All 65 July ladder cells within .0023 MRR of the fp32 rerun. |

## Task 2 (exploit type)

| # | sev | where | paper says | code does | impact |
|---|---|---|---|---|---|
| C1 | A | `run_baselines_v2.py:366–375` | RNN .142, "long exploit texts require more advanced architectures" (tex:281) | Classifies from `h_n` after a **right-padded** 300-step sequence. | As released RNN .141±.003; corrected (last real token) .436±.031. GRU/LSTM/BiLSTM **unaffected** (as-released vs corrected .858/.858, .819/.824, .857/.856): their gates carry state through the zero pad embeddings. Only the vanilla RNN collapses. |
| C2 | B | July cell 32 | — | July pools with `out.mean(dim=1)` over all 300 positions, padding included — a third architecture. | Rerun (seed 42): RNN .706, GRU .862, LSTM .837, BiLSTM .794 (July .694/.863/.844/.820). |
| C3 | B | tex:599 | vocab "min frequency ≥3, yielding ~19K tokens" | min-freq 3 over the 64,413 train texts gives **175,062** entries. | Correct the number. |
| C4 | A | `run_baselines_v2.py:270,285` | TF-IDF(50k, (1,2)) | Text cut to 2,000 characters; App. D omits this. July used full text. | With the cut: DT .702±.007, LR .772, SVM .799 (published .708/.772/.799). Full text: .735/.770/.783. Decision (Ben): keep the cut, state it. |
| C5 | A | `run_baselines_v2.py:426–518` | SecBERT protocol | Repo: `text[:1000]`, **unweighted** CE (`outputs.loss`, :489), no clipping, **no validation** (scored after the last epoch). July: class-weighted CE (weights from the full 64K train), clip 1.0, full text, last epoch. | Repo protocol, val-selected, 3 seeds: SecBERT .852±.007 (published .846), SecureBERT .892±.008, CySecBERT .898±.003. July protocol (seed 42): .862/.906/.899. Val selection picked the last epoch in 8 of 9 runs; only SecureBERT moves (.892 vs .897 final-epoch). |
| C6 | A | tex:599 | SecBERT "on a 15K subsample" | Subsample drawn by `default_rng(42).choice` (repo) vs `RandomState(SEED).choice` (July): different rows. Full 64,413 never run before. | `full_64413` rows (appendix, 1 seed). |
| C7 | A | `data/benchmark_v2/all_results.json` (commit 5db7c2c) | Table 5 source | Every Task 2 row, **BoW included**, has Weighted-F1 = Macro-F1 + 0.036 and Accuracy = Macro-F1 + 0.032, marked `"verified": true` (measured SVM: .838/.836 vs stored .835/.831). Tasks 1/3 rows still hold the 550/393-query values. | Must not be cited; superseded by results.csv. |
| C8 | A | `scripts/plot_baseline_radar.py:140–181, 286–335` (e0c1153) | radar figure | Neural Task 2 values hard-coded from the paper, Weighted-F1/Accuracy estimated by BoW offset (`_fallback_neural_metrics`); Tasks 1/3 from stale `all_results.json`; RNN silently dropped. | **Fixed on this branch:** reads results.csv only, estimation path deleted, exits with an error on any missing value; all 16 models plotted. |
| C9 | C | `baselines.py`, `serious_baselines.py` | — | Both target the v1 five-task benchmark (`data/benchmark/`); `serious_baselines.py:259` hybrid uses α=0.5 on the sparse score. Neither produced a v2 number. | Mark as legacy in the code release. |
| C10 | B | tex:602 | RNN "~3 min each"; SecBERT "~90 min" | Measured (M5 Pro, GPU shared by two jobs): BoW 7–40 s; RNN/GRU/LSTM 5–36 min; BiLSTM (packed) ~61 min; SecBERT ~17 min, SecureBERT/CySecBERT ~40–45 min per 15K run; full-corpus encoding 4 min (MiniLM) to ~30 min (base). | Rewrite from provenance.json. |
| C11 | B | — | one run per model | No seeds; MPS is not bit-deterministic. | Seeds change a conclusion: July's single-run CySecBERT − SecureBERT Δ "excludes zero"; with 3 seeds Δ .006 [−.004, .016] includes zero. |
| C12 | C | DarkBERT | — | `s2w-ai/DarkBERT` is gated; licence not accepted. | Not run; Ben has requested access himself. |
| C13 | C | audit implementation | — | Corrected unidirectional RNN/GRU/LSTM: seed 42 of RNN used `pack_padded_sequence`, the rest gather the output at the last real token (verified equal on MPS to <6e-7, ~3× faster). BiLSTM always packs. | None. |

## Text claims checked against the rerun

| # | sev | where | paper says | rerun | fix |
|---|---|---|---|---|---|
| T1 | B | tex:312 | Task 3 paired ΔMRR 0.005 [−0.011, 0.020] | 0.0041 [−0.0115, 0.0197] | write 0.004 [−0.012, 0.020] |
| T2 | — | tex:247 | Task 1 paired ΔMRR 0.003 [−0.013, 0.019] | 0.0031 [−0.0129, 0.0190] | matches |
| T3 | — | tex:227, 294 | bootstrap half-width "about 0.02" | 0.018–0.021 | matches |
| T4 | A | tex:281 | BiLSTM .871 best; RNN .142; GRU ≈ LSTM .826; SecBERT .846 < BiLSTM; "bidirectionality is key" | BiLSTM .856, GRU .858 > LSTM .824 (Δ .029 [.016, .043]), RNN .436, SecBERT .852 ≈ BiLSTM; best CySecBERT .898 ≈ SecureBERT .892 (Δ .006 [−.004, .016]) | rewrite the paragraph |
| T5 | A | tex:335 | "BiLSTM at macro-F1~0.87" | best is CySecBERT .898 | rewrite |
| T6 | — | tex:281 | XSS and injection easier (F1 > 0.82) | mean over models: XSS .941, injection .864 | holds |
| T7 | — | Table 5 headers | three hardest = info_disc, mem_corr, auth | mean per-class F1: info .637, mem .750, auth .760 (dos .765) | holds |

## Not regenerated

- Tables 1–2 (7.45M-document corpus statistics) and the Jaccard figure: built from the full collection, not from a baseline run.
- App. C MinHash near-duplicate check: not re-run (exact and prefix duplicates measured instead, D5).
- 240-row manual audit packet: human annotation.
- "Per-source audit risk scores" (tex:337): no code or table computes them; per-source retrieval MRR is regenerated instead (`per_source` rows).
- The FP-CVE forum-post pilot is regenerated (`fp_cve` rows) although the current tex no longer reports it.
- B6 (fine-tuned bi-encoder): run by another session; it crashed (MPS OOM) on 10/28 06:37 and will be restarted after the full-64K runs.
