# Three-way reconciliation

Published = camera-ready tex as of 2026-10-27 (Tables 4/6 hold the July values) and, in brackets,
the submitted tex (`HackerSignal_NeurIPS.bak-2026-10-27`). July = `hackersignal_corrected_results.csv`.
Rerun = this audit (`results.csv`). Δ = rerun − published (current tex). Flag: |Δ| > 0.005.

## Table 4 — Task 1

| model | metric | published [submitted] | July | rerun | Δ | verdict |
|---|---|---|---|---|---|---|
| MiniLM-L6-v2 | r_at_1 | 0.656 [0.431] | 0.6563 | 0.6553 | -0.0007 | match |
| MiniLM-L6-v2 | r_at_5 | 0.776 [0.656] | 0.7764 | 0.7764 | +0.0004 | match |
| MiniLM-L6-v2 | r_at_10 | 0.809 [0.727] | 0.8090 | 0.8090 | +0.0000 | match |
| MiniLM-L6-v2 | mrr | 0.709 [0.53] | 0.7091 | 0.7085 | -0.0005 | match |
| mpnet-base-v2 | r_at_1 | 0.626 [0.46] | 0.6256 | 0.6256 | -0.0004 | match |
| mpnet-base-v2 | r_at_5 | 0.756 [0.64] | 0.7563 | 0.7563 | +0.0003 | match |
| mpnet-base-v2 | r_at_10 | 0.792 [0.718] | 0.7925 | 0.7925 | +0.0005 | match |
| mpnet-base-v2 | mrr | 0.683 [0.54] | 0.6826 | 0.6823 | -0.0007 | match |
| BGE-base-v1.5 | r_at_1 | 0.654 [0.429] | 0.6538 | 0.6533 | -0.0007 | match |
| BGE-base-v1.5 | r_at_5 | 0.789 [0.666] | 0.7894 | 0.7889 | -0.0001 | match |
| BGE-base-v1.5 | r_at_10 | 0.822 [0.742] | 0.8216 | 0.8226 | +0.0006 | match |
| BGE-base-v1.5 | mrr | 0.714 [0.534] | 0.7141 | 0.7141 | +0.0001 | match |
| E5-base-v2 | r_at_1 | 0.649 [0.493] | 0.6487 | 0.6492 | +0.0002 | match |
| E5-base-v2 | r_at_5 | 0.789 [0.707] | 0.7889 | 0.7894 | +0.0004 | match |
| E5-base-v2 | r_at_10 | 0.825 [0.769] | 0.8246 | 0.8236 | -0.0014 | match |
| E5-base-v2 | mrr | 0.71 [0.587] | 0.7096 | 0.7097 | -0.0003 | match |
| BM25 | r_at_1 | 0.594 [0.445] | 0.5940 | 0.5950 | +0.0010 | match |
| BM25 | r_at_5 | 0.69 [0.64] | 0.6899 | 0.6884 | -0.0016 | match |
| BM25 | r_at_10 | 0.716 [0.707] | 0.7161 | 0.7151 | -0.0009 | match |
| BM25 | mrr | 0.636 [0.528] | 0.6357 | 0.6357 | -0.0003 | match |
| Hybrid BM25+mpnet | r_at_1 | 0.678 [0.486] | 0.6779 | 0.6789 | +0.0009 | match |
| Hybrid BM25+mpnet | r_at_5 | 0.755 [0.675] | 0.7548 | 0.7533 | -0.0017 | match |
| Hybrid BM25+mpnet | r_at_10 | 0.77 [0.736] | 0.7698 | 0.7678 | -0.0022 | match |
| Hybrid BM25+mpnet | mrr | 0.712 [0.567] | 0.7116 | 0.7110 | -0.0010 | match |

## Table 6 — Task 3

| model | metric | published [submitted] | July | rerun | Δ | verdict |
|---|---|---|---|---|---|---|
| MiniLM-L6-v2 | r_at_1 | 0.671 [0.453] | 0.6712 | 0.6702 | -0.0008 | match |
| MiniLM-L6-v2 | r_at_5 | 0.788 [0.669] | 0.7877 | 0.7877 | -0.0003 | match |
| MiniLM-L6-v2 | r_at_10 | 0.82 [0.728] | 0.8198 | 0.8198 | -0.0002 | match |
| MiniLM-L6-v2 | mrr | 0.723 [0.546] | 0.7229 | 0.7223 | -0.0007 | match |
| mpnet-base-v2 | r_at_1 | 0.647 [0.461] | 0.6470 | 0.6470 | -0.0000 | match |
| mpnet-base-v2 | r_at_5 | 0.777 [0.657] | 0.7771 | 0.7766 | -0.0004 | match |
| mpnet-base-v2 | r_at_10 | 0.81 [0.725] | 0.8098 | 0.8093 | -0.0007 | match |
| mpnet-base-v2 | mrr | 0.704 [0.544] | 0.7036 | 0.7032 | -0.0008 | match |
| BGE-base-v1.5 | r_at_1 | 0.668 [0.438] | 0.6675 | 0.6670 | -0.0010 | match |
| BGE-base-v1.5 | r_at_5 | 0.799 [0.677] | 0.7987 | 0.7982 | -0.0008 | match |
| BGE-base-v1.5 | r_at_10 | 0.828 [0.753] | 0.8277 | 0.8282 | +0.0002 | match |
| BGE-base-v1.5 | mrr | 0.726 [0.546] | 0.7258 | 0.7257 | -0.0003 | match |
| E5-base-v2 | r_at_1 | 0.663 [0.509] | 0.6628 | 0.6633 | +0.0003 | match |
| E5-base-v2 | r_at_5 | 0.798 [0.725] | 0.7977 | 0.7987 | +0.0007 | match |
| E5-base-v2 | r_at_10 | 0.831 [0.779] | 0.8309 | 0.8298 | -0.0012 | match |
| E5-base-v2 | mrr | 0.721 [0.602] | 0.7211 | 0.7213 | +0.0003 | match |
| BM25 | r_at_1 | 0.619 [0.443] | 0.6185 | 0.6196 | +0.0006 | match |
| BM25 | r_at_5 | 0.714 [0.654] | 0.7144 | 0.7129 | -0.0011 | match |
| BM25 | r_at_10 | 0.741 [0.715] | 0.7413 | 0.7403 | -0.0007 | match |
| BM25 | mrr | 0.66 [0.533] | 0.6602 | 0.6602 | +0.0002 | match |
| Hybrid BM25+mpnet | r_at_1 | 0.697 [0.491] | 0.6965 | 0.6976 | +0.0006 | match |
| Hybrid BM25+mpnet | r_at_5 | 0.774 [0.69] | 0.7740 | 0.7724 | -0.0016 | match |
| Hybrid BM25+mpnet | r_at_10 | 0.788 [0.743] | 0.7882 | 0.7861 | -0.0019 | match |
| Hybrid BM25+mpnet | mrr | 0.73 [0.577] | 0.7303 | 0.7298 | -0.0002 | match |

## Table 5 — Task 2 (rerun = primary variant, mean over seeds)

| model | metric | published | July | rerun (variant) | Δ | verdict |
|---|---|---|---|---|---|---|
| Decision Tree | macro_f1 | 0.708 | 0.7406 | 0.7020 (repo_trunc2000) | -0.0060 | **DIFFERS** |
| Decision Tree | f1_info_disclosure | 0.46 | 0.5373 | 0.4490 (repo_trunc2000) | -0.0110 | **DIFFERS** |
| Decision Tree | f1_memory_corruption | 0.65 | 0.6723 | 0.6436 (repo_trunc2000) | -0.0064 | match |
| Decision Tree | f1_auth_access | 0.6 | 0.6961 | 0.6018 (repo_trunc2000) | +0.0018 | match |
| TF-IDF + LR | macro_f1 | 0.772 | 0.7717 | 0.7717 (repo_trunc2000) | -0.0003 | match |
| TF-IDF + LR | f1_info_disclosure | 0.66 | 0.6569 | 0.6569 (repo_trunc2000) | -0.0031 | match |
| TF-IDF + LR | f1_memory_corruption | 0.69 | 0.7308 | 0.6852 (repo_trunc2000) | -0.0048 | match |
| TF-IDF + LR | f1_auth_access | 0.79 | 0.7939 | 0.7885 (repo_trunc2000) | -0.0015 | match |
| SVM | macro_f1 | 0.799 | 0.7817 | 0.7990 (repo_trunc2000) | +0.0000 | match |
| SVM | f1_info_disclosure | 0.7 | 0.6592 | 0.6957 (repo_trunc2000) | -0.0043 | match |
| SVM | f1_memory_corruption | 0.68 | 0.6783 | 0.6838 (repo_trunc2000) | +0.0038 | match |
| SVM | f1_auth_access | 0.82 | 0.8173 | 0.8227 (repo_trunc2000) | +0.0027 | match |
| RNN | macro_f1 | 0.142 | 0.6944 | 0.4361 (packed_last_token) | +0.2941 | **DIFFERS** |
| RNN | f1_info_disclosure | 0.12 | 0.5519 | 0.2979 (packed_last_token) | +0.1779 | **DIFFERS** |
| RNN | f1_memory_corruption | 0.1 | 0.6916 | 0.4171 (packed_last_token) | +0.3171 | **DIFFERS** |
| RNN | f1_auth_access | 0.2 | 0.6882 | 0.4403 (packed_last_token) | +0.2403 | **DIFFERS** |
| GRU | macro_f1 | 0.826 | 0.8628 | 0.8579 (packed_last_token) | +0.0319 | **DIFFERS** |
| GRU | f1_info_disclosure | 0.69 | 0.7112 | 0.6975 (packed_last_token) | +0.0075 | match |
| GRU | f1_memory_corruption | 0.78 | 0.8431 | 0.8416 (packed_last_token) | +0.0616 | **DIFFERS** |
| GRU | f1_auth_access | 0.8 | 0.8267 | 0.8115 (packed_last_token) | +0.0115 | **DIFFERS** |
| LSTM | macro_f1 | 0.826 | 0.8440 | 0.8240 (packed_last_token) | -0.0020 | match |
| LSTM | f1_info_disclosure | 0.68 | 0.6921 | 0.6306 (packed_last_token) | -0.0494 | **DIFFERS** |
| LSTM | f1_memory_corruption | 0.82 | 0.8190 | 0.7707 (packed_last_token) | -0.0493 | **DIFFERS** |
| LSTM | f1_auth_access | 0.8 | 0.8054 | 0.7805 (packed_last_token) | -0.0195 | **DIFFERS** |
| BiLSTM | macro_f1 | 0.871 | 0.8197 | 0.8556 (packed_last_token) | -0.0154 | **DIFFERS** |
| BiLSTM | f1_info_disclosure | 0.73 | 0.6517 | 0.6863 (packed_last_token) | -0.0437 | **DIFFERS** |
| BiLSTM | f1_memory_corruption | 0.86 | 0.8081 | 0.8403 (packed_last_token) | -0.0197 | **DIFFERS** |
| BiLSTM | f1_auth_access | 0.82 | 0.7758 | 0.8133 (packed_last_token) | -0.0067 | match |
| SecBERT | macro_f1 | 0.846 | 0.8545 | 0.8525 (subsample15000) | +0.0065 | **DIFFERS** |
| SecBERT | f1_info_disclosure | 0.69 | 0.6852 | 0.6663 (subsample15000) | -0.0237 | **DIFFERS** |
| SecBERT | f1_memory_corruption | 0.79 | 0.8039 | 0.8150 (subsample15000) | +0.0250 | **DIFFERS** |
| SecBERT | f1_auth_access | 0.82 | 0.8500 | 0.8259 (subsample15000) | +0.0059 | match |
| SecureBERT | macro_f1 |  | 0.8966 | 0.8920 (subsample15000) |  | new row |
| SecureBERT | f1_info_disclosure |  | 0.7799 | 0.8001 (subsample15000) |  | new row |
| SecureBERT | f1_memory_corruption |  | 0.8846 | 0.8957 (subsample15000) |  | new row |
| SecureBERT | f1_auth_access |  | 0.8618 | 0.8547 (subsample15000) |  | new row |
| CySecBERT | macro_f1 |  | 0.9122 | 0.8981 (subsample15000) |  | new row |
| CySecBERT | f1_info_disclosure |  | 0.8131 | 0.7864 (subsample15000) |  | new row |
| CySecBERT | f1_memory_corruption |  | 0.9358 | 0.9032 (subsample15000) |  | new row |
| CySecBERT | f1_auth_access |  | 0.8780 | 0.8639 (subsample15000) |  | new row |

Per-class published values have two decimals, so per-class rows use tolerance 0.01.
Explanations for every DIFFERS row are in discrepancies.md (cause column).
