# HackerSignal — Replication Guide

Anonymous repository for:  
**"HackerSignal: A Temporally Structured, Multi-Source Benchmark for Cybersecurity Threat Intelligence"**  
*NeurIPS 2026 Evaluations & Datasets Track (under review)*

Dataset (full + test splits): [`DatasetSubmission/HackerSignal`](https://huggingface.co/datasets/DatasetSubmission/HackerSignal)

---

## Table of Contents

1. [Repository layout](#1-repository-layout)
2. [Prerequisites](#2-prerequisites)
3. [Data download](#3-data-download)
4. [Replicating the baselines (Tables 2–4)](#4-replicating-the-baselines)
5. [Regenerating figures](#5-regenerating-figures)
6. [Regenerating LaTeX paper](#6-regenerating-the-latex-paper)
7. [Collecting new data](#7-collecting-new-data)
8. [Expected runtimes](#8-expected-runtimes)

---

## 1. Repository layout

```
exploit-text-graph/
├── src/etg/
│   ├── collectors/         # Data collection scripts (see §7)
│   ├── benchmark/          # Baseline models and evaluation
│   │   ├── baselines.py            # TF-IDF, BM25, bi-encoder, SecBERT classifiers
│   │   ├── run_baselines_v2.py     # Per-task baseline runner
│   │   ├── serious_baselines.py    # Orchestrated multi-seed overnight runner
│   │   └── splits.py              # Temporal split construction
│   ├── data/
│   │   ├── dataset_stats.py       # Generates data/stats/
│   │   └── unify_hacker_communities.py  # Builds unified JSONL corpus
│   └── viz/
│       └── publication.py         # Shared matplotlib style
├── scripts/
│   ├── build_task2_exploit_type.py     # Build Task 2 ETC splits
│   ├── plot_baseline_radar.py          # Figure: baseline_results_summary.{pdf,png}
│   ├── plot_temporal_heatmap.py        # Figure: temporal_heatmap.{pdf,png}
│   ├── plot_domain_overlap.py          # Figure: domain_overlap.{pdf,png}
│   └── plot_corpus_stats.py            # Figure: corpus_temporal_distribution.{pdf,png}
├── data/
│   ├── benchmark_v2/               # Test splits + precomputed results (committed)
│   │   ├── task1_cve_linkage/test.jsonl
│   │   ├── task2_exploit_type/test.jsonl
│   │   ├── task3_temporal_generalization/test.jsonl
│   │   └── all_results.json        # Canonical reported metrics
│   ├── stats/                      # Corpus statistics (committed)
│   ├── audits/                     # LLM-judge and manual audit artifacts (committed)
│   └── release/                    # Source release manifest + Croissant (committed)
├── paper/
│   ├── HackerSignal_neurips2026.tex    # Paper LaTeX source
│   └── figures/                        # Pre-built figures (PDF/PNG)
└── docs/
    ├── datasheet.md
    └── release_governance.md
```

---

## 2. Prerequisites

```bash
# Python 3.10+ recommended
cd exploit-text-graph
pip install -e .

# For dense retrieval baselines (Tasks 1 & 3):
pip install -e ".[sentence-transformers]"

# For SecBERT classification (Task 2):
pip install -e ".[transformers]"

# Verify installation
python -c "import etg; print('etg OK')"
```

**Hardware**: All baselines run on CPU. Neural models (SecBERT, bi-encoder) run faster on GPU/MPS. The full 5-seed overnight suite requires ~8 hours on an M-series Mac or ~3 hours on an A100.

---

## 3. Data download

The benchmark test splits are already committed in `data/benchmark_v2/` and can be used immediately. Train/val splits and the full corpus are available on HuggingFace:

```python
from datasets import load_dataset

# Full corpus
ds = load_dataset("DatasetSubmission/HackerSignal", name="corpus")

# Task splits (train + val + test)
task1 = load_dataset("DatasetSubmission/HackerSignal", name="task1_cve_linkage")
task2 = load_dataset("DatasetSubmission/HackerSignal", name="task2_exploit_type")
task3 = load_dataset("DatasetSubmission/HackerSignal", name="task3_temporal_generalization")

# Save to expected locations
task1.save_to_disk("data/benchmark_v2/task1_cve_linkage")
task2.save_to_disk("data/benchmark_v2/task2_exploit_type")
task3.save_to_disk("data/benchmark_v2/task3_temporal_generalization")
```

Or with the CLI:

```bash
# Download and place splits
python -c "
from datasets import load_dataset
for name in ['task1_cve_linkage', 'task2_exploit_type', 'task3_temporal_generalization']:
    ds = load_dataset('DatasetSubmission/HackerSignal', name=name)
    for split, data in ds.items():
        data.to_json(f'data/benchmark_v2/{name}/{split}.jsonl')
print('Done')
"
```

---

## 4. Replicating the baselines

### Quick verification (test split only, ~10 minutes)

Run all lexical and dense baselines on the test splits:

```bash
PYTHONPATH=src python3 -m etg.benchmark.run_baselines_v2 \
    --tasks 1 2 3 \
    --benchmark-dir data/benchmark_v2 \
    --output data/results/quick_run.json
```

This reproduces the BM25, TF-IDF+LR, SVM, MiniLM, and MPNet rows in Tables 2–4.

### Full multi-seed replication (~3–8 hours)

The overnight suite adds SecBERT (Task 1 & 3) and bi-encoder/hybrid/reranker (Task 2), each over 5 random seeds:

```bash
PYTHONPATH=src python3 -m etg.benchmark.serious_baselines \
    --benchmark-dir data/benchmark_v2 \
    --results-dir data/results/overnight \
    --mode full
```

Progress is written to `data/results/overnight/status/`. Results aggregate to `data/results/overnight/overnight_summary.json`.

### Ablations (Table 5)

```bash
for kind in source-layer temporal-window short-row dedup multilingual; do
    PYTHONPATH=src python3 -m etg.benchmark.serious_baselines \
        --benchmark-dir data/benchmark_v2 \
        --results-dir data/results/ablations \
        --mode ablation --ablation-kind $kind
done
```

### Reproducing precomputed results

`data/benchmark_v2/all_results.json` contains the exact numbers reported in Tables 2–4 (mean ± std across 5 seeds). To verify these without re-training:

```bash
python3 -c "
import json
with open('data/benchmark_v2/all_results.json') as f:
    r = json.load(f)
import pprint; pprint.pprint(r)
"
```

---

## 5. Regenerating figures

All figure scripts write to `paper/figures/`.

### Figure 2 — Temporal heatmap (requires full corpus JSONL or uses stats.json fallback)

```bash
# Fast path: uses precomputed stats (uniform distribution within source date range)
PYTHONPATH=src python3 scripts/plot_temporal_heatmap.py

# Accurate path: streams per-source JSONL files for exact year counts
# (requires downloading corpus JSONL files from HuggingFace to data/)
PYTHONPATH=src python3 scripts/plot_temporal_heatmap.py --data-dir data
```

### Figure 3 — Domain overlap heatmap (requires full corpus JSONL or cached matrix)

```bash
# If domain_overlap_cache.json exists (precomputed):
PYTHONPATH=src python3 scripts/plot_domain_overlap.py

# First run with full data builds and saves the cache:
PYTHONPATH=src python3 scripts/plot_domain_overlap.py --data-dir data
```

### Figure 4 — Baseline radar chart

```bash
# Default legend font size (9pt)
PYTHONPATH=src python3 scripts/plot_baseline_radar.py

# Larger legend
PYTHONPATH=src python3 scripts/plot_baseline_radar.py --legend-fontsize 11
```

This script reads Tasks 1 & 3 results from `data/benchmark_v2/all_results.json` and re-trains Task 2 classifiers from the committed test split. Allow ~10–15 minutes for the neural models.

### Supplemental — Corpus composition figures

```bash
PYTHONPATH=src python3 scripts/plot_corpus_stats.py
# Outputs: corpus_temporal_distribution.pdf/png, source_layer_composition.pdf/png
```

---

## 6. Regenerating the LaTeX paper

The paper source is `paper/HackerSignal_neurips2026.tex`. Figures are pre-built in `paper/figures/`. To compile:

```bash
cd paper
pdflatex HackerSignal_neurips2026.tex
bibtex HackerSignal_neurips2026
pdflatex HackerSignal_neurips2026.tex
pdflatex HackerSignal_neurips2026.tex
```

Or with `latexmk`:

```bash
cd paper && latexmk -pdf HackerSignal_neurips2026.tex
```

---

## 7. Collecting new data

All collection scripts are under `src/etg/collectors/`. Each collector is self-contained and writes a `{source}_posts.jsonl` + `{source}_cve_index.jsonl` pair to `data/`.

```bash
# Example: update NVD CVE records
PYTHONPATH=src python3 -m etg.collectors.nvd --output-dir data

# Example: scrape 0x00sec
PYTHONPATH=src python3 -m etg.collectors.discourse_communities \
    --community 0x00sec --output-dir data

# After collecting, rebuild the unified corpus:
PYTHONPATH=src python3 -m etg.data.unify_hacker_communities \
    --data-dir data --output data/unified_hacker_communities_neurips.jsonl

# Rebuild benchmark splits:
PYTHONPATH=src python3 -m etg.benchmark.build_benchmark_v2 \
    --corpus data/unified_hacker_communities_neurips.jsonl \
    --output-dir data/benchmark_v2
```

---

## 8. Expected runtimes

| Step | Hardware | Time |
|------|----------|------|
| Quick baseline run (lexical + dense, all tasks) | CPU (MacBook) | ~10 min |
| Full 5-seed overnight suite | M2 Pro / A100 | 3–8 hours |
| Radar chart (neural Task 2 models) | Apple MPS | ~15 min |
| Temporal heatmap (stats.json fallback) | CPU | < 30 sec |
| Domain overlap (from JSONL, 10 sources) | CPU | ~5 min |
| Paper compilation | — | < 1 min |

---

## Citation

```bibtex
@article{hackersignal2026,
  title   = {HackerSignal: A Temporally Structured, Multi-Source Benchmark
             for Cybersecurity Threat Intelligence},
  author  = {Anonymous},
  year    = {2026},
  note    = {NeurIPS 2026 Evaluations \& Datasets Track (under review)}
}
```

---

*Anonymous submission — author information redacted for double-blind review.*
