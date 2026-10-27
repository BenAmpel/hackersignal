<div align="center">

# HackerSignal

**A Large-Scale Multi-Source Dataset Linking Hacker Community Discourse to the CVE Vulnerability Lifecycle**

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS%202026-Evaluations%20%26%20Datasets%20Track-8A2BE2)](https://neurips.cc/Conferences/2026)
[![arXiv](https://img.shields.io/badge/arXiv-2605.03158-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.03158)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-HackerSignal-FFD21E)](https://huggingface.co/datasets/DatasetSubmission/HackerSignal)
[![Croissant](https://img.shields.io/badge/metadata-Croissant%20%F0%9F%A5%90-orange)](https://huggingface.co/datasets/DatasetSubmission/HackerSignal/blob/main/etg_croissant.json)
[![Datasheet](https://img.shields.io/badge/docs-Datasheet%20for%20Datasets-blue)](docs/datasheet.md)
[![Data license: CC BY 4.0](https://img.shields.io/badge/data%20license-CC%20BY%204.0-lightgrey)](LICENSE-DATA)
[![Code license: MIT](https://img.shields.io/badge/code%20license-MIT-green)](LICENSE)
<br>
[![Documents](https://img.shields.io/badge/documents-7.45M-success)](#at-a-glance)
[![Sources](https://img.shields.io/badge/sources-64-success)](#at-a-glance)
[![CVEs](https://img.shields.io/badge/NVD%20CVEs-340%2C536-success)](#at-a-glance)
[![Tasks](https://img.shields.io/badge/benchmark%20tasks-3-success)](#benchmark-tasks)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![GitHub stars](https://img.shields.io/github/stars/BenAmpel/hackersignal?style=social)](https://github.com/BenAmpel/hackersignal)
[![Last commit](https://img.shields.io/github/last-commit/BenAmpel/hackersignal)](https://github.com/BenAmpel/hackersignal/commits/main)

[Paper (arXiv)](https://arxiv.org/abs/2605.03158) ·
[Dataset](https://huggingface.co/datasets/DatasetSubmission/HackerSignal) ·
[Benchmark tasks](#benchmark-tasks) ·
[Quick start](#quick-start) ·
[Reproduce](#reproducing-the-baselines) ·
[Citation](#citation)

</div>

---

**HackerSignal** is a cyber threat intelligence (CTI) dataset and benchmark that links what hackers discuss in
underground and security communities to the vulnerabilities those discussions concern. It aggregates
**7,447,646 exact-deduplicated documents from 64 public forum and source identifiers** across eight source
layers, from hacker forums and exploit archives to NVD CVE descriptions, security advisories, bug-bounty reports,
fix commits and CISA Known Exploited Vulnerabilities. On top of the corpus it defines **three benchmark tasks with
temporal train/validation/test splits**: CVE linkage retrieval, 8-class exploit type classification, and
CVE-disjoint temporal generalization to vulnerabilities published after training.

HackerSignal is published in the **NeurIPS 2026 Evaluations & Datasets Track**. Preprint:
[arXiv:2605.03158](https://arxiv.org/abs/2605.03158).

## At a glance

| | |
|---|---|
| Documents | 7,447,646 (exact-deduplicated) |
| Sources | 64 public forum / source identifiers in 8 source layers |
| Hacker-community layer | 6,966,014 posts (largest forums: Gayanku, Antichat, Evolution, AntiOnline, HackForums) |
| Vulnerability reference | 340,536 NVD CVE descriptions (the retrieval corpus for Tasks 1 and 3) |
| Other layers | Exploit archives, exploit Q&A, advisories, bug-bounty disclosures, fix commits, CISA KEV |
| Languages | Mostly English, with Russian and Chinese forum content |
| Splits | Temporal: test data postdate training data in every task |
| Metadata | Croissant JSON-LD, Datasheet for Datasets, source release manifest |
| License | Data CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA)), code MIT ([LICENSE](LICENSE)); per-source terms in [release governance](docs/release_governance.md) |

## Benchmark tasks

| Task | Type | Train | Val | Test | Corpus / total | Metrics |
|---|---|---:|---:|---:|---:|---|
| 1. CVE Linkage Retrieval (CVE-R) | Retrieval | 56,692 | 2,584 | 1,990 | 340,536 | R@1, R@5, R@10, MRR |
| 2. Exploit Type Classification (ETC) | 8-class classification | 64,413 | 4,735 | 1,735 | 70,883 | Macro-F1, per-class F1 |
| 3. Temporal Generalization (TG) | CVE-disjoint retrieval | 56,833 | 2,535 | 1,898 | 340,536 | R@1, R@5, R@10, MRR |

- **Task 1 (CVE-R):** given exploit, advisory, fix-commit or bug-bounty text, retrieve the CVE it refers to from all
  340,536 NVD descriptions. Gold links come from source metadata.
- **Task 2 (ETC):** classify exploit text into injection, XSS, memory corruption, denial of service, file inclusion,
  authentication and access control, remote code execution, or information disclosure. Train < 2020,
  validation 2020–2022, test ≥ 2023.
- **Task 3 (TG):** the Task 1 retrieval problem with strictly disjoint gold CVE sets: training CVEs were published
  before 2022, validation CVEs in 2022–2023, and test CVEs in 2024 or later. The Task 3 test queries are a subset
  of the Task 1 test queries.

The baseline ladder covers BM25, four dense bi-encoders (all-MiniLM-L6-v2, all-mpnet-base-v2, BGE-base-en-v1.5,
E5-base-v2) and a BM25 + dense hybrid for retrieval, and Decision Tree, TF-IDF + logistic regression, linear SVM,
RNN, GRU, LSTM, BiLSTM and SecBERT for classification. Baseline scores are reported in the paper (Tables 4–6).

## Quick start

The test splits are committed in `data/benchmark_v2/`. Training and validation splits and the full corpus are on
the Hugging Face Hub as JSONL files:

```python
from datasets import load_dataset

base = "hf://datasets/DatasetSubmission/HackerSignal/benchmark_v2"
task1 = load_dataset("json", data_files={
    "train": f"{base}/task1_cve_linkage/train.jsonl",
    "validation": f"{base}/task1_cve_linkage/val.jsonl",
    "test": f"{base}/task1_cve_linkage/test.jsonl",
})
corpus = load_dataset("json", data_files=f"{base}/task1_cve_linkage/corpus.jsonl", split="train")

print(task1["test"][0]["text"][:200], "->", task1["test"][0]["cve_id"])
```

Replace `task1_cve_linkage` with `task2_exploit_type` (fields `text`, `label`) or
`task3_temporal_generalization` (fields `text`, `cve_id`) for the other tasks.

## Reproducing the baselines

```bash
git clone https://github.com/BenAmpel/hackersignal.git
cd hackersignal
pip install -e ".[sentence-transformers]"
```

Place the train, validation and corpus files from the Hub under `data/benchmark_v2/<task>/`, then run:

```bash
# all three tasks; --task 1|2|3 runs one task, --skip-dense / --skip-neural shorten the run
PYTHONPATH=src python3 -m etg.benchmark.run_baselines_v2 \
    --data-dir data/benchmark_v2 \
    --output data/benchmark_v2/results.json
```

Neural models use Apple MPS when it is available and fall back to CPU. Dense encoding of the 340K-document corpus
takes from a few minutes (MiniLM) to tens of minutes (base-sized encoders) per model on a recent GPU or Apple
Silicon machine.

## Repository layout

```
hackersignal/
├── src/etg/
│   ├── collectors/        # data collection scripts, one per source
│   ├── data/              # corpus unification, preprocessing and statistics
│   ├── benchmark/         # split construction and baseline runners
│   ├── eval/              # retrieval metrics
│   └── scraper/           # generic forum crawlers
├── scripts/               # task builders and figure scripts
├── data/
│   ├── benchmark_v2/      # committed test splits and task metadata
│   ├── stats/             # corpus statistics
│   ├── audits/            # manual-audit packets and LLM-judge artifacts
│   └── release/           # source release manifest and validation
└── docs/
    ├── datasheet.md
    └── release_governance.md
```

## Collecting and rebuilding the data

Each collector in `src/etg/collectors/` writes a `{source}_posts.jsonl` and `{source}_cve_index.jsonl` pair to
`data/`:

```bash
PYTHONPATH=src python3 -m etg.collectors.nvd --output-dir data
PYTHONPATH=src python3 -m etg.data.unify_hacker_communities \
    --data-dir data --output data/unified_hacker_communities_neurips.jsonl
PYTHONPATH=src python3 -m etg.benchmark.build_benchmark_v2 \
    --corpus data/unified_hacker_communities_neurips.jsonl --output-dir data/benchmark_v2
```

## Responsible use

HackerSignal is released for academic and defensive cybersecurity research. Author identifiers are pseudonymized
with SHA-256, provenance is preserved for every record, and sources are released under one of four modes
(redistributable, research-only, metadata/pointer, excluded). Do not use the data to fine-tune automated exploit
generation, to de-anonymize forum users, or as an operational threat feed without human review. Takedown and
correction requests are handled as described in [docs/release_governance.md](docs/release_governance.md).

## FAQ

**What is HackerSignal?** A 7.45M-document cyber threat intelligence corpus from 64 public sources, with a
three-task benchmark for linking hacker, exploit and advisory text to CVE identifiers under temporal distribution
shift.

**Where do I download it?** From the Hugging Face Hub at
[DatasetSubmission/HackerSignal](https://huggingface.co/datasets/DatasetSubmission/HackerSignal). The test splits
are also in this repository.

**How are the labels made?** From platform metadata, not manual annotation: source-provided CVE links for Tasks 1
and 3, and ExploitDB, 0day.today and HackerOne type fields mapped to eight classes by deterministic rules for
Task 2. A 240-row stratified audit packet is in `data/audits/`.

**How is leakage prevented?** Every task uses a temporal split, and Task 3 gold CVE sets are disjoint across
train, validation and test.

## Citation

If you use HackerSignal, please cite:

```bibtex
@inproceedings{ampel2026hackersignal,
  title     = {{HackerSignal}: A Large-Scale Multi-Source Dataset Linking Hacker Community Discourse
               to the {CVE} Vulnerability Lifecycle},
  author    = {Ampel, Benjamin M. and Samtani, Sagar},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS),
               Evaluations and Datasets Track},
  year      = {2026},
  eprint    = {2605.03158},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  url       = {https://arxiv.org/abs/2605.03158}
}
```

GitHub's **Cite this repository** button reads the same entry from [`CITATION.cff`](CITATION.cff).

## License

The benchmark data and documentation are released under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-DATA) for the sources whose terms allow
redistribution; research-only and pointer-only sources follow the terms listed in
[docs/release_governance.md](docs/release_governance.md). The code is released under the [MIT License](LICENSE).

## Contact

Benjamin M. Ampel (Georgia State University, bampel@gsu.edu) and Sagar Samtani (Indiana University,
ssamtani@iu.edu). Please open an issue for bugs, data errors or takedown requests.
