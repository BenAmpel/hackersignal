---
language:
- en
- zh
- ru
license: cc-by-4.0
size_categories:
- 1M<n<10M
task_categories:
- text-classification
- text-retrieval
tags:
- cybersecurity
- exploit
- vulnerability
- CVE
- hacker-forum
- threat-intelligence
- NLP
- benchmark
pretty_name: HackerSignal
dataset_info:
  splits:
  - name: full
    num_examples: 7447646
configs:
- config_name: default
  data_files:
  - split: full
    path: unified_hacker_communities_neurips_public.jsonl
- config_name: sample
  data_files:
  - split: full
    path: hackersignal_sample_10k.jsonl
- config_name: task1_cve_linkage
  data_files:
  - split: train
    path: benchmark_v2/task1_cve_linkage/train.jsonl
  - split: validation
    path: benchmark_v2/task1_cve_linkage/val.jsonl
  - split: test
    path: benchmark_v2/task1_cve_linkage/test.jsonl
  - split: corpus
    path: benchmark_v2/task1_cve_linkage/corpus.jsonl
- config_name: task2_exploit_type
  data_files:
  - split: train
    path: benchmark_v2/task2_exploit_type/train.jsonl
  - split: validation
    path: benchmark_v2/task2_exploit_type/val.jsonl
  - split: test
    path: benchmark_v2/task2_exploit_type/test.jsonl
- config_name: task3_temporal_generalization
  data_files:
  - split: train
    path: benchmark_v2/task3_temporal_generalization/train.jsonl
  - split: validation
    path: benchmark_v2/task3_temporal_generalization/val.jsonl
  - split: test
    path: benchmark_v2/task3_temporal_generalization/test.jsonl
  - split: corpus
    path: benchmark_v2/task3_temporal_generalization/corpus.jsonl
---

# HackerSignal

A large-scale, multi-source dataset linking hacker community discourse, exploit databases, vulnerability advisories, and fix commits through a shared CVE identifier space.

## Overview

| Statistic | Value |
|-----------|-------|
| Documents | 7,447,646 (exact-deduplicated) |
| Sources | 64 public forum/source identifiers |
| Source layers | 8 |
| Temporal span | 1988--2026 |
| CVE-linked rows | 360,004 |
| Benchmark tasks | 3 |

## Quick Start

```python
from datasets import load_dataset

# Load the 10K stratified sample (24 MB)
sample = load_dataset("DatasetSubmission/HackerSignal", "sample")

# Load a benchmark task
task1 = load_dataset("DatasetSubmission/HackerSignal", "task1_cve_linkage")

# Load the full corpus (12 GB)
full = load_dataset("DatasetSubmission/HackerSignal", "default")
```

## Source Layers

| Layer | Description | Rows |
|-------|-------------|------|
| `hacker_community` | Forum posts from hacker/security communities | 6,974,128 |
| `exploit_archive` | Published exploit code and advisories | 197,393 |
| `vulnerability_reference` | NVD CVE descriptions and structured advisories | 150,916 |
| `exploit_qa_reference` | Security Q&A and tutorial content | 68,791 |
| `advisory_reference` | Vendor and third-party security advisories | 30,361 |
| `bug_bounty_disclosure` | Public bug bounty reports | 13,720 |
| `fix_commit_reference` | Vulnerability fix commit messages | 11,104 |
| `exploitation_reference` | Active exploitation indicators (CISA KEV) | 1,233 |

## Benchmark Tasks

### Task 1: CVE Linkage Retrieval (CVE-R)
Cross-source temporally OOD entity grounding: given exploit/advisory evidence text, retrieve the correct NVD CVE entry from a corpus of 340K descriptions. Queries come from 22 source identifiers across exploit archives, advisories, fix commits, and bug bounty reports; filtered to ≥8 tokens.

| Split | Rows |
|-------|------|
| Train | 56,692 |
| Val | 2,584 |
| Test | 1,990 |
| Corpus | 340,536 |

### Task 2: Exploit Type Classification (ETC)
8-class temporal OOD classification of exploit posts by type: injection, XSS, memory corruption, DoS, file inclusion, authentication/access bypass, RCE, and information disclosure. Sourced from ExploitDB and HackerOne public reports; split by publication year so test posts postdate all training examples.

| Split | Rows |
|-------|------|
| Train | 64,413 |
| Val | 4,735 |
| Test | 1,735 |

### Task 3: Temporal Generalization (TG)
Identical retrieval formulation to Task 1 but with strict CVE-disjoint constraint: C_train ∩ C_test = ∅. Split by CVE publication year (train: <2022, test: 2024+) to test generalization to wholly unseen vulnerabilities.

| Split | Rows |
|-------|------|
| Train | 56,833 |
| Val | 2,535 |
| Test | 1,898 |
| Corpus | 340,536 |

## Schema

Each record in the main corpus contains:

| Field | Type | Description |
|-------|------|-------------|
| `unified_id` | string | SHA-256-derived release identifier |
| `source_dataset` | string | Input source file or dataset name |
| `source_layer` | string | Normalized layer (see above) |
| `text` | string | UTF-8 post/advisory text (max 8000 chars) |
| `timestamp` | string | ISO 8601 publication datetime (UTC) |
| `forum_id` | string | Source identifier |
| `author_hash` | string | SHA-256 pseudonymized author |
| `release_mode` | string | Governance mode (see below) |

## Release Governance

Records follow source-specific release modes:

- **`redistributable_text`** (5 sources): Full text included under public-domain or CC licenses.
- **`research_text_with_terms`** (9 sources): Text included with source attribution; no commercial reuse.
- **`metadata_or_pointer_only`** (13 sources): Only metadata, timestamps, CVE refs, text hashes, and lengths; raw text withheld.

See `docs/release_governance.md` for the full source matrix, takedown policy, and responsible-use restrictions.

## Responsible Use

This dataset is intended for **defensive cybersecurity research** only. Prohibited uses include:
- Training systems for automated exploit code generation or malware
- De-anonymizing forum participants outside approved research protocols
- Operational blocking or law-enforcement decisions based solely on model outputs
- Republishing text from `metadata_or_pointer_only` sources

## Known Issues

- **HuggingFace dataset viewer row count**: The viewer may display a partial count (~1M) for the `default/full` split because HF's streaming viewer cannot fully index files over ~10 GB. The actual row count is 7,447,646 as validated against the SHA-256 manifest.
- **Content-scanner warnings**: HF's automated scanner flags several files (including the sample JSONL) as "Unsafe" due to exploit and vulnerability text. This is expected for a cybersecurity research dataset and does not indicate malicious content. See the Responsible Use section below.

## Limitations

- English-language bias; non-English communities may be underrepresented
- Historical forum imports skew toward 2015--2017
- Task labels are weak (CVE-linkage-derived), not human-annotated
- ~8% near-duplicate content remains after exact deduplication
- 1.57M rows under 8 tokens (useful for interaction analysis, not classification)

## Citation

```bibtex
@misc{hackersignal2026,
  title={{HackerSignal}: A Temporally Structured, Multi-Source Benchmark for Cybersecurity Threat Intelligence},
  author={Anonymous},
  year={2026},
  howpublished={arXiv preprint}
}
```

## License

Code and metadata: CC BY 4.0. Source text retains source-specific terms documented in the governance addendum and release manifest.
