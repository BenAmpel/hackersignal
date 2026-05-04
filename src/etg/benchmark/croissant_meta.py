"""Generate a Croissant machine-readable metadata file for the HackerSignal dataset.

Croissant is the ML dataset metadata standard required by NeurIPS 2025 D&B.
Spec: https://github.com/mlcommons/croissant

This script produces ``etg_croissant.json`` in JSON-LD format, which can be
validated at https://huggingface.co/spaces/MLCommons/croissant-editor or via
the mlcroissant Python library.

Usage
-----
    python -m etg.benchmark.croissant_meta --output etg_croissant.json
"""

from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Croissant record fields
# ---------------------------------------------------------------------------

_RECORD_SETS = [
    {
        "@type": "cr:RecordSet",
        "@id": "posts",
        "name": "posts",
        "description": (
            "Forum posts, exploit advisories, vulnerability descriptions, and fix-commit "
            "messages from 64 public forum/source identifiers. Each record is one document."
        ),
        "field": [
            {
                "@type": "cr:Field",
                "@id": "posts/unified_id",
                "name": "unified_id",
                "description": "SHA-256-derived release identifier.",
                "dataType": "sc:Text",
            },
            {
                "@type": "cr:Field",
                "@id": "posts/source_dataset",
                "name": "source_dataset",
                "description": "Input source file or dataset name.",
                "dataType": "sc:Text",
            },
            {
                "@type": "cr:Field",
                "@id": "posts/source_layer",
                "name": "source_layer",
                "description": "Normalized layer, such as hacker_community or vulnerability_reference.",
                "dataType": "sc:Text",
            },
            {
                "@type": "cr:Field",
                "@id": "posts/text",
                "name": "text",
                "description": "Full text of the post or advisory (UTF-8, max 8 000 chars).",
                "dataType": "sc:Text",
            },
            {
                "@type": "cr:Field",
                "@id": "posts/timestamp",
                "name": "timestamp",
                "description": "Publication date-time (ISO 8601, UTC).",
                "dataType": "sc:DateTime",
            },
            {
                "@type": "cr:Field",
                "@id": "posts/forum_id",
                "name": "forum_id",
                "description": "Source identifier (e.g. 'exploitdb', 'nvd', 'hackforums').",
                "dataType": "sc:Text",
            },
            {
                "@type": "cr:Field",
                "@id": "posts/author_hash",
                "name": "author_hash",
                "description": "SHA-256 hash of the source-namespaced author string (pseudonymised).",
                "dataType": "sc:Text",
            },
            {
                "@type": "cr:Field",
                "@id": "posts/release_mode",
                "name": "release_mode",
                "description": "Release-governance mode: redistributable_text, research_text_with_terms, metadata_or_pointer_only, or excluded.",
                "dataType": "sc:Text",
            },
        ],
    },
    {
        "@type": "cr:RecordSet",
        "@id": "cve_index",
        "name": "cve_index",
        "description": (
            "Cross-source CVE linkage index mapping post IDs to CVE identifiers. "
            "One row per (post, CVE) pair."
        ),
        "field": [
            {
                "@type": "cr:Field",
                "@id": "cve_index/exploit_id",
                "name": "exploit_id",
                "description": "Foreign key to posts/id.",
                "dataType": "sc:Text",
            },
            {
                "@type": "cr:Field",
                "@id": "cve_index/cve_id",
                "name": "cve_id",
                "description": "CVE identifier in the form CVE-YYYY-NNNNN.",
                "dataType": "sc:Text",
            },
            {
                "@type": "cr:Field",
                "@id": "cve_index/exploit_text",
                "name": "exploit_text",
                "description": "First 500 characters of the source post text.",
                "dataType": "sc:Text",
            },
            {
                "@type": "cr:Field",
                "@id": "cve_index/title",
                "name": "title",
                "description": "Advisory or post title.",
                "dataType": "sc:Text",
            },
            {
                "@type": "cr:Field",
                "@id": "cve_index/published",
                "name": "published",
                "description": "Publication date-time (ISO 8601, UTC).",
                "dataType": "sc:DateTime",
            },
            {
                "@type": "cr:Field",
                "@id": "cve_index/source",
                "name": "source",
                "description": "Source identifier matching posts/forum_id.",
                "dataType": "sc:Text",
            },
        ],
    },
    # --- Benchmark splits ---
    {
        "@type": "cr:RecordSet",
        "@id": "task1_exploit_clf",
        "name": "task1_exploit_clf",
        "description": (
            "Task 1 — Exploit Relevance Classification. "
            "Binary classification: label=1 if exploit-relevant, label=0 otherwise. "
            "Split into train / val / test by publication date (pre-2022 / 2022–2023 / 2024+)."
        ),
        "field": [
            {"@type": "cr:Field", "@id": "task1/id",        "name": "id",        "dataType": "sc:Text"},
            {"@type": "cr:Field", "@id": "task1/text",      "name": "text",      "dataType": "sc:Text"},
            {"@type": "cr:Field", "@id": "task1/label",     "name": "label",     "dataType": "sc:Integer",
             "description": "0 = not exploit-relevant, 1 = exploit-relevant"},
            {"@type": "cr:Field", "@id": "task1/source",    "name": "source",    "dataType": "sc:Text"},
            {"@type": "cr:Field", "@id": "task1/timestamp", "name": "timestamp", "dataType": "sc:DateTime"},
            {"@type": "cr:Field", "@id": "task1/split",     "name": "split",     "dataType": "sc:Text"},
        ],
    },
    {
        "@type": "cr:RecordSet",
        "@id": "task2_cve_linkage",
        "name": "task2_cve_linkage",
        "description": (
            "Task 2 — Quality-Controlled CVE Linkage Retrieval. "
            "Given source evidence text, retrieve the metadata-linked NVD CVE entry from a corpus of ~340 K CVE descriptions. "
            "The split filters unretrievable empty/high-risk rows and is evaluated as a ranking problem (Recall@K, MRR)."
        ),
        "field": [
            {"@type": "cr:Field", "@id": "task2/id",        "name": "id",        "dataType": "sc:Text"},
            {"@type": "cr:Field", "@id": "task2/text",      "name": "text",      "dataType": "sc:Text"},
            {"@type": "cr:Field", "@id": "task2/cve_id",    "name": "cve_id",    "dataType": "sc:Text"},
            {"@type": "cr:Field", "@id": "task2/source",    "name": "source",    "dataType": "sc:Text"},
            {"@type": "cr:Field", "@id": "task2/timestamp", "name": "timestamp", "dataType": "sc:DateTime"},
            {"@type": "cr:Field", "@id": "task2/split",     "name": "split",     "dataType": "sc:Text"},
        ],
    },
    {
        "@type": "cr:RecordSet",
        "@id": "task3_severity",
        "name": "task3_severity",
        "description": (
            "Task 3 — Severity Prediction. "
            "4-class classification: predict CVSS severity bucket "
            "(0=low, 1=medium, 2=high, 3=critical) from advisory text."
        ),
        "field": [
            {"@type": "cr:Field", "@id": "task3/id",             "name": "id",             "dataType": "sc:Text"},
            {"@type": "cr:Field", "@id": "task3/text",           "name": "text",           "dataType": "sc:Text"},
            {"@type": "cr:Field", "@id": "task3/text_raw",       "name": "text_raw",       "dataType": "sc:Text",
             "description": "Original text before explicit severity/CVSS leakage strings were stripped from benchmark input."},
            {"@type": "cr:Field", "@id": "task3/severity",       "name": "severity",       "dataType": "sc:Text",
             "description": "Severity label string: low / medium / high / critical"},
            {"@type": "cr:Field", "@id": "task3/severity_label", "name": "severity_label", "dataType": "sc:Integer",
             "description": "Severity as integer: 0=low 1=medium 2=high 3=critical"},
            {"@type": "cr:Field", "@id": "task3/source",         "name": "source",         "dataType": "sc:Text"},
            {"@type": "cr:Field", "@id": "task3/timestamp",      "name": "timestamp",      "dataType": "sc:DateTime"},
            {"@type": "cr:Field", "@id": "task3/split",          "name": "split",          "dataType": "sc:Text"},
        ],
    },
    {
        "@type": "cr:RecordSet",
        "@id": "task4_hacker_exploit_labeling",
        "name": "task4_hacker_exploit_labeling",
        "description": (
            "Task 4 — Hacker Exploit Labeling. "
            "Ternary weak-supervision task over hacker-community messages: "
            "0=non-exploit/noise, 1=vulnerability discussion, 2=actionable exploit intelligence."
        ),
        "field": [
            {"@type": "cr:Field", "@id": "task4/id",         "name": "id",         "dataType": "sc:Text"},
            {"@type": "cr:Field", "@id": "task4/text",       "name": "text",       "dataType": "sc:Text"},
            {"@type": "cr:Field", "@id": "task4/label",      "name": "label",      "dataType": "sc:Integer",
             "description": "0=non_exploit_noise, 1=vulnerability_discussion, 2=actionable_exploit"},
            {"@type": "cr:Field", "@id": "task4/label_name", "name": "label_name", "dataType": "sc:Text"},
            {"@type": "cr:Field", "@id": "task4/cve_ids",    "name": "cve_ids",    "dataType": "sc:Text"},
            {"@type": "cr:Field", "@id": "task4/source",     "name": "source",     "dataType": "sc:Text"},
            {"@type": "cr:Field", "@id": "task4/timestamp",  "name": "timestamp",  "dataType": "sc:DateTime"},
            {"@type": "cr:Field", "@id": "task4/split",      "name": "split",      "dataType": "sc:Text"},
        ],
    },
    {
        "@type": "cr:RecordSet",
        "@id": "task5_hacker_signal_detection",
        "name": "task5_hacker_signal_detection",
        "description": (
            "Task 5 — Hacker Exploit Signal Detection. "
            "Joint task: detect actionable exploit intelligence in hacker-community messages "
            "and retrieve the associated NVD CVE context when applicable."
        ),
        "field": [
            {"@type": "cr:Field", "@id": "task5/id",               "name": "id",               "dataType": "sc:Text"},
            {"@type": "cr:Field", "@id": "task5/text",             "name": "text",             "dataType": "sc:Text"},
            {"@type": "cr:Field", "@id": "task5/actionable_label", "name": "actionable_label", "dataType": "sc:Integer",
             "description": "0=not_actionable_exploit_signal, 1=actionable_exploit_signal"},
            {"@type": "cr:Field", "@id": "task5/cve_id",           "name": "cve_id",           "dataType": "sc:Text",
             "description": "Primary CVE target for actionable records; empty for non-actionable records."},
            {"@type": "cr:Field", "@id": "task5/cve_ids",          "name": "cve_ids",          "dataType": "sc:Text"},
            {"@type": "cr:Field", "@id": "task5/source",           "name": "source",           "dataType": "sc:Text"},
            {"@type": "cr:Field", "@id": "task5/timestamp",        "name": "timestamp",        "dataType": "sc:DateTime"},
            {"@type": "cr:Field", "@id": "task5/split",            "name": "split",            "dataType": "sc:Text"},
        ],
    },
]


def build_croissant(
    hf_repo: str = "ben-ampel/etg-exploit-text-graph",
    version: str = "1.0.0",
    license_url: str = "https://creativecommons.org/licenses/by/4.0/",
) -> dict:
    """Return the Croissant JSON-LD metadata dict."""
    return {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "sc": "https://schema.org/",
            "dct": "http://purl.org/dc/terms/",
            "rai": "http://mlcommons.org/croissant/RAI/",
        },
        "@type": "sc:Dataset",
        "name": "HackerSignal",
        "description": (
            "HackerSignal is a large-scale, multi-source dataset linking "
            "hacker community discourse, exploit databases, vulnerability advisories, and "
            "fix commits through a shared CVE identifier space. It spans 64 public "
            "forum/source identifiers across eight source layers, covers 1988–2026, and "
            "supports five benchmark tasks: exploit relevance classification, CVE linkage "
            "retrieval, leakage-sanitized severity prediction, hacker exploit labeling, "
            "and hacker exploit signal detection."
        ),
        "url": f"https://huggingface.co/datasets/{hf_repo}",
        "version": version,
        "license": license_url,
        "usageInfo": (
            "Code and metadata are CC BY 4.0 unless otherwise noted. Source text follows "
            "source-specific release modes documented in docs/release_governance.md; "
            "terms-ambiguous sources may be metadata_or_pointer_only."
        ),
        "isAccessibleForFree": True,
        "creator": [
            {
                "@type": "sc:Person",
                "name": "Benjamin Ampel",
                "email": "ben.ampel@gmail.com",
            }
        ],
        "keywords": [
            "cybersecurity",
            "exploit",
            "vulnerability",
            "CVE",
            "hacker forum",
            "threat intelligence",
            "NLP",
            "information retrieval",
            "text classification",
        ],
        "datePublished": "2026",
        "inLanguage": ["en", "zh", "ru"],
        "citation": (
            "Ampel, B. (2026). HackerSignal — A Multi-Source Dataset "
            "for Hacker Community Analysis and CVE Linkage. "
            "Proceedings of NeurIPS Datasets & Benchmarks Track."
        ),
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": "hf-repo",
                "name": "HuggingFace Repository",
                "contentUrl": f"https://huggingface.co/datasets/{hf_repo}",
                "encodingFormat": "application/jsonlines",
                "sha256": "See versioned manifest checksums in the HuggingFace repository.",
            }
        ],
        "recordSet": _RECORD_SETS,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate HackerSignal Croissant metadata JSON-LD")
    parser.add_argument("--output", default="etg_croissant.json")
    parser.add_argument("--hf-repo", default="ben-ampel/etg-exploit-text-graph")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--license", default="https://creativecommons.org/licenses/by/4.0/")
    args = parser.parse_args()

    doc = build_croissant(
        hf_repo=args.hf_repo,
        version=args.version,
        license_url=args.license,
    )

    out = Path(args.output)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Croissant metadata written to: {out}")
    print(f"Validate at: https://huggingface.co/spaces/MLCommons/croissant-editor")


if __name__ == "__main__":
    main()
