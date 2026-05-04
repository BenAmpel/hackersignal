"""Build a machine-readable ETG source release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_POLICIES: dict[str, dict[str, str]] = {
    "nvd_posts": {"access": "NVD API/export", "license": "US government public domain", "release_mode": "redistributable_text", "artifact": "text"},
    "cisa_kev_posts": {"access": "CISA JSON feed", "license": "US government public domain", "release_mode": "redistributable_text", "artifact": "text"},
    "github_advisory_posts": {"access": "GitHub Advisory GraphQL/API", "license": "CC BY 4.0 at source", "release_mode": "redistributable_text", "artifact": "text"},
    "exploitdb_posts": {"access": "ExploitDB CSV/repository export", "license": "CC BY-SA 4.0 at source", "release_mode": "redistributable_text", "artifact": "text"},
    "exploitdb_hf_posts": {"access": "HuggingFace dataset import", "license": "upstream dataset terms", "release_mode": "research_text_with_terms", "artifact": "research_text"},
    "cvefixes_posts": {"access": "Zenodo SQLite", "license": "CC BY 4.0 at source", "release_mode": "redistributable_text", "artifact": "text"},
    "hackerone_posts": {"access": "Public HuggingFace dataset", "license": "upstream public dataset terms", "release_mode": "research_text_with_terms", "artifact": "research_text"},
    "dtl_exploits_posts": {"access": "Public research artifact", "license": "research-only/source-specific", "release_mode": "metadata_or_pointer_only", "artifact": "pointer"},
    "deepdarkcti_public_broad_2026-04-18_raw": {"access": "GitHub public dataset", "license": "repository terms", "release_mode": "research_text_with_terms", "artifact": "research_text"},
    "0x00sec_posts": {"access": "Public Discourse API/pages", "license": "site terms", "release_mode": "metadata_or_pointer_only", "artifact": "pointer"},
    "hackersploit_posts": {"access": "Public Discourse API/pages", "license": "site terms", "release_mode": "metadata_or_pointer_only", "artifact": "pointer"},
    "parrotsec_posts": {"access": "Public Discourse API/pages", "license": "site terms", "release_mode": "metadata_or_pointer_only", "artifact": "pointer"},
    "hackthebox_posts": {"access": "Public Discourse API/pages", "license": "site terms", "release_mode": "metadata_or_pointer_only", "artifact": "pointer"},
    "go4expert_posts": {"access": "Public pages/archive", "license": "mixed source terms", "release_mode": "metadata_or_pointer_only", "artifact": "pointer"},
    "fulldisclosure_posts": {"access": "Public mailing-list archive", "license": "mixed source terms", "release_mode": "metadata_or_pointer_only", "artifact": "pointer"},
    "seebug_posts": {"access": "Public pages/API", "license": "mixed source terms", "release_mode": "metadata_or_pointer_only", "artifact": "pointer"},
    "vulnlab_posts": {"access": "Public pages/archive", "license": "mixed source terms", "release_mode": "metadata_or_pointer_only", "artifact": "pointer"},
    "zeroscience_posts": {"access": "Public pages/archive", "license": "mixed source terms", "release_mode": "metadata_or_pointer_only", "artifact": "pointer"},
    "zeroday_posts": {"access": "Public import/archive", "license": "mixed source terms", "release_mode": "metadata_or_pointer_only", "artifact": "pointer"},
    "public_exploits_posts": {"access": "Public import/archive", "license": "mixed source terms", "release_mode": "metadata_or_pointer_only", "artifact": "pointer"},
    "gayanku_posts": {"access": "Kaggle dataset import", "license": "dataset-specific Kaggle terms", "release_mode": "research_text_with_terms", "artifact": "research_text"},
    "hackforums_posts": {"access": "Kaggle dataset import", "license": "dataset-specific Kaggle terms", "release_mode": "research_text_with_terms", "artifact": "research_text"},
    "kaeli_hacker_posts": {"access": "Kaggle dataset import", "license": "dataset-specific Kaggle terms", "release_mode": "research_text_with_terms", "artifact": "research_text"},
    "crackingarena_posts": {"access": "Kaggle/public dataset import", "license": "dataset-specific terms", "release_mode": "research_text_with_terms", "artifact": "research_text"},
    "evolution_posts": {"access": "Public dataset import", "license": "dataset-specific terms", "release_mode": "research_text_with_terms", "artifact": "research_text"},
    "hacker_exploits_posts": {"access": "Public dataset import", "license": "dataset-specific terms", "release_mode": "research_text_with_terms", "artifact": "research_text"},
    "antionline_posts": {"access": "Public dataset/page import", "license": "mixed source terms", "release_mode": "metadata_or_pointer_only", "artifact": "pointer"},
    "cve_hacker_forum_posts": {"access": "Public research/dataset import", "license": "source-specific terms", "release_mode": "metadata_or_pointer_only", "artifact": "pointer"},
}


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or path.is_dir():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    input_manifest: Path = Path("data/unified_hacker_communities_neurips_manifest.json"),
    output: Path = Path("data/release/source_release_manifest.json"),
    dataset_version: str = "1.0.0-rc1",
    doi: str = "TBD after immutable archive deposit",
) -> dict[str, Any]:
    src = json.loads(input_manifest.read_text(encoding="utf-8")) if input_manifest.exists() else {}
    source_rows = src.get("source_written_rows") or src.get("source_rows") or {}
    rows = []
    for source, count in sorted(source_rows.items()):
        policy = SOURCE_POLICIES.get(
            source,
            {
                "access": "source-specific import",
                "license": "unknown or mixed source terms",
                "release_mode": "metadata_or_pointer_only",
                "artifact": "pointer",
            },
        )
        source_path = Path("data") / f"{source}.jsonl"
        rows.append(
            {
                "source": source,
                "row_count": int(count),
                "access_mechanism": policy["access"],
                "license_or_terms": policy["license"],
                "release_mode": policy["release_mode"],
                "released_artifact": policy["artifact"],
                "sha256": _sha256_file(source_path),
                "source_file": str(source_path) if source_path.exists() else None,
                "takedown_status": "eligible_for_downgrade_or_removal_on_request",
            }
        )
    governed_release = Path("data/unified_hacker_communities_neurips_public.jsonl")
    release_file = governed_release if governed_release.exists() else Path(src.get("output_path", "data/unified_hacker_communities_neurips.jsonl"))
    manifest = {
        "dataset": "ETG: Exploit Text Graph",
        "version": dataset_version,
        "doi": doi,
        "maintainer": "Benjamin Ampel <ben.ampel@gmail.com>",
        "issue_tracker": "GitHub issues for the accompanying repository",
        "source_manifest": str(input_manifest),
        "release_file": str(release_file),
        "release_file_sha256": _sha256_file(release_file),
        "croissant": "data/etg_croissant.json",
        "datasheet": "docs/datasheet.md",
        "governance": "docs/release_governance.md",
        "sources": rows,
        "release_mode_counts": dict(sorted({m: sum(1 for r in rows if r["release_mode"] == m) for m in {r["release_mode"] for r in rows}}.items())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(manifest, output.with_suffix(".md"))
    return manifest


def build_governed_release(
    input_jsonl: Path = Path("data/unified_hacker_communities_neurips.jsonl"),
    manifest_path: Path = Path("data/release/source_release_manifest.json"),
    output_jsonl: Path = Path("data/unified_hacker_communities_neurips_public.jsonl"),
) -> dict[str, Any]:
    """Write a public release JSONL that enforces source release modes.

    Pointer-only sources retain provenance, timestamps, thread/source IDs, CVE
    references, and text fingerprints, but raw text fields are withheld.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policies = {row["source"]: row for row in manifest["sources"]}
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    redacted = 0
    written = 0
    with input_jsonl.open(encoding="utf-8", errors="replace") as src, output_jsonl.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("source_dataset") or row.get("source")
            policy = policies.get(source, {})
            release_mode = policy.get("release_mode", "metadata_or_pointer_only")
            artifact = policy.get("released_artifact", "pointer")
            row["release_mode"] = release_mode
            row["released_artifact"] = artifact
            text = row.get("text") or ""
            raw = row.get("text_raw") or ""
            if release_mode == "metadata_or_pointer_only" or artifact == "pointer":
                row["text_sha256"] = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest() if text else ""
                row["text_length"] = len(text)
                row["text_raw_sha256"] = hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest() if raw else ""
                row["text_raw_length"] = len(raw)
                row["text"] = None
                row["text_raw"] = None
                row["redaction_reason"] = "metadata_or_pointer_only_release_mode"
                redacted += 1
            counts[release_mode] = counts.get(release_mode, 0) + 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    summary = {
        "input": str(input_jsonl),
        "output": str(output_jsonl),
        "manifest": str(manifest_path),
        "rows_written": written,
        "redacted_rows": redacted,
        "release_mode_row_counts": dict(sorted(counts.items())),
        "output_sha256": _sha256_file(output_jsonl),
    }
    summary_path = output_jsonl.with_suffix(".manifest.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def validate_governed_release(
    release_jsonl: Path = Path("data/unified_hacker_communities_neurips_public.jsonl"),
    manifest_path: Path = Path("data/release/source_release_manifest.json"),
    output: Path = Path("data/release/governed_release_validation.json"),
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pointer_sources = {
        row["source"]
        for row in manifest["sources"]
        if row.get("release_mode") == "metadata_or_pointer_only" or row.get("released_artifact") == "pointer"
    }
    violations: list[dict[str, Any]] = []
    rows = 0
    pointer_rows = 0
    with release_jsonl.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            source = row.get("source_dataset") or row.get("source")
            if source in pointer_sources:
                pointer_rows += 1
                if row.get("text") or row.get("text_raw"):
                    violations.append(
                        {
                            "row": rows,
                            "source": source,
                            "unified_id": row.get("unified_id"),
                            "has_text": bool(row.get("text")),
                            "has_text_raw": bool(row.get("text_raw")),
                        }
                    )
                    if len(violations) >= 25:
                        break
    report = {
        "release": str(release_jsonl),
        "manifest": str(manifest_path),
        "rows_checked": rows,
        "pointer_rows_checked": pointer_rows,
        "violations": violations,
        "passed": not violations,
        "release_sha256": _sha256_file(release_jsonl),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def write_markdown(manifest: dict[str, Any], path: Path) -> None:
    lines = [
        "# ETG Source Release Manifest",
        "",
        f"- Version: `{manifest['version']}`",
        f"- DOI: `{manifest['doi']}`",
        f"- Maintainer: {manifest['maintainer']}",
        f"- Release file SHA-256: `{manifest.get('release_file_sha256') or 'pending'}`",
        "",
        "| Source | Rows | Access | License/terms | Release mode | Artifact |",
        "|---|---:|---|---|---|---|",
    ]
    for row in manifest["sources"]:
        lines.append(
            f"| `{row['source']}` | {row['row_count']:,} | {row['access_mechanism']} | "
            f"{row['license_or_terms']} | `{row['release_mode']}` | `{row['released_artifact']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ETG source release manifest")
    parser.add_argument("--input-manifest", type=Path, default=Path("data/unified_hacker_communities_neurips_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/release/source_release_manifest.json"))
    parser.add_argument("--version", default="1.0.0-rc1")
    parser.add_argument("--doi", default="TBD after immutable archive deposit")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("build-governed-release")
    p.add_argument("--input-jsonl", type=Path, default=Path("data/unified_hacker_communities_neurips.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("data/release/source_release_manifest.json"))
    p.add_argument("--output-jsonl", type=Path, default=Path("data/unified_hacker_communities_neurips_public.jsonl"))

    p = sub.add_parser("validate-governed-release")
    p.add_argument("--release-jsonl", type=Path, default=Path("data/unified_hacker_communities_neurips_public.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("data/release/source_release_manifest.json"))
    p.add_argument("--output", type=Path, default=Path("data/release/governed_release_validation.json"))

    args = parser.parse_args()
    if args.command == "build-governed-release":
        out = build_governed_release(args.input_jsonl, args.manifest, args.output_jsonl)
    elif args.command == "validate-governed-release":
        out = validate_governed_release(args.release_jsonl, args.manifest, args.output)
    else:
        manifest = build_manifest(args.input_manifest, args.output, args.version, args.doi)
        out = {"sources": len(manifest["sources"]), "release_mode_counts": manifest["release_mode_counts"]}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
