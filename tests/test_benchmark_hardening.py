import csv
import json
from pathlib import Path

import pytest

from etg.benchmark.audits import leakage_audit, overlap_audit, quality_by_task, sample_manual, summarize_llm_judge
from etg.benchmark.release_manifest import build_governed_release, build_manifest, validate_governed_release
from etg.benchmark.serious_baselines import aggregate
from etg.benchmark.serious_baselines import build_manifest as build_baseline_manifest
from etg.benchmark.splits import build_task2, build_task4, build_task5


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _tiny_benchmark(root: Path) -> Path:
    bench = root / "benchmark"
    for split in ("train", "val", "test"):
        rows = []
        for source in ("src_a", "src_b"):
            for label in (0, 1):
                rows.append(
                    {
                        "id": f"{split}-{source}-{label}",
                        "text": f"{source} {split} label {label} buffer overflow example",
                        "source": source,
                        "label": label,
                        "split": split,
                        "timestamp": "2024-01-01",
                    }
                )
        _write_jsonl(bench / "task1_exploit_clf" / f"{split}.jsonl", rows)

    for split in ("train", "val", "test"):
        _write_jsonl(
            bench / "task2_cve_linkage" / f"{split}.jsonl",
            [
                {
                    "id": f"{split}-q-{i}",
                    "text": f"query text {i} {split}",
                    "source": "src_a",
                    "cve_id": f"CVE-2024-000{i}",
                    "title": f"title {i}",
                    "split": split,
                }
                for i in range(3)
            ],
        )
    _write_jsonl(
        bench / "task2_cve_linkage" / "corpus.jsonl",
        [{"cve_id": "CVE-2024-0001", "text": "corpus text"}],
    )

    for split in ("train", "val", "test"):
        _write_jsonl(
            bench / "task3_severity" / f"{split}.jsonl",
            [
                {
                    "id": f"{split}-s-{i}",
                    "text": "sanitized advisory text without explicit template",
                    "text_raw": "Severity: HIGH\nCVSS: 8.1",
                    "severity_label": i % 4,
                    "severity": "high",
                    "source": "src_a",
                    "split": split,
                }
                for i in range(4)
            ],
        )
    return bench


def test_manual_audit_sampler_is_deterministic_and_stratified(tmp_path: Path):
    bench = _tiny_benchmark(tmp_path)
    out1 = tmp_path / "audit1"
    out2 = tmp_path / "audit2"
    sample_manual(bench, out1, n_task1=12, n_task2=6, seed=7)
    sample_manual(bench, out2, n_task1=12, n_task2=6, seed=7)

    p1 = out1 / "task1_manual_label_audit_packet.csv"
    p2 = out2 / "task1_manual_label_audit_packet.csv"
    assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")

    rows = list(csv.DictReader(p1.open(encoding="utf-8")))
    assert len(rows) == 12
    assert {row["benchmark_label"] for row in rows} == {"0", "1"}
    assert {row["split"] for row in rows} == {"train", "val", "test"}
    assert all(row["annotator_label"] == "" for row in rows)


def test_task2_quality_control_filters_unretrievable_rows(tmp_path: Path):
    data = tmp_path / "data"
    _write_jsonl(
        data / "github_advisory_cve_index.jsonl",
        [
            {
                "exploit_id": "ghsa-empty",
                "cve_id": "CVE-2026-0001",
                "title": "Useful GitHub advisory title but high risk source",
                "published": "2026-01-01",
            }
        ],
    )
    _write_jsonl(
        data / "cisa_kev_cve_index.jsonl",
        [
            {
                "exploit_id": "CVE-2026-0002",
                "cve_id": "CVE-2026-0002",
                "title": "Short",
                "exploit_text": "",
                "published": "2026-01-02",
            },
            {
                "exploit_id": "CVE-2026-0003",
                "cve_id": "CVE-2026-0003",
                "title": "Vendor product remote code execution vulnerability affecting gateway appliance",
                "exploit_text": "",
                "published": "2026-01-03",
            },
        ],
    )
    _write_jsonl(
        data / "nvd_cve_dict.jsonl",
        [{"cve_id": "CVE-2026-0003", "description": "Vendor product remote code execution in gateway appliance."}],
    )

    meta = build_task2(data, tmp_path / "benchmark", seed=1)
    rows = [
        json.loads(line)
        for line in (tmp_path / "benchmark" / "task2_cve_linkage" / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert meta["quality_controlled"] is True
    assert meta["skipped"]["quality_source_filter"] == 1
    assert meta["skipped"]["empty_or_short_query"] == 1
    assert len(rows) == 1
    assert rows[0]["text"].startswith("Vendor product remote code execution")


def test_hacker_exploit_labeling_and_signal_detection_build_from_weak_rules(tmp_path: Path):
    data = tmp_path / "data"
    actionable = "CVE-2026-0004 proof of concept exploit payload for remote code execution in gateway appliance"
    discussion = "CVE-2026-0005 vulnerability advisory patch notes for gateway appliance deployment remediation timeline details"
    noise = "marketplace account trade discussion with enough words to pass the minimum token filter"
    _write_jsonl(
        data / "hackforums_posts.jsonl",
        [
            {"id": "a1", "text": actionable, "timestamp": "2024-01-02T00:00:00+00:00", "forum_id": "hackforums"},
            {"id": "d1", "text": discussion, "timestamp": "2024-01-03T00:00:00+00:00", "forum_id": "hackforums"},
            {"id": "n1", "text": noise, "timestamp": "2024-01-04T00:00:00+00:00", "forum_id": "hackforums"},
        ],
    )
    _write_jsonl(
        data / "hackforums_cve_index.jsonl",
        [
            {"exploit_id": "a1", "cve_id": "CVE-2026-0004", "exploit_text": actionable, "published": "2024-01-02"},
        ],
    )
    _write_jsonl(
        data / "nvd_cve_dict.jsonl",
        [{"cve_id": "CVE-2026-0004", "description": "Gateway appliance remote code execution."}],
    )

    meta4 = build_task4(data, tmp_path / "benchmark", seed=1)
    test4 = [
        json.loads(line)
        for line in (tmp_path / "benchmark" / "task4_hacker_exploit_labeling" / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    labels = {row["id"]: row["label"] for row in test4}
    assert meta4["label_map"]["2"] == "actionable_exploit"
    assert labels == {"a1": 2, "d1": 1, "n1": 0}

    meta5 = build_task5(data, tmp_path / "benchmark", seed=1)
    test5 = [
        json.loads(line)
        for line in (tmp_path / "benchmark" / "task5_hacker_signal_detection" / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert meta5["unique_actionable_cves"] == 1
    assert any(row["id"] == "a1" and row["actionable_label"] == 1 and row["cve_id"] == "CVE-2026-0004" for row in test5)
    assert any(row["actionable_label"] == 0 for row in test5)


def test_task3_leakage_audit_passes_sanitized_and_fails_strict(tmp_path: Path):
    bench = _tiny_benchmark(tmp_path)
    report = leakage_audit(bench, tmp_path / "leakage.json", fail_on_strict=True)
    assert report["passed"] is True
    assert report["strict_sanitized_hits"] == 0
    assert report["strict_raw_hits"] > 0

    path = bench / "task3_severity" / "test.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["text"] = "Severity: HIGH"
    _write_jsonl(path, rows)
    with pytest.raises(SystemExit):
        leakage_audit(bench, tmp_path / "leakage_failed.json", fail_on_strict=True)


def test_overlap_quality_and_release_manifests(tmp_path: Path):
    bench = _tiny_benchmark(tmp_path)
    overlap = overlap_audit(bench, tmp_path / "overlap.json")
    quality = quality_by_task(bench, tmp_path / "quality.json")
    assert "task1" in overlap["tasks"]
    assert quality["tasks"]["task1"]["train"]["rows"] == 4

    input_manifest = tmp_path / "unified_manifest.json"
    input_manifest.write_text(
        json.dumps(
            {
                "output_path": str(tmp_path / "release.jsonl"),
                "source_written_rows": {"nvd_posts": 3, "unknown_posts": 2},
            }
        ),
        encoding="utf-8",
    )
    manifest = build_manifest(input_manifest, tmp_path / "source_release_manifest.json")
    assert len(manifest["sources"]) == 2
    assert all(row["release_mode"] for row in manifest["sources"])
    assert (tmp_path / "source_release_manifest.md").exists()

    baseline_manifest = build_baseline_manifest(tmp_path / "serious_baselines_manifest.json")
    assert {1, 2, 3, 4, 5} == set(baseline_manifest["seeds"])
    assert any(row["model"] == "cross_encoder_reranker" for row in baseline_manifest["baselines"])


def test_governed_release_redacts_pointer_only_text(tmp_path: Path):
    source_manifest = tmp_path / "unified_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "output_path": str(tmp_path / "full.jsonl"),
                "source_written_rows": {"0x00sec_posts": 1, "nvd_posts": 1},
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "source_release_manifest.json"
    build_manifest(source_manifest, manifest_path)
    _write_jsonl(
        tmp_path / "full.jsonl",
        [
            {"unified_id": "p1", "source_dataset": "0x00sec_posts", "text": "withhold me", "text_raw": "withhold raw"},
            {"unified_id": "t1", "source_dataset": "nvd_posts", "text": "keep me", "text_raw": "keep raw"},
        ],
    )
    governed = tmp_path / "public.jsonl"
    summary = build_governed_release(tmp_path / "full.jsonl", manifest_path, governed)
    assert summary["redacted_rows"] == 1
    rows = [json.loads(line) for line in governed.read_text(encoding="utf-8").splitlines()]
    pointer = rows[0]
    assert pointer["release_mode"] == "metadata_or_pointer_only"
    assert pointer["text"] is None and pointer["text_raw"] is None
    assert pointer["text_sha256"]
    assert rows[1]["release_mode"] == "redistributable_text"
    assert rows[1]["text"] == "keep me"
    validation = validate_governed_release(governed, manifest_path, tmp_path / "validation.json")
    assert validation["passed"] is True


def test_serious_baseline_aggregate_skips_cache_json_and_includes_sample_metrics(tmp_path: Path):
    seed_dir = tmp_path / "results" / "seed_1"
    seed_dir.mkdir(parents=True)
    (seed_dir / "task2_biencoder_sentence-transformers__all-MiniLM-L6-v2_corpus_cves.json").write_text(
        json.dumps(["CVE-2024-0001"]),
        encoding="utf-8",
    )
    (seed_dir / "task2_hybrid_seed1.json").write_text(
        json.dumps(
            {
                "kind": "hybrid_sparse_dense",
                "status": "ok",
                "metrics": {"recall_at_1": 0.25, "mrr": 0.5, "n_queries": 4},
            }
        ),
        encoding="utf-8",
    )

    out = aggregate(tmp_path / "results", tmp_path / "summary.json")
    metric_rows = {(row["model"], row["split"], row["metric"]): row for row in out["rows"]}
    assert ("hybrid_sparse_dense", "test_sample", "recall_at_1") in metric_rows
    assert metric_rows[("hybrid_sparse_dense", "test_sample", "recall_at_1")]["mean"] == 0.25


def test_llm_judge_summary_is_marked_nonhuman_and_counts_consensus(tmp_path: Path):
    task1 = tmp_path / "task1_llm.csv"
    with task1.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "audit_id",
                "benchmark_label",
                "cti_triage_status",
                "vuln_research_status",
                "cti_triage_needs_human_review",
                "vuln_research_needs_human_review",
                "llm_consensus",
                "llm_disagreement",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "audit_id": "T1-0001",
                "benchmark_label": "1",
                "cti_triage_status": "ok",
                "vuln_research_status": "ok",
                "cti_triage_needs_human_review": "false",
                "vuln_research_needs_human_review": "false",
                "llm_consensus": "1",
                "llm_disagreement": "false",
            }
        )

    task2 = tmp_path / "task2_llm.csv"
    with task2.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "audit_id",
                "cti_triage_status",
                "vuln_research_status",
                "cti_triage_needs_human_review",
                "vuln_research_needs_human_review",
                "llm_consensus",
                "llm_disagreement",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "audit_id": "CVE-0001",
                "cti_triage_status": "ok",
                "vuln_research_status": "ok",
                "cti_triage_needs_human_review": "false",
                "vuln_research_needs_human_review": "true",
                "llm_consensus": "True",
                "llm_disagreement": "false",
            }
        )

    summary = summarize_llm_judge(task1, task2, tmp_path / "summary.json")
    assert summary["audit_mode"] == "llm_assisted_not_human"
    assert summary["human_labels_present"] is False
    assert summary["tasks"]["task1"]["benchmark_agreement_on_consensus"] == 1.0
    assert summary["tasks"]["task2"]["consensus_precision_estimate"] == 1.0
    assert summary["tasks"]["task2"]["needs_human_review_rows"] == 1
