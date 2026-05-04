# HackerSignal Release Governance Addendum

This addendum documents source provenance, redistribution posture, and
responsible-release controls for HackerSignal. It is intended to be distributed with
the dataset card, datasheet, Croissant metadata, and benchmark scripts.

The machine-readable source of truth is
`data/release/source_release_manifest.json`, with a human-readable table in
`data/release/source_release_manifest.md`. The current release-candidate
manifest covers 27 source files: 5 `redistributable_text`, 9
`research_text_with_terms`, and 13 `metadata_or_pointer_only`.

## Release Modes

HackerSignal separates source records into release modes. These modes are encoded in
release manifests and should be treated as binding for downstream mirrors.

| Release mode | Meaning | User-facing artifact |
|---|---|---|
| `redistributable_text` | Source text can be redistributed under a public-domain, CC, permissive, or inherited dataset license. | Text, normalized metadata, provenance, and CVE links. |
| `research_text_with_terms` | Source text was publicly accessible, but redistribution depends on source terms or research-only conditions. | Text may be hosted only in the controlled research release with source attribution, no commercial reuse, and takedown support. |
| `metadata_or_pointer_only` | Source terms are unclear, restrictive, or likely to create reviewer/user risk if text is mirrored. | Stable source URL or source ID, timestamp, thread metadata, hashes, labels, and derived features; raw text is withheld. |
| `excluded` | Source should not be included in public releases. | Counts may appear in aggregate audits only. |

## Source Matrix

| Source layer / source | Collection mechanism | Current license or terms posture | Default release mode | Notes |
|---|---|---|---|---|
| NVD | REST/API export | US government public domain | `redistributable_text` | Included as CVE retrieval corpus. |
| CISA KEV | JSON feed | US government public domain | `redistributable_text` | Known-exploited catalog records. |
| GitHub Advisory | GraphQL API | CC BY 4.0 at source | `redistributable_text` | Include source attribution. |
| ExploitDB | CSV/repository export | CC BY-SA 4.0 at source | `redistributable_text` | Preserve share-alike attribution. |
| CVEfixes | Zenodo SQLite | CC BY 4.0 at source | `redistributable_text` | Fix commits and CVE metadata. |
| HackerOne | Public HuggingFace dataset | Public dataset terms vary by upstream release | `research_text_with_terms` | Keep provenance and source dataset citation. |
| DTL Exploits / AZSecure-derived imports | Public research artifact | Research-only / source-specific terms | `metadata_or_pointer_only` unless explicit redistribution approval is documented | Avoid broad raw-text mirroring. |
| Kaggle forum imports | Kaggle-hosted public datasets | Dataset-specific Kaggle licenses vary | `research_text_with_terms` or `metadata_or_pointer_only` by source file | Record exact Kaggle dataset URL and license in manifest. |
| DeepDarkCTI | GitHub public dataset | Repository license applies | `research_text_with_terms` | Preserve repo citation and commit/version. |
| 0x00sec, HackerSploit, ParrotSec, Hack The Box | Public Discourse APIs/pages | Site terms govern reuse | `metadata_or_pointer_only` by default for public release; `research_text_with_terms` for controlled review package | Prefer IDs, URLs, normalized metadata, hashes, and derived labels in broad release. |
| Go4Expert, Full Disclosure, Seebug, Vulnerability-Lab, ZeroScience | Public pages or archives | Mixed source terms | `metadata_or_pointer_only` unless license permits redistribution | Use source URLs and extracted CVE/provenance fields when terms are unclear. |

## Takedown and Correction Policy

Requests to remove, correct, or restrict a record should be sent through the
repository issue tracker or maintainer email listed in the datasheet. The
maintainer will:

1. Acknowledge the request within 10 business days.
2. Verify the affected source records through `unified_id`, source URL, or
   source-specific identifier.
3. Remove raw text from the next patch release when redistribution is disputed.
4. Preserve aggregate counts where doing so does not identify the requester.
5. Publish a changelog entry and updated manifest checksum for the patched
   release.

## Reviewer and Public Access

Reviewer-accessible artifacts must include:

- Versioned JSONL files or pointer-only equivalents. The public release file
  is `data/unified_hacker_communities_neurips_public.jsonl`; the full
  research build is not the default public artifact because it may contain raw
  text from sources whose manifest mode is `metadata_or_pointer_only`.
- Manifest with checksums, row counts, release modes, and source URLs
  (`data/release/source_release_manifest.json`).
- Croissant metadata.
- Datasheet and this governance addendum.
- Benchmark split metadata and runnable evaluation scripts for Tasks 1--5,
  including the hacker exploit-labeling and actionable signal-detection splits.
- Audit packets and reports under `data/audits/`, including manual label/CVE
  packets, Task 3 leakage audit, split-overlap audit, and quality-by-task
  audit.

Public releases should use immutable version tags. If a source changes its
terms after release, the next patch release should downgrade that source to
`metadata_or_pointer_only` until permissions are clarified.

## Responsible-Use Restrictions

HackerSignal is intended for defensive cybersecurity research, temporal retrieval,
provenance-aware benchmark design, and vulnerability lifecycle analysis. The
release disallows:

- Training or evaluating systems intended to generate exploit code, malware, or
  automated attack playbooks.
- De-anonymizing forum participants or linking pseudonymous identities across
  sources outside an approved research protocol.
- Operational blocking, law-enforcement, or employment decisions based solely
  on HackerSignal-trained model outputs.
- Republishing source text from `metadata_or_pointer_only` sources.

## Governed Public Export

The command below enforces the source manifest before public upload:

```bash
PYTHONPATH=src python -m etg.benchmark.release_manifest build-governed-release
PYTHONPATH=src python -m etg.benchmark.release_manifest validate-governed-release
```

For `metadata_or_pointer_only` sources, the governed export preserves
provenance, source IDs, timestamps, CVE references, text hashes, and text
lengths, but withholds `text` and `text_raw`. The validation report is written
to `data/release/governed_release_validation.json`.
