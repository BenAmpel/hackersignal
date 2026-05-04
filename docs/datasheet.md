# Datasheet for the HackerSignal Dataset

*Following the Gebru et al. (2021) datasheet format.*

---

## Motivation

**For what purpose was the dataset created?**
HackerSignal was created to support NLP and machine learning research on cybersecurity threat intelligence. Specifically, it enables: (1) CVE linkage retrieval — given exploit/advisory evidence text, retrieve the correct CVE description from a 340K-entry NVD corpus; (2) exploit type classification — 8-class vulnerability type prediction (injection, XSS, memory corruption, DoS, file inclusion, auth/access control, RCE, information disclosure) with temporal OOD evaluation; and (3) temporal CVE-disjoint generalization — prospective retrieval where the gold CVE sets in train and test are strictly disjoint. No prior *public* dataset links hacker community discourse, exploit databases, vulnerability advisories, and fix commits through a shared CVE identifier space at this scale.

**Who created the dataset, and on behalf of which entity?**
Anonymous (redacted for double-blind review). The dataset is an independent research contribution and is not created on behalf of any commercial entity.

**Who funded the creation of the dataset?**
This work was conducted independently. Prior related work by the same authors (prior work, ISI 2020 / MISQ 2024) was funded in part by DARPA and the NSF. HackerSignal itself received no direct external funding.

---

## Composition

**What do the instances represent?**
Each instance is a text document - a forum post, exploit advisory, CVE description, commit message, or security report - associated with a publication timestamp, source identifier, source layer, and pseudonymised author hash. A separate CVE index maps instances to CVE identifiers where source metadata exposes those links.

**How many instances are there in total?**
The NeurIPS release contains 7,447,646 exact-deduplicated rows across 64 public forum/source identifiers and eight source layers. The unified build consumed 8,262,749 input rows and removed 814,922 exact duplicate rows by normalized-text fingerprint. A release audit found 360,004 retained rows containing CVE identifiers.

**Does the dataset contain all possible instances, or is it a sample?**
It is a near-complete collection for sources with public APIs or structured feeds (NVD, GitHub Advisory, CISA KEV, HackerOne). For scraped or forum/API sources, coverage is limited by what was publicly accessible at collection time (2024-2026), rate limits, public pagination depth, and release-mode restrictions. The dataset does not include posts behind authentication walls.

**What data does each instance consist of?**
- `unified_id`: SHA-256-derived release identifier
- `source_dataset`: source file or collection name
- `source_layer`: normalized layer such as `hacker_community`, `exploit_archive`, `vulnerability_reference`, or `fix_commit_reference`
- `text`: UTF-8 post/advisory text (max 8,000 characters)
- `text_raw`: source text before release cleaning, where available
- `timestamp`: ISO 8601 publication datetime (UTC)
- `forum_id`: source identifier string
- `author_hash`: SHA-256 hash of source + author identifier (pseudonymised)
- `thread_url`, `thread_title`, `section`: preserved when exposed by the source

**Is there a label or target associated with each instance?**
The base corpus has no labels. The three benchmark splits add labels:
- Task 1 (CVE-R): gold CVE identifier for retrieval evaluation (metadata-derived linkage to 340K NVD corpus; train <2022, val 2022–2023, test 2024+)
- Task 2 (ETC): 8-class exploit type label (injection, xss, memory_corruption, dos, file_inclusion, auth_access, rce, info_disclosure) derived from ExploitDB, 0day.today, and HackerOne metadata; train <2020, val 2020–2022, test ≥2023
- Task 3 (TG): gold CVE identifier for CVE-disjoint retrieval evaluation; splits defined by CVE publication year so that train and test gold CVE sets are strictly disjoint

**Is any information missing from individual instances?**
All retained rows in the audited release have parseable timestamps, but some timestamps are placeholders or scrape dates when the original publication date was unavailable. Author information is pseudonymised and the original author string is not recoverable. Some imported sources contain garbled characters from source encoding errors.

**Are there relationships between instances?**
Yes. The CVE index creates explicit linkages between instances and CVE records. Within hacker forums, thread URL, thread title, and section/category structure are preserved where available, especially in newer Discourse-based crawls such as 0x00sec, HackerSploit, ParrotSec, and Hack The Box. Some legacy imports do not preserve thread structure and are marked `uncategorized`.

**Are there any errors, sources of noise, or redundancies?**
- **Duplicates and near-duplicates**: Exact cross-source duplicates are removed in the NeurIPS release. A deterministic 3% SimHash audit estimates that approximately 609,029 retained rows (8.18%) remain near-duplicates, usually from quoting, reposting, marketplace relisting, or mirrored advisories.
- **Date noise**: Some timestamps reflect scraping date rather than original publication date (flagged in `meta.json` per source).
- **Language noise**: The corpus is predominantly English, but `deepdarkcti` and several community sources contain multilingual posts (including Russian, Turkish, Persian, Arabic, and Chinese).
- **Short rows**: The release contains 1,569,263 rows under eight tokens. These are useful for interaction analysis but should usually be filtered for text-only classification and retrieval tasks.
- **Task-label caveats**: Task 1 (CVE-R) and Task 3 (TG) labels are metadata-derived CVE linkages (source URL or CVE regex extraction), not manually adjudicated ground truth; CVE-index absence is not a manual negative annotation; a 240-row stratified manual audit packet is provided in `data/audits/` for transparency. Task 2 (ETC) labels are derived from structured metadata fields (ExploitDB exploit descriptions and tags, 0day.today exploit type fields, HackerOne vulnerability type prefixes) mapped to an 8-class taxonomy via deterministic rules; they are not human-validated annotations. All three tasks use temporal splits so that evaluation reflects prospective performance on data postdating training.

**Is the dataset self-contained, or does it link to external resources?**
The governed public release is self-contained for sources whose terms permit text redistribution. Sources with ambiguous or restrictive terms are released as metadata/pointer-only records following `docs/release_governance.md`: provenance, source IDs, timestamps, CVE references, text hashes, and text lengths are retained, while `text` and `text_raw` are withheld. The CVE linkage retrieval tasks (Tasks 1 and 3) use NVD CVE descriptions as the retrieval corpus, which are included in the `nvd_posts.jsonl` file. No live external resources are required at inference time for redistributable-text records.

**Does the dataset contain data that might be considered confidential?**
No private communications, medical records, financial records, or government classified information is intentionally included. All text was collected from publicly accessible sources and no authentication was bypassed. Public availability does not eliminate privacy risk, so the release pseudonymises authors, documents source modes, supports takedown/correction requests, and can downgrade terms-ambiguous sources to metadata/pointer-only release.

**Does the dataset contain data that might be considered offensive or harmful?**
Yes. Hacker forums contain discussions of exploitation techniques, malware, and illegal activity. Advisory databases contain descriptions of real vulnerabilities that could inform attack development. Researchers should exercise appropriate judgment about downstream use. See **Recommended Uses** below.

---

## Collection Process

**How was the data collected?**
Data was collected through four mechanisms:
1. **Public APIs**: NVD (NIST), GitHub Advisory Database (GraphQL), CISA KEV (JSON feed), HackerOne (HuggingFace Hub)
2. **Public dataset ingestion**: Gayanku (Kaggle), HackForums (Kaggle), DTL Exploits (GitHub/AZSecure), CVEfixes (Zenodo), DeepDarkCTI (GitHub), Kaeli hacker forums (Kaggle), CrackingArena (Kaggle), ZeroDay (GitHub)
3. **Web scraping and public forum APIs**: ExploitDB (CSV), Vulnerability-Lab, ZeroScience Lab, 0x00sec, HackerSploit, ParrotSec, Hack The Box, Go4Expert, Full Disclosure mailing list archive, Seebug (partial)
4. **Pre-processed datasets**: ExploitDB (HuggingFace Hub)

**Who collected the data?**
Scripts were written and executed by the dataset authors. Collection occurred between January and April 2026.

**Over what time frame was the data collected?**
Collection scripts ran between January 2026 and April 2026. The audited release contains timestamped records spanning 1988 to April 2026, with the bulk of structured NVD records beginning in 2002.

**Were any ethical review processes conducted?**
The dataset consists of publicly accessible OSINT data, but the release treats public accessibility as insufficient by itself. No personally identifiable information was deliberately collected. Author strings are pseudonymised via SHA-256 hashing. No IRB review was required for collection of public OSINT data under the authors' institution's guidelines; however, researchers who build on HackerSignal for studies involving inferences about individuals should consult their own IRB.

**Did the individuals whose data is collected consent to collection?**
Individuals did not provide study-specific consent. Participants posted to public internet forums or public reporting/advisory channels, and all sources were publicly indexed or exposed through public APIs. The release therefore minimizes identity signals, preserves provenance, and applies source-specific release modes rather than assuming that public visibility alone justifies unrestricted redistribution.

---

## Preprocessing / Cleaning / Labeling

**Was any preprocessing or cleaning done?**
- HTML tags stripped from forum post content
- HTML entities unescaped
- Text truncated at 8,000 characters where required by source-specific loaders
- Encoding errors replaced with the Unicode replacement character (U+FFFD)
- Source provenance normalized into `source_dataset`, `source_layer`, `forum_id`, optional thread/category fields, and release-mode manifests
- Exact duplicates removed across sources by normalized-text SHA-256 fingerprint
- Near-duplicates removed via MinHash-LSH (128 permutations, 3-char k-shingles, Jaccard > 0.7)
- Reviewer-facing audit packets and leakage/overlap/quality reports are generated under `data/audits/`

**Was the raw data saved?**
Raw scraped HTML is not distributed. The processed JSONL files are the primary artifact.

**Is the software used to preprocess available?**
Yes — the collection and preprocessing scripts are available in the `src/etg/collectors/` directory of the accompanying code repository.

**Were any labels created by human annotators?**
No. All labels are derived automatically:
- Task 1 (CVE-R) labels: metadata-derived CVE linkages from source URL or CVE regex extraction; only low/medium CVE-linkage audit risk sources included (CISA KEV, CVEfixes, HackerOne)
- Task 2 (ETC) labels: exploit type from structured metadata fields (ExploitDB exploit descriptions and tags, 0day.today exploit type fields, HackerOne vulnerability type prefixes) mapped to an 8-class taxonomy via deterministic rules
- Task 3 (TG) labels: same CVE linkage derivation as Task 1; splits defined by CVE publication year to enforce strict gold-set disjointness

**What mechanisms were used to validate labels?**
Task 1 (CVE-R) and Task 3 (TG): CVE linkages are provided by source metadata and retained only for source families with low or medium CVE-linkage audit risk (CISA KEV, CVEfixes, and HackerOne). The split requires at least eight query tokens and uses titles as fallback query evidence when a body snippet is missing. A 240-row CVE-link stratified packet is prepared for manual precision auditing. A LLM-assisted triage pass reached consensus on 233/240 CVE-R rows and marked 115 consensus rows as correct CVE links; this automated audit is used to filter high-risk source families and prioritize human review rather than as a precision claim. The two LLM judges had 97.08% raw agreement, Cohen's kappa 0.9417, Krippendorff's nominal alpha 0.9418, and Gwet's AC1 0.9417. Task 3 (TG) additionally verifies programmatically that |C_train ∩ C_test| = 0 (strict CVE disjointness). Task 2 (ETC): exploit type labels are mapped from structured metadata to an 8-class taxonomy via deterministic rules and validated against a 240-row stratified manual audit packet in `data/audits/`. Source families (ExploitDB, 0day.today, HackerOne) were chosen for their structured metadata quality; records without a parseable exploit type are excluded.

---

## Uses

**Has the dataset been used for any tasks already?**
Earlier, smaller versions of the hacker forum data were used in prior work (ISI 2020) for exploit classification and in subsequent work (MISQ 2024) for hacker community analysis. Those datasets were not publicly released. HackerSignal is the first public release.

**What (other) tasks could the dataset be used for?**
- Exploit prediction and early warning systems
- Vulnerability lifecycle analysis (forum discussion → CVE assignment lag)
- Cross-lingual threat intelligence (for multilingual subsets)
- Hacker exploit labeling and actionable exploit-signal detection
- Graph neural network research (exploit–CVE–fix commit knowledge graph)
- Pre-training domain-adapted language models for cybersecurity
- Author attribution / pseudonym linking (subject to ethical constraints)

**Is there anything about the composition of the dataset that might impact future uses?**
- The forum sources skew toward English-language, Western cybersecurity communities. Threats emerging from non-English communities may be underrepresented.
- High-volume historical forum and dark-market imports skew toward 2015-2017 or earlier. Newer public sources such as DeepDarkCTI, 0x00sec, HackerSploit, ParrotSec, and Hack The Box extend into 2026 but are smaller production slices.
- CISA KEV covers only actively exploited vulnerabilities; using KEV alone as a "critical" proxy will over-represent critical labels relative to the true severity distribution.
- Task 1 (CVE-R) and Task 3 (TG) should not be interpreted as fully manually validated CVE-linkage benchmarks; labels are metadata-derived and CVE-index absence is not a manual negative annotation. Both tasks are quality-controlled by filtering to low/medium audit-risk source families (CISA KEV, CVEfixes, HackerOne).
- Task 2 (ETC) labels are derived from structured metadata rather than human annotation; the 8-class taxonomy covers the most common vulnerability types but may not capture rare or ambiguous classes equally well. Macro-F1 is the primary metric to surface per-class difficulty rather than accuracy alone.
- Some public forum/community sources may be released as metadata/pointer-only records depending on source terms and takedown requests.

**Are there tasks for which the dataset should not be used?**
- **Automated attack generation**: HackerSignal should not be used to fine-tune models for automated exploit code generation.
- **Targeting individuals**: Author hashes should not be used to attempt de-anonymisation or to link individuals across forums for non-research purposes.
- **Operational threat intelligence without human review**: Model outputs on HackerSignal should not be used as inputs to automated blocking or incident response systems without human review.

---

## Distribution

**Will the dataset be distributed to third parties?**
Yes. The dataset will be publicly available on HuggingFace Datasets at `DatasetSubmission/HackerSignal`.

**How will the dataset be distributed?**
As governed JSONL files on HuggingFace Hub, accessible via the `datasets` library: `load_dataset("DatasetSubmission/HackerSignal")`. The default public artifact is `data/unified_hacker_communities_neurips_public.jsonl`, which enforces source-level release modes from `data/release/source_release_manifest.json`.

**When will the dataset be distributed?**
Concurrent with camera-ready submission following peer review.

**Will the dataset be distributed under a copyright or other IP license?**
Collection code, benchmark code, schemas, manifests, Croissant metadata, and release metadata are released under **Creative Commons Attribution 4.0 International (CC BY 4.0)** unless otherwise noted. Redistributed source text retains source-specific terms documented in the governance addendum, provenance metadata, and `data/release/source_release_manifest.json`.

**Have any third parties imposed IP-based restrictions on the data?**
NVD data is in the public domain (US government work). ExploitDB data is licensed CC BY-SA 4.0 at source; our release inherits this for those records. GitHub Advisory data is licensed CC BY 4.0 at source. CISA KEV is public domain. Forum and community post data was collected only from publicly accessible pages or public forum APIs; users should consult the provenance fields, `docs/release_governance.md`, and `data/release/source_release_manifest.json` before redistributing source text outside research use. The current manifest classifies 5 sources as redistributable text, 9 as research text with terms, and 13 as metadata/pointer-only.

**Do any export controls apply?**
The dataset describes publicly known vulnerabilities, exploits, and security research. No information is classified. Standard academic export control policies apply.

---

## Maintenance

**Who will be maintaining the dataset?**
Anonymous (see dataset repository issue tracker).

**How can owners/curators be contacted?**
Via GitHub issues at the accompanying repository, or by email. Removal, correction, or release-mode downgrade requests follow the takedown/correction procedure in `docs/release_governance.md`.

**Will the dataset be updated?**
Yes — annual updates are planned to incorporate new CVE records, CISA KEV entries, and newly published advisories. Version numbers follow semantic versioning (MAJOR.MINOR.PATCH).

**If the dataset becomes obsolete, how will this be communicated?**
Via a deprecation notice in the HuggingFace dataset card and repository README.

**Is there a mechanism for others to contribute?**
Contributions via GitHub pull requests are welcome, subject to the same provenance and quality standards described here.

**Will older versions remain available?**
Yes — HuggingFace Hub preserves all tagged versions.

---

*Datasheet version 1.0 — April 2026*
