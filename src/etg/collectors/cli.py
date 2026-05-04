"""Unified CLI for ETG data collectors.

Usage::

    python -m etg.collectors.cli --all
    python -m etg.collectors.cli --exploitdb --nvd --nvd-api-key KEY
    python -m etg.collectors.cli --packetstorm --ps-max-pages 200
    python -m etg.collectors.cli --fulldisclosure --fd-start-year 2010
    python -m etg.collectors.cli --build-ev-pairs
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from etg.collectors.exploitdb import collect as collect_exploitdb
from etg.collectors.nvd import collect as collect_nvd
from etg.collectors.packetstorm import collect as collect_packetstorm
from etg.collectors.zeroday import collect as collect_zeroday
from etg.collectors.github_advisories import collect as collect_github_advisories
from etg.collectors.cisa_kev import collect as collect_cisa_kev
from etg.collectors.go4expert import collect as collect_go4expert
from etg.collectors.antionline import collect as collect_antionline
from etg.collectors.hackforums_importer import collect as collect_hackforums
from etg.collectors.gayanku_importer import collect as collect_gayanku
from etg.collectors.hackerone_importer import collect as collect_hackerone
from etg.collectors.exploitdb_hf_importer import collect as collect_exploitdb_hf
from etg.collectors.cvefixes_importer import collect as collect_cvefixes
from etg.collectors.evolution_importer import collect as collect_evolution
from etg.collectors.zeroxzerosec import collect as collect_0x00sec
from etg.collectors.discourse_communities import (
    collect_hackersploit,
    collect_hackthebox,
    collect_parrotsec,
)
from etg.collectors.zeroscience_scraper import collect as collect_zeroscience
from etg.collectors.vulnlab_scraper import collect as collect_vulnlab
from etg.collectors.seebug_scraper import collect as collect_seebug
from etg.collectors.antichat_scraper import collect as collect_antichat
from etg.collectors.dtl_el import (
    collect_hacker_exploits,
    collect_exploits as collect_dtl_exploits,
    collect_public_exploits,
    collect_kaeli,
    collect_cve_hacker_forum,
)
from etg.collectors.dedup import deduplicate_posts
from etg.collectors.ev_builder import build as build_ev_pairs

log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m etg.collectors.cli",
        description="Collect exploit/vuln data from public sources and build EV pairs.",
    )

    # --- collector flags ---
    p.add_argument("--all", action="store_true", help="Run all collectors then build EV pairs.")
    p.add_argument("--exploitdb", action="store_true", help="Run the Exploit-DB collector.")
    p.add_argument("--nvd", action="store_true", help="Run the NVD collector.")
    p.add_argument(
        "--nvd-api-key",
        metavar="KEY",
        default=None,
        help="NVD API key (50 req/30 s instead of 5 req/30 s).",
    )
    p.add_argument(
        "--nvd-start-year",
        type=int,
        default=2002,
        metavar="INT",
        help="Earliest year to collect from NVD (default: 2002).",
    )
    p.add_argument("--packetstorm", action="store_true", help="Run the Packet Storm collector.")
    p.add_argument(
        "--ps-max-pages",
        type=int,
        default=500,
        metavar="INT",
        help="Maximum pages to fetch from Packet Storm (default: 500).",
    )
    p.add_argument("--fulldisclosure", action="store_true", help="Run the Full Disclosure collector.")
    p.add_argument(
        "--fd-start-year",
        type=int,
        default=2002,
        metavar="INT",
        help="Earliest year to collect from Full Disclosure (default: 2002).",
    )
    p.add_argument("--zeroday", action="store_true", help="Run the 0day.today collector (local HDF5).")
    p.add_argument(
        "--zeroday-h5",
        metavar="PATH",
        default=None,
        help="Path to 0DayData.h5 (default: OneDrive path).",
    )
    p.add_argument("--github", action="store_true", help="Run the GitHub Security Advisories collector.")
    p.add_argument(
        "--github-token",
        metavar="TOKEN",
        default=None,
        help="GitHub personal access token (raises rate limit from 60/hr to 5000/hr).",
    )
    p.add_argument("--cisa-kev", action="store_true", help="Run the CISA Known Exploited Vulnerabilities collector.")

    # --- DTL-EL HDF5 sources ---
    p.add_argument("--hacker-exploits", action="store_true",
                   help="Convert hackerExploits.h5 (real hacker forum posts, ~2.5 GB).")
    p.add_argument("--hacker-exploits-h5", metavar="PATH", default=None,
                   help="Path to hackerExploits.h5 (default: OneDrive/DTL-EL/data/).")
    p.add_argument("--dtl-exploits", action="store_true",
                   help="Convert exploits.h5 (DTL-EL source-domain public exploits, ~96 k).")
    p.add_argument("--dtl-exploits-h5", metavar="PATH", default=None,
                   help="Path to exploits.h5.")
    p.add_argument("--public-exploits", action="store_true",
                   help="Convert PublicExploits.h5 (ExploitDB subset with CVE + CVSS, 24 k).")
    p.add_argument("--public-exploits-h5", metavar="PATH", default=None,
                   help="Path to PublicExploits.h5.")
    p.add_argument("--kaeli", action="store_true",
                   help="Convert KaeliHackerExploits.h5 (384 MB hacker-forum subset).")
    p.add_argument("--kaeli-h5", metavar="PATH", default=None,
                   help="Path to KaeliHackerExploits.h5.")
    p.add_argument("--cve-hacker-forum", action="store_true",
                   help="Convert CVEHackerForum.h5 (81 annotated AntiChat CVE posts).")
    p.add_argument("--cve-hacker-forum-h5", metavar="PATH", default=None,
                   help="Path to CVEHackerForum.h5.")

    p.add_argument("--go4expert", action="store_true",
                   help="Scrape go4expert.com (XenForo 1.x security forums).")
    p.add_argument("--go4expert-delay", type=float, default=1.5, metavar="SECS",
                   help="Request delay for go4expert (default: 1.5 s).")
    p.add_argument("--go4expert-max-pages", type=int, default=500, metavar="N",
                   help="Max thread-list pages per subforum for go4expert (default: 500).")

    p.add_argument("--antionline", action="store_true",
                   help="Scrape antionline.com security subforums via RSS + thread fetch.")
    p.add_argument("--antionline-delay", type=float, default=2.0, metavar="SECS",
                   help="Request delay for antionline (default: 2.0 s).")

    # --- offline dataset importers ---
    p.add_argument("--hackforums", action="store_true",
                   help="Import HackForums Internet Archive dataset (.jl.gz files).")
    p.add_argument("--hackforums-dir", metavar="PATH", default=None,
                   help="Directory containing hackforums-out-fix.jl.*.gz files.")

    p.add_argument("--gayanku", action="store_true",
                   help="Import gayanku darkweb forum CSV (Clearnedup_ALL_7.csv).")
    p.add_argument("--gayanku-csv", metavar="PATH", default=None,
                   help="Path to Clearnedup_ALL_7.csv.")

    p.add_argument("--hackerone", action="store_true",
                   help="Import HackerOne disclosed bug reports (HuggingFace dataset).")
    p.add_argument("--hackerone-dir", metavar="PATH", default=None,
                   help="Directory containing the HackerOne HuggingFace dataset.")

    p.add_argument("--exploitdb-hf", action="store_true",
                   help="Import ExploitDB HuggingFace dataset (Q&A format).")
    p.add_argument("--exploitdb-hf-dir", metavar="PATH", default=None,
                   help="Directory containing the ExploitDB HuggingFace dataset.")

    p.add_argument("--cvefixes", action="store_true",
                   help="Import CVEfixes SQLite dataset (CVEs + fix commits).")
    p.add_argument("--cvefixes-dir", metavar="PATH", default=None,
                   help="Directory containing the CVEfixes .db file.")

    p.add_argument("--evolution", action="store_true",
                   help="Import Evolution dark-web forum TSV dataset (Zenodo).")
    p.add_argument("--evolution-dir", metavar="PATH", default=None,
                   help="Directory containing the extracted Evolution dataset.")

    # --- live scrapers (new communities) ---
    p.add_argument("--0x00sec", dest="zeroxzerosec", action="store_true",
                   help="Scrape 0x00sec Discourse forum (forum.0x00sec.org).")
    p.add_argument("--0x00sec-delay", dest="zeroxzerosec_delay", type=float, default=1.0,
                   metavar="SECS", help="Request delay for 0x00sec (default: 1.0 s).")
    p.add_argument("--0x00sec-max-pages", dest="zeroxzerosec_max_pages", type=int, default=200,
                   metavar="N", help="Max topic-list pages per category (default: 200).")

    p.add_argument("--zeroscience", action="store_true",
                   help="Scrape Zeroscience.mk advisory archive.")
    p.add_argument("--zeroscience-delay", type=float, default=1.5, metavar="SECS",
                   help="Request delay for Zeroscience (default: 1.5 s).")

    p.add_argument("--vulnlab", action="store_true",
                   help="Scrape Vulnerability-Lab.com advisory archive.")
    p.add_argument("--vulnlab-delay", type=float, default=2.0, metavar="SECS",
                   help="Request delay for Vulnerability-Lab (default: 2.0 s).")
    p.add_argument("--vulnlab-max-pages", type=int, default=50, metavar="N",
                   help="Max listing pages per category for Vulnerability-Lab (default: 50).")

    p.add_argument("--seebug", action="store_true",
                   help="Scrape Seebug.org vulnerability database (Chinese, CVE-linked).")
    p.add_argument("--seebug-delay", type=float, default=2.0, metavar="SECS",
                   help="Request delay for Seebug (default: 2.0 s).")
    p.add_argument("--seebug-max-pages", type=int, default=3000, metavar="N",
                   help="Max listing pages for Seebug (default: 3000).")

    p.add_argument("--antichat", action="store_true",
                   help="Scrape AntiChat.xyz vBulletin forum (Russian, security subforums).")
    p.add_argument("--antichat-delay", type=float, default=1.5, metavar="SECS",
                   help="Request delay for AntiChat (default: 1.5 s).")
    p.add_argument("--antichat-max-pages", type=int, default=200, metavar="N",
                   help="Max thread-list pages per subforum for AntiChat (default: 200).")

    p.add_argument("--hackersploit", action="store_true",
                   help="Scrape HackerSploit Discourse forum security categories.")
    p.add_argument("--hackersploit-delay", type=float, default=1.0, metavar="SECS",
                   help="Request delay for HackerSploit (default: 1.0 s).")
    p.add_argument("--hackersploit-max-pages", type=int, default=200, metavar="N",
                   help="Max topic-list pages per HackerSploit category (default: 200).")
    p.add_argument("--hackersploit-max-topics", type=int, default=0, metavar="N",
                   help="Optional cap on HackerSploit topics for smoke runs (0 = unlimited).")

    p.add_argument("--parrotsec", action="store_true",
                   help="Scrape ParrotSec Discourse community security categories.")
    p.add_argument("--parrotsec-delay", type=float, default=1.0, metavar="SECS",
                   help="Request delay for ParrotSec (default: 1.0 s).")
    p.add_argument("--parrotsec-max-pages", type=int, default=200, metavar="N",
                   help="Max topic-list pages per ParrotSec category (default: 200).")
    p.add_argument("--parrotsec-max-topics", type=int, default=0, metavar="N",
                   help="Optional cap on ParrotSec topics for smoke runs (0 = unlimited).")

    p.add_argument("--hackthebox", action="store_true",
                   help="Scrape Hack The Box Discourse forum selected categories.")
    p.add_argument("--hackthebox-delay", type=float, default=1.0, metavar="SECS",
                   help="Request delay for Hack The Box (default: 1.0 s).")
    p.add_argument("--hackthebox-max-pages", type=int, default=200, metavar="N",
                   help="Max topic-list pages per Hack The Box category (default: 200).")
    p.add_argument("--hackthebox-max-topics", type=int, default=0, metavar="N",
                   help="Optional cap on Hack The Box topics for smoke runs (0 = unlimited).")

    p.add_argument("--deduplicate", action="store_true",
                   help="Cross-file deduplication: remove duplicate posts across all *_posts.jsonl files.")
    p.add_argument("--dedup-dry-run", action="store_true",
                   help="Report duplicates without modifying files (use with --deduplicate).")
    p.add_argument("--build-ev-pairs", action="store_true", help="Run EV pair builder only.")

    # --- output / execution ---
    p.add_argument(
        "--output-dir",
        default="data/",
        metavar="PATH",
        help="Root output directory (default: data/).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="INT",
        help="Number of parallel collector threads (default: 1).",
    )

    # --- logging ---
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress scrapling logger output.",
    )

    return p


def _print_summary(results: list[tuple[str, int | str]]) -> None:
    """Print a simple table: collector name → post count (or error)."""
    if not results:
        return
    col_w = max(len(name) for name, _ in results) + 2
    print()
    print(f"{'Collector':<{col_w}}  {'Posts / Result':>15}")
    print("-" * (col_w + 18))
    for name, count in results:
        value = str(count) if isinstance(count, int) else count
        print(f"{name:<{col_w}}  {value:>15}")
    print()


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --- configure logging ---
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    if args.quiet:
        logging.getLogger("scrapling").setLevel(logging.ERROR)

    # --- determine which collectors to run ---
    run_exploitdb = args.all or args.exploitdb
    run_nvd = args.all or args.nvd
    run_packetstorm = args.all or args.packetstorm
    run_fulldisclosure = args.all or args.fulldisclosure
    run_zeroday = args.all or args.zeroday
    run_github = args.all or args.github
    run_cisa_kev = args.all or args.cisa_kev
    run_go4expert    = args.all or args.go4expert
    run_antionline   = args.all or args.antionline
    run_hackforums   = args.all or args.hackforums
    run_gayanku      = args.all or args.gayanku
    run_hackerone    = args.all or args.hackerone
    run_exploitdb_hf = args.all or args.exploitdb_hf
    run_cvefixes     = args.all or args.cvefixes
    run_evolution    = args.all or args.evolution
    run_0x00sec      = args.all or args.zeroxzerosec
    run_zeroscience  = args.all or args.zeroscience
    run_vulnlab      = args.all or args.vulnlab
    run_seebug       = args.all or args.seebug
    run_antichat     = args.all or args.antichat
    run_hackersploit = args.all or args.hackersploit
    run_parrotsec    = args.all or args.parrotsec
    run_hackthebox   = args.all or args.hackthebox
    run_hacker_exploits = args.all or args.hacker_exploits
    run_dtl_exploits    = args.all or args.dtl_exploits
    run_public_exploits = args.all or args.public_exploits
    run_kaeli           = args.all or args.kaeli
    run_cve_hacker_forum = args.all or args.cve_hacker_forum
    run_dedup = args.deduplicate
    run_ev = args.all or args.build_ev_pairs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- build collector task list ---
    # Each task is (label, callable, kwargs)
    tasks: list[tuple[str, object, dict]] = []

    if run_exploitdb:
        tasks.append(("ExploitDB", collect_exploitdb, {
            "clone_dir": output_dir / "exploitdb_repo",
            "output": output_dir / "exploitdb_posts.jsonl",
            "cve_index_output": output_dir / "exploitdb_cve_index.jsonl",
        }))

    if run_nvd:
        nvd_kwargs: dict = {
            "output": output_dir / "nvd_posts.jsonl",
            "cve_dict_output": output_dir / "nvd_cve_dict.jsonl",
            "start_year": args.nvd_start_year,
        }
        if args.nvd_api_key:
            nvd_kwargs["api_key"] = args.nvd_api_key
        tasks.append(("NVD", collect_nvd, nvd_kwargs))

    if run_packetstorm:
        tasks.append((
            "PacketStorm",
            collect_packetstorm,
            {
                "output": output_dir / "packetstorm_posts.jsonl",
                "cve_index_output": output_dir / "packetstorm_cve_index.jsonl",
                "max_pages": args.ps_max_pages,
            },
        ))

    if run_fulldisclosure:
        from etg.collectors.fulldisclosure import collect as collect_fulldisclosure

        tasks.append((
            "FullDisclosure",
            collect_fulldisclosure,
            {
                "output": output_dir / "fulldisclosure_posts.jsonl",
                "start_year": args.fd_start_year,
            },
        ))

    if run_zeroday:
        zd_kwargs: dict = {
            "output": output_dir / "zeroday_posts.jsonl",
            "cve_index_output": output_dir / "zeroday_cve_index.jsonl",
        }
        if args.zeroday_h5:
            zd_kwargs["h5_path"] = Path(args.zeroday_h5)
        tasks.append(("ZeroDayToday", collect_zeroday, zd_kwargs))

    if run_github:
        gh_kwargs: dict = {
            "output": output_dir / "github_advisory_posts.jsonl",
            "cve_index_output": output_dir / "github_advisory_cve_index.jsonl",
        }
        if args.github_token:
            gh_kwargs["github_token"] = args.github_token
        tasks.append(("GitHubAdvisories", collect_github_advisories, gh_kwargs))

    if run_cisa_kev:
        tasks.append(("CISA_KEV", collect_cisa_kev, {
            "output": output_dir / "cisa_kev_posts.jsonl",
            "cve_index_output": output_dir / "cisa_kev_cve_index.jsonl",
        }))

    if run_go4expert:
        tasks.append(("Go4Expert", collect_go4expert, {
            "output": output_dir / "go4expert_posts.jsonl",
            "cve_index_output": output_dir / "go4expert_cve_index.jsonl",
            "request_delay": args.go4expert_delay,
            "max_list_pages": args.go4expert_max_pages,
        }))

    if run_antionline:
        tasks.append(("AntiOnline", collect_antionline, {
            "output": output_dir / "antionline_posts.jsonl",
            "cve_index_output": output_dir / "antionline_cve_index.jsonl",
            "request_delay": args.antionline_delay,
        }))

    if run_hackforums:
        hf_kwargs: dict = {
            "output": output_dir / "hackforums_posts.jsonl",
            "cve_index_output": output_dir / "hackforums_cve_index.jsonl",
        }
        if args.hackforums_dir:
            hf_kwargs["data_dir"] = Path(args.hackforums_dir)
        tasks.append(("HackForums", collect_hackforums, hf_kwargs))

    if run_gayanku:
        gk_kwargs: dict = {
            "output_dir": output_dir,   # write one file per forum
        }
        if args.gayanku_csv:
            gk_kwargs["csv_path"] = Path(args.gayanku_csv)
        tasks.append(("Gayanku Darkweb", collect_gayanku, gk_kwargs))

    if run_hackerone:
        h1_kwargs: dict = {
            "output": output_dir / "hackerone_posts.jsonl",
            "cve_index_output": output_dir / "hackerone_cve_index.jsonl",
        }
        if args.hackerone_dir:
            h1_kwargs["data_dir"] = Path(args.hackerone_dir)
        tasks.append(("HackerOne", collect_hackerone, h1_kwargs))

    if run_exploitdb_hf:
        edbhf_kwargs: dict = {
            "output": output_dir / "exploitdb_hf_posts.jsonl",
            "cve_index_output": output_dir / "exploitdb_hf_cve_index.jsonl",
        }
        if args.exploitdb_hf_dir:
            edbhf_kwargs["data_dir"] = Path(args.exploitdb_hf_dir)
        tasks.append(("ExploitDB-HF", collect_exploitdb_hf, edbhf_kwargs))

    if run_cvefixes:
        cvef_kwargs: dict = {
            "output": output_dir / "cvefixes_posts.jsonl",
            "cve_index_output": output_dir / "cvefixes_cve_index.jsonl",
        }
        if args.cvefixes_dir:
            cvef_kwargs["data_dir"] = Path(args.cvefixes_dir)
        tasks.append(("CVEfixes", collect_cvefixes, cvef_kwargs))

    if run_evolution:
        evo_kwargs: dict = {
            "output": output_dir / "evolution_posts.jsonl",
            "cve_index_output": output_dir / "evolution_cve_index.jsonl",
        }
        if args.evolution_dir:
            evo_kwargs["data_dir"] = Path(args.evolution_dir)
        tasks.append(("Evolution", collect_evolution, evo_kwargs))

    if run_0x00sec:
        tasks.append(("0x00sec", collect_0x00sec, {
            "output": output_dir / "0x00sec_posts.jsonl",
            "cve_index_output": output_dir / "0x00sec_cve_index.jsonl",
            "request_delay": args.zeroxzerosec_delay,
            "max_pages": args.zeroxzerosec_max_pages,
        }))

    if run_zeroscience:
        tasks.append(("Zeroscience", collect_zeroscience, {
            "output": output_dir / "zeroscience_posts.jsonl",
            "cve_index_output": output_dir / "zeroscience_cve_index.jsonl",
            "request_delay": args.zeroscience_delay,
        }))

    if run_vulnlab:
        tasks.append(("VulnerabilityLab", collect_vulnlab, {
            "output": output_dir / "vulnlab_posts.jsonl",
            "cve_index_output": output_dir / "vulnlab_cve_index.jsonl",
            "request_delay": args.vulnlab_delay,
            "max_pages_per_cat": args.vulnlab_max_pages,
        }))

    if run_seebug:
        tasks.append(("Seebug", collect_seebug, {
            "output": output_dir / "seebug_posts.jsonl",
            "cve_index_output": output_dir / "seebug_cve_index.jsonl",
            "request_delay": args.seebug_delay,
            "max_pages": args.seebug_max_pages,
        }))

    if run_antichat:
        tasks.append(("AntiChat", collect_antichat, {
            "output": output_dir / "antichat_posts.jsonl",
            "cve_index_output": output_dir / "antichat_cve_index.jsonl",
            "request_delay": args.antichat_delay,
            "max_pages": args.antichat_max_pages,
        }))

    if run_hackersploit:
        tasks.append(("HackerSploit", collect_hackersploit, {
            "output": output_dir / "hackersploit_posts.jsonl",
            "cve_index_output": output_dir / "hackersploit_cve_index.jsonl",
            "request_delay": args.hackersploit_delay,
            "max_pages": args.hackersploit_max_pages,
            "max_topics": args.hackersploit_max_topics or None,
        }))

    if run_parrotsec:
        tasks.append(("ParrotSec", collect_parrotsec, {
            "output": output_dir / "parrotsec_posts.jsonl",
            "cve_index_output": output_dir / "parrotsec_cve_index.jsonl",
            "request_delay": args.parrotsec_delay,
            "max_pages": args.parrotsec_max_pages,
            "max_topics": args.parrotsec_max_topics or None,
        }))

    if run_hackthebox:
        tasks.append(("HackTheBox", collect_hackthebox, {
            "output": output_dir / "hackthebox_posts.jsonl",
            "cve_index_output": output_dir / "hackthebox_cve_index.jsonl",
            "request_delay": args.hackthebox_delay,
            "max_pages": args.hackthebox_max_pages,
            "max_topics": args.hackthebox_max_topics or None,
        }))

    if run_hacker_exploits:
        he_kwargs: dict = {
            "output": output_dir / "hacker_exploits_posts.jsonl",
            "cve_index_output": output_dir / "hacker_exploits_cve_index.jsonl",
        }
        if args.hacker_exploits_h5:
            from pathlib import Path as _Path
            he_kwargs["h5_path"] = _Path(args.hacker_exploits_h5)
        tasks.append(("HackerExploits", collect_hacker_exploits, he_kwargs))

    if run_dtl_exploits:
        de_kwargs: dict = {
            "output": output_dir / "dtl_exploits_posts.jsonl",
            "cve_index_output": output_dir / "dtl_exploits_cve_index.jsonl",
        }
        if args.dtl_exploits_h5:
            from pathlib import Path as _Path
            de_kwargs["h5_path"] = _Path(args.dtl_exploits_h5)
        tasks.append(("DTL_Exploits", collect_dtl_exploits, de_kwargs))

    if run_public_exploits:
        pe_kwargs: dict = {
            "output": output_dir / "public_exploits_posts.jsonl",
            "cve_index_output": output_dir / "public_exploits_cve_index.jsonl",
        }
        if args.public_exploits_h5:
            from pathlib import Path as _Path
            pe_kwargs["h5_path"] = _Path(args.public_exploits_h5)
        tasks.append(("PublicExploits", collect_public_exploits, pe_kwargs))

    if run_kaeli:
        kaeli_kwargs: dict = {
            "output": output_dir / "kaeli_hacker_posts.jsonl",
            "cve_index_output": output_dir / "kaeli_hacker_cve_index.jsonl",
        }
        if args.kaeli_h5:
            from pathlib import Path as _Path
            kaeli_kwargs["h5_path"] = _Path(args.kaeli_h5)
        tasks.append(("Kaeli", collect_kaeli, kaeli_kwargs))

    if run_cve_hacker_forum:
        chf_kwargs: dict = {
            "output": output_dir / "cve_hacker_forum_posts.jsonl",
            "cve_index_output": output_dir / "cve_hacker_forum_cve_index.jsonl",
        }
        if args.cve_hacker_forum_h5:
            from pathlib import Path as _Path
            chf_kwargs["h5_path"] = _Path(args.cve_hacker_forum_h5)
        tasks.append(("CVEHackerForum", collect_cve_hacker_forum, chf_kwargs))

    # --- run collectors ---
    summary: list[tuple[str, int | str]] = []

    def _run(label: str, fn, kwargs: dict) -> tuple[str, int | str]:
        log.info("Starting collector: %s", label)
        try:
            result = fn(**kwargs)
            # collect() returns (post_count, aux_count) tuple; take the first element
            if isinstance(result, tuple):
                count = result[0]
            elif result is None:
                count = 0
            else:
                count = int(result)
            log.info("Finished collector: %s — %d posts", label, count)
            return label, count
        except Exception as exc:
            log.error("Collector %s failed: %s", label, exc, exc_info=True)
            return label, f"ERROR: {exc}"

    if tasks:
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(_run, label, fn, kwargs): label
                    for label, fn, kwargs in tasks
                }
                for future in as_completed(futures):
                    summary.append(future.result())
        else:
            for label, fn, kwargs in tasks:
                summary.append(_run(label, fn, kwargs))

    # --- cross-file deduplication ---
    if run_dedup:
        log.info("Running cross-file deduplication…")
        try:
            dedup_stats = deduplicate_posts(output_dir, dry_run=args.dedup_dry_run)
            removed = dedup_stats.get("total_removed", 0)
            after   = dedup_stats.get("total_posts_after", 0)
            label   = "dedup (dry run)" if args.dedup_dry_run else "dedup"
            summary.append((label, f"removed {removed:,}, kept {after:,}"))
            log.info("Dedup complete: removed=%d, kept=%d", removed, after)
        except Exception as exc:
            log.error("Dedup failed: %s", exc, exc_info=True)
            summary.append(("dedup", f"ERROR: {exc}"))

    # --- build EV pairs ---
    if run_ev:
        log.info("Building EV pairs…")
        try:
            extra_indexes = []
            for idx_name in (
                "go4expert_cve_index.jsonl",
                "antionline_cve_index.jsonl",
                "zeroday_cve_index.jsonl",
                "github_advisory_cve_index.jsonl",
                "packetstorm_cve_index.jsonl",
                "cisa_kev_cve_index.jsonl",
                "hacker_exploits_cve_index.jsonl",
                "dtl_exploits_cve_index.jsonl",
                "public_exploits_cve_index.jsonl",
                "kaeli_hacker_cve_index.jsonl",
                "cve_hacker_forum_cve_index.jsonl",
                "hackforums_cve_index.jsonl",
                "gayanku_cve_index.jsonl",
                "hackerone_cve_index.jsonl",
                "exploitdb_hf_cve_index.jsonl",
                "cvefixes_cve_index.jsonl",
                "evolution_cve_index.jsonl",
                "0x00sec_cve_index.jsonl",
                "zeroscience_cve_index.jsonl",
                "vulnlab_cve_index.jsonl",
                "seebug_cve_index.jsonl",
                "antichat_cve_index.jsonl",
                "hackersploit_cve_index.jsonl",
                "parrotsec_cve_index.jsonl",
                "hackthebox_cve_index.jsonl",
            ):
                idx_path = output_dir / idx_name
                if idx_path.exists() and idx_path.stat().st_size > 0:
                    extra_indexes.append(idx_path)
            ev_result = build_ev_pairs(
                exploitdb_cve_index=output_dir / "exploitdb_cve_index.jsonl",
                nvd_cve_dict=output_dir / "nvd_cve_dict.jsonl",
                output=output_dir / "ev_pairs.jsonl",
                extra_cve_indexes=extra_indexes or None,
            )
            if isinstance(ev_result, tuple):
                ev_count = sum(ev_result)
            elif hasattr(ev_result, "__len__"):
                ev_count = len(ev_result)
            else:
                ev_count = int(ev_result or 0)
            summary.append(("EV pairs built", ev_count))
            log.info("EV pair build complete — %d pairs", ev_count)
        except Exception as exc:
            log.error("EV pair build failed: %s", exc, exc_info=True)
            summary.append(("EV pairs built", f"ERROR: {exc}"))

    _print_summary(summary)


if __name__ == "__main__":
    main()
