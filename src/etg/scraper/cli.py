"""Command-line interface for the CTI forum scraper.

Usage examples:

  # Scrape all ONLINE clearnet forums, write to default paths
  python -m etg.scraper.cli

  # Scrape specific forums by name substring
  python -m etg.scraper.cli --forum bhf --forum cracking

  # Include .onion sites (Tor must be running on 127.0.0.1:9050)
  python -m etg.scraper.cli --onion

  # Custom output paths and rate limit
  python -m etg.scraper.cli --output out/posts.jsonl --delay 2.0

  # Print index stats without scraping
  python -m etg.scraper.cli --list-forums

  # Dry run: detect forum software for each ONLINE forum
  python -m etg.scraper.cli --detect-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m etg.scraper.cli",
        description="Download CTI forum posts into the ETG pipeline format.",
    )
    p.add_argument(
        "--output", default="data/real_posts.jsonl",
        help="Minimal ForumPost JSONL path (default: data/real_posts.jsonl)",
    )
    p.add_argument(
        "--raw-output", default="data/raw_posts.jsonl",
        help="Full RawPost metadata JSONL path (default: data/raw_posts.jsonl)",
    )
    p.add_argument(
        "--index", default=None,
        help="Custom forum.md path or URL (default: fetch live from GitHub)",
    )
    p.add_argument(
        "--forum", action="append", dest="forums", default=None,
        metavar="NAME",
        help="Scrape only forums whose name/id contains NAME (repeatable)",
    )
    p.add_argument(
        "--onion", action="store_true",
        help="Include .onion sites (requires Tor on 127.0.0.1:9050)",
    )
    p.add_argument(
        "--tor-proxy", default="socks5://127.0.0.1:9050",
        help="Tor SOCKS5 proxy URL (default: socks5://127.0.0.1:9050)",
    )
    p.add_argument(
        "--max-threads", type=int, default=0,
        help="Max threads per forum (0 = use per-software-type defaults: "
             "10 000 for XenForo/Invision/phpBB/MyBB/Discourse, 2 000 for unknown)",
    )
    p.add_argument(
        "--max-pages", type=int, default=0,
        help="Max thread-list pages per forum (0 = use per-software-type defaults: "
             "500 for structured forums, 100 for unknown)",
    )
    p.add_argument(
        "--delay", type=float, default=0.0,
        help="Seconds between requests (0 = use per-software-type defaults)",
    )
    p.add_argument(
        "--max-posts-per-thread", type=int, default=0,
        help="Cap posts collected from any one thread (0 = unlimited)",
    )
    p.add_argument(
        "--max-thread-pages", type=int, default=0,
        help="Cap paginated pages fetched within a single thread (0 = unlimited / scraper default)",
    )
    p.add_argument(
        "--crawl-mode", default="deep",
        choices=["deep", "broad"],
        help="Crawl strategy: deep exhausts threads more fully; broad caps per-thread collection to reach more unique threads",
    )
    p.add_argument(
        "--forum-profile-file", default=None,
        help="Path to a JSON file with per-forum proxy/cookie/crawl overrides",
    )
    p.add_argument(
        "--no-checkpoint", action="store_true",
        help="Disable checkpoint file (re-scrape everything)",
    )
    p.add_argument(
        "--checkpoint", default="data/.scrape_checkpoint.json",
        help="Checkpoint file path (default: data/.scrape_checkpoint.json)",
    )
    p.add_argument(
        "--list-forums", action="store_true",
        help="Print the ONLINE forum list and exit",
    )
    p.add_argument(
        "--detect-only", action="store_true",
        help="Detect forum software for each ONLINE forum and exit (no scraping)",
    )
    p.add_argument(
        "--cookies",
        default=None,
        metavar="FILE",
        help="Path to a Netscape cookie file for authenticated scraping. "
             "Export from your browser using the 'Cookie-Editor' extension "
             "(Export → Netscape format). Applies to all forums in this run.",
    )
    p.add_argument(
        "--forum-workers", type=int, default=1,
        metavar="N",
        help="Number of forums to scrape in parallel (default: 1). "
             "4–6 is a practical sweet spot; too many risks IP bans.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress scrapling's per-request INFO lines (keeps etg.scraper messages)",
    )
    p.add_argument(
        "--benchmark", action="store_true",
        help="Run a capped live coverage benchmark instead of the full scrape pipeline",
    )
    p.add_argument(
        "--benchmark-output", default="data/deepdarkcti_benchmark.md",
        help="Markdown benchmark report path (default: data/deepdarkcti_benchmark.md)",
    )
    p.add_argument(
        "--benchmark-json", default="data/deepdarkcti_benchmark.json",
        help="JSON benchmark report path (default: data/deepdarkcti_benchmark.json)",
    )
    p.add_argument(
        "--benchmark-thread-urls", type=int, default=3,
        help="Max candidate thread URLs to probe per forum during benchmark (default: 3)",
    )
    p.add_argument(
        "--benchmark-posts", type=int, default=20,
        help="Max qualifying posts to collect per forum during benchmark (default: 20)",
    )
    p.add_argument(
        "--benchmark-timeout", type=int, default=60,
        help="Per-forum wall-clock timeout in seconds for benchmark mode (default: 60)",
    )
    p.add_argument(
        "--corpus-audit", action="store_true",
        help="Audit a raw crawl JSONL and optionally build a filtered/deduped local corpus",
    )
    p.add_argument(
        "--audit-input", default=None,
        help="Raw JSONL input for corpus audit mode (defaults to --raw-output path)",
    )
    p.add_argument(
        "--audit-output", default="data/corpus_audit.md",
        help="Markdown corpus audit output path (default: data/corpus_audit.md)",
    )
    p.add_argument(
        "--audit-json", default="data/corpus_audit.json",
        help="JSON corpus audit output path (default: data/corpus_audit.json)",
    )
    p.add_argument(
        "--filtered-output", default=None,
        help="If set in corpus audit mode, write a filtered/deduped ForumPost JSONL here",
    )
    p.add_argument(
        "--allow-fallback-timestamps", action="store_true",
        help="Keep rows whose timestamp falls back to scraped_at in corpus audit mode",
    )
    p.add_argument(
        "--english-only", action="store_true",
        help="Keep only English-like rows in corpus audit mode",
    )
    p.add_argument(
        "--apply-cti-filter", action="store_true",
        help="Require CTI keyword relevance in corpus audit mode",
    )
    p.add_argument(
        "--min-cti-score", type=int, default=0,
        help="Minimum CTI score when --apply-cti-filter is enabled (default: 0)",
    )
    p.add_argument(
        "--min-tokens", type=int, default=8,
        help="Minimum token length for corpus audit filtering (default: 8)",
    )
    p.add_argument(
        "--max-tokens", type=int, default=3000,
        help="Maximum token length for corpus audit filtering (default: 3000)",
    )
    p.add_argument(
        "--no-near-dedup", action="store_true",
        help="Disable near-duplicate filtering in corpus audit mode",
    )
    return p


def _cmd_list_forums(args) -> None:
    from etg.scraper.forum_index import online_entries, onion_entries, clearnet_entries

    index_source = Path(args.index) if args.index and Path(args.index).exists() else args.index
    entries = online_entries(index_source)
    total = len(entries)
    onion = sum(1 for e in entries if e.is_onion)
    clearnet = total - onion

    print(f"\n{'Forum':<40} {'Type':<8} URL")
    print("-" * 80)
    for e in entries:
        kind = "onion" if e.is_onion else "clear"
        print(f"{e.name:<40} {kind:<8} {e.url}")
    print(f"\nTotal ONLINE: {total}  (clearnet: {clearnet}, onion: {onion})")


def _cmd_detect(args) -> None:
    from etg.scraper.detector import detect, ForumKind
    from etg.scraper.scrapers.base import make_fetcher, resp_html
    from etg.scraper.forum_index import online_entries

    index_source = Path(args.index) if args.index and Path(args.index).exists() else args.index
    entries = online_entries(index_source)
    if not args.onion:
        entries = [e for e in entries if not e.is_onion]
    if args.forums:
        filt = [f.lower() for f in args.forums]
        entries = [
            e for e in entries
            if any(f in e.forum_id.lower() or f in e.name.lower() for f in filt)
        ]

    print(f"\n{'Forum':<40} {'Detected':<12} URL")
    print("-" * 90)
    fetcher = make_fetcher()
    for entry in entries:
        kind = detect(entry.url)
        if kind == ForumKind.UNKNOWN:
            try:
                proxy = args.tor_proxy if entry.is_onion else None
                resp = fetcher.get(entry.url, timeout=15, proxy=proxy)
                kind = detect(entry.url, resp_html(resp))
                time.sleep(args.delay)
            except Exception as exc:
                log.debug("Detection fetch error %s: %s", entry.url, exc)
        print(f"{entry.name:<40} {kind.name:<12} {entry.url}")


def _cmd_benchmark(args) -> None:
    from etg.scraper.benchmark import run_benchmark

    index_source = Path(args.index) if args.index and Path(args.index).exists() else args.index
    results = run_benchmark(
        markdown_output=args.benchmark_output,
        json_output=args.benchmark_json,
        index_source=index_source,
        include_onion=args.onion,
        forum_filter=args.forums,
        tor_proxy=args.tor_proxy,
        max_thread_urls=args.benchmark_thread_urls,
        max_posts=args.benchmark_posts,
        max_list_pages=args.max_pages or 3,
        per_forum_timeout=args.benchmark_timeout,
        request_delay=args.delay or None,
        cookies=args.cookies,
    )
    print(f"Benchmark complete: {len(results)} forums")
    print(f"Markdown: {args.benchmark_output}")
    print(f"JSON: {args.benchmark_json}")


def _cmd_corpus_audit(args) -> None:
    from etg.data.corpus_tools import audit_raw_corpus, build_filtered_corpus, default_corpus_config

    raw_input = Path(args.audit_input or args.raw_output)
    config = default_corpus_config()
    config.require_real_timestamp = not args.allow_fallback_timestamps
    if args.english_only:
        config.min_ascii_ratio = 0.72
        config.use_langdetect = True
    else:
        config.min_ascii_ratio = 0.0
        config.use_langdetect = False
    config.apply_cti_filter = args.apply_cti_filter
    config.min_cti_score = args.min_cti_score
    config.min_tokens = args.min_tokens
    config.max_tokens = args.max_tokens
    config.near_dedup = not args.no_near_dedup

    if args.filtered_output:
        report = build_filtered_corpus(
            raw_input,
            args.filtered_output,
            config=config,
            audit_json=args.audit_json,
            audit_markdown=args.audit_output,
        )
        print(f"Filtered posts: {report['filtered_posts']}")
        print(f"Filtered output: {args.filtered_output}")
    else:
        report = audit_raw_corpus(raw_input)
        Path(args.audit_json).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        from etg.data.corpus_tools import render_audit_markdown
        Path(args.audit_output).write_text(
            render_audit_markdown(report),
            encoding="utf-8",
        )
    print(f"Audit markdown: {args.audit_output}")
    print(f"Audit json: {args.audit_json}")


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.quiet:
        # Suppress scrapling's per-request INFO/WARNING chatter entirely
        logging.getLogger("scrapling").setLevel(logging.ERROR)

    if args.list_forums:
        _cmd_list_forums(args)
        return

    if args.detect_only:
        _cmd_detect(args)
        return

    if args.benchmark:
        _cmd_benchmark(args)
        return

    if args.corpus_audit:
        _cmd_corpus_audit(args)
        return

    from etg.scraper.pipeline import run
    index_source = (
        Path(args.index) if args.index and Path(args.index).exists() else args.index
    )
    run(
        output=args.output,
        raw_output=args.raw_output,
        index_source=index_source,
        include_onion=args.onion,
        forum_filter=args.forums,
        # None = use per-software-type defaults from _KIND_OVERRIDES
        max_threads=args.max_threads or None,
        max_list_pages=args.max_pages or None,
        request_delay=args.delay or None,
        max_posts_per_thread=args.max_posts_per_thread or None,
        max_thread_pages=args.max_thread_pages or None,
        crawl_mode=args.crawl_mode,
        forum_profile_file=args.forum_profile_file,
        checkpoint=None if args.no_checkpoint else args.checkpoint,
        tor_proxy=args.tor_proxy,
        log_level=args.log_level,
        forum_workers=args.forum_workers,
        cookies=args.cookies,
    )


if __name__ == "__main__":
    main()
