"""CTI forum scraper — Scrapling-powered workflow for deepdarkCTI forum.md.

Quick-start:

    from etg.scraper.pipeline import run, iter_posts

    # Write all ONLINE clearnet posts to data/real_posts.jsonl
    run()

    # Stream posts for specific forums without writing files
    for raw, post in iter_posts(forum_filter=["bhf", "cracking"]):
        print(post.forum_id, post.timestamp, post.text[:80])
"""

from .forum_index import load_index, online_entries, clearnet_entries, onion_entries
from .models import ForumEntry, RawPost
from .pipeline import run, iter_posts

__all__ = [
    "load_index",
    "online_entries",
    "clearnet_entries",
    "onion_entries",
    "ForumEntry",
    "RawPost",
    "run",
    "iter_posts",
]
