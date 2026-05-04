"""Per-forum-software scraper implementations."""

from .base import BaseScraper
from .discourse import DiscourseScraper
from .generic import GenericScraper
from .invision import InvisionScraper
from .mybb import MyBBScraper
from .phpbb import PhpBBScraper
from .xenforo import XenForoScraper

__all__ = [
    "BaseScraper",
    "DiscourseScraper",
    "GenericScraper",
    "InvisionScraper",
    "MyBBScraper",
    "PhpBBScraper",
    "XenForoScraper",
]
