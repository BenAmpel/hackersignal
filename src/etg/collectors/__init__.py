"""ETG data collectors.

Collector modules have different optional dependencies, so public package
exports are resolved lazily instead of importing every scraper at package
import time.
"""

__all__ = [
    "collect_exploitdb",
    "collect_nvd",
    "collect_packetstorm",
    "collect_fulldisclosure",
    "build_ev_pairs",
]


def __getattr__(name: str):
    if name == "collect_exploitdb":
        from etg.collectors.exploitdb import collect
    elif name == "collect_nvd":
        from etg.collectors.nvd import collect
    elif name == "collect_packetstorm":
        from etg.collectors.packetstorm import collect
    elif name == "collect_fulldisclosure":
        from etg.collectors.fulldisclosure import collect
    elif name == "build_ev_pairs":
        from etg.collectors.ev_builder import build as collect
    else:
        raise AttributeError(name)
    globals()[name] = collect
    return collect
