"""CAREER Research Task 1 ETG pipeline over HackerSignal.

Research Task 1.1 defines an Exploit Text Graph (ETG) as a cumulative
Graph-of-Words over hacker exploit text: nodes are words used in or before time
spell t, edges are word co-occurrences in posts in or before t, and node
features are word-level attributes. This module implements that initial ETG.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm.auto import tqdm
from scipy.linalg import orthogonal_procrustes
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import TruncatedSVD

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+#.-]{2,}")
CVE_RE = re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE)
STOPWORDS = {
    # articles / determiners
    "the", "and", "for", "are", "was", "were", "has", "had",
    "not", "can", "may", "all", "any", "its", "our",
    # pronouns
    "you", "your", "they", "their", "them", "we", "he", "she",
    "it", "his", "her", "who", "whom", "mine", "yours", "ours",
    "theirs", "himself", "herself", "itself", "yourself", "ourselves",
    "themselves", "anyone", "someone", "nobody", "everybody",
    "something", "anything", "nothing", "everything", "none", "both",
    "each", "every", "few", "many", "most", "some", "such",
    # aux / modal verbs
    "will", "would", "could", "should", "might", "must", "shall",
    "does", "did", "been", "being", "have", "had", "having",
    "get", "got", "getting", "make", "made", "making",
    # common adverbs / conjunctions
    "but", "also", "just", "very", "even", "still", "only",
    "now", "back", "well", "often", "then", "too", "yet",
    "more", "most", "much", "many", "less", "least",
    "about", "after", "before", "because", "between", "through",
    "into", "from", "with", "that", "this", "these", "those",
    "than", "when", "where", "which", "while", "there", "here",
    "over", "under", "again", "same", "other", "another", "either",
    "neither", "however", "therefore", "although", "though",
    "since", "until", "unless", "without", "during", "within",
    "using", "via", "per", "via",
    # common verbs
    "see", "look", "know", "think", "want", "give", "take",
    "come", "say", "tell", "try", "put", "keep", "let",
    "use", "used", "need", "needs", "needed", "said", "says",
    "like", "liked", "seems", "seem", "seems",
    # generic nouns (non-security)
    "thing", "things", "way", "ways", "time", "times",
    "part", "parts", "point", "number", "case", "line",
    "area", "place", "fact", "side", "end", "day", "man",
    "world", "people", "person", "group", "lot", "bit",
    # forum / internet noise
    "https", "http", "url", "click", "expand", "reply", "post",
    "thread", "thanks", "thank", "please", "help", "what",
    "actually", "basically", "literally", "simply", "really",
    "already", "always", "never", "probably", "usually",
    "sometimes", "maybe", "perhaps", "anyway", "instead",
    "together", "along", "around", "between", "behind",
    # misc single-sense words caught in the run
    "die", "ive", "dont", "didnt", "doesnt", "cant", "wont",
    "isnt", "arent", "wasnt", "werent",
    # additional generic words observed in hub terms
    "how", "review", "new", "good", "old", "free", "best",
    "work", "works", "find", "found", "show", "shows",
    "open", "run", "runs", "set", "high", "low",
    "read", "write", "send", "start", "stop", "add", "added",
    "call", "calls", "called", "check", "create", "include",
    "list", "return", "note", "update", "allow", "allows",
    "support", "supports", "change", "changes", "fix", "fixes",
    "buy", "sell", "price", "sale", "deal", "offer",
    "question", "answer", "info", "information",
    "out", "order",
}


PIPELINE_CACHE_VERSION = "career_rt1_cache_v5"


@dataclass(frozen=True)
class HackerSignalPost:
    id: str
    text: str
    timestamp: datetime
    source_dataset: str
    source_layer: str
    forum_id: str


@dataclass(frozen=True)
class RuntimePlan:
    device: str
    mode: str
    data_path: str
    record_limit: int | None
    n_spells: int
    vocab_size: int
    window: int
    min_edge_weight: int
    max_edges_per_spell: int
    dgt_max_nodes: int
    dgt_hidden_dim: int
    dgt_heads: int
    dgt_layers: int
    dgt_epochs: int
    lap_pe_k: int
    dgt_edge_batch: int | None = None  # None = auto-computed from GPU memory at runtime
    min_date: str | None = None  # ISO date string e.g. "2016-01-01"; None = no lower bound
    temporal_loss_weight: float = 0.1  # weight for temporal context loss (0.1 avoids ablation inversion)
    pe_type: str = "laplacian"           # "laplacian" | "rwpe" | "mose" | "none"
    use_residual_bypass: bool = True     # learnable alpha bypass (v4 default on)
    temporal_gate: bool = False          # per-node GRU-style gate replaces global alpha
    use_time_embedding: bool = True      # include spell-index embedding in transformer
    time_encoding: str = "learned_discrete"  # "learned_discrete" | "learned_linear"
    rwpe_attention_bias: bool = False    # add pairwise RWPE dot-product to attn mask
    use_trend_seasonal: bool = False     # TIDFormer trend+seasonal decomposition in temporal loss


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def stable_fingerprint(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def data_file_fingerprint(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def snapshot_fingerprint(snapshots: list[dict]) -> str:
    rows = []
    for snap in snapshots:
        edge_digest = hashlib.sha256(
            json.dumps(sorted(snap["edge_counts"].items())[:25_000], default=_json_default).encode("utf-8")
        ).hexdigest()[:16]
        term_digest = hashlib.sha256(
            json.dumps(sorted(snap["term_counts"].items())[:25_000], default=_json_default).encode("utf-8")
        ).hexdigest()[:16]
        rows.append(
            {
                "spell": snap["spell"],
                "start": snap["start"],
                "end": snap["end"],
                "posts": snap["posts"],
                "active_terms": snap["active_terms"],
                "edges": snap["edges"],
                "edge_weight": snap["edge_weight"],
                "edge_digest": edge_digest,
                "term_digest": term_digest,
            }
        )
    return stable_fingerprint({"version": PIPELINE_CACHE_VERSION, "snapshots": rows})


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_cached_pickle(path: str | Path, expected_fingerprint: str):
    path = Path(path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not path.exists() or not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != expected_fingerprint:
            return None
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def save_cached_pickle(path: str | Path, obj, fingerprint: str, metadata: dict | None = None) -> None:
    path = Path(path)
    _atomic_pickle(path, obj)
    manifest = {"cache_version": PIPELINE_CACHE_VERSION, "fingerprint": fingerprint}
    if metadata:
        manifest.update(metadata)
    _atomic_write_text(path.with_suffix(path.suffix + ".manifest.json"), json.dumps(manifest, indent=2, default=_json_default))


def detect_device() -> str:
    """Prefer CUDA, then MPS, then CPU."""

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def query_gpu_free_memory(device: str) -> int:
    """Return estimated free GPU memory in bytes, or 0 if unknown / CPU."""
    try:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return int(free)
        if device == "mps":
            # Apple Silicon uses unified memory; no direct free-memory query.
            # Use a conservative 4 GB lower bound so auto-sizing stays safe.
            return 4 * 1024 ** 3
    except Exception:
        pass
    return 0


def auto_max_nodes(
    device: str,
    hidden_dim: int,
    heads: int,
    layers: int,
    n_snapshots: int = 12,
    cap: int = 6_000,
) -> int:
    """Estimate the largest *max_nodes* that fits in available GPU memory.

    The dominant cost is the transformer self-attention matrix: O(N²) per
    layer per head in float32, doubled for the backward pass.  A 40 % safety
    margin is applied to leave room for activations, gradients, and the
    optimizer state.
    """
    free = query_gpu_free_memory(device)
    if free == 0:
        return min(2_000, cap)
    budget = int(free * 0.40)
    # float32 attention per forward+backward: heads × layers × N² × 4 bytes × 2
    # bool adjacency masks per snapshot:      n_snapshots × N² × 1 byte
    a = heads * layers * 4 * 2
    b = max(n_snapshots, 1)
    n_sq = budget / max(a + b, 1)
    return max(64, min(int(math.sqrt(n_sq)), cap))


def auto_edge_batch(device: str, hidden_dim: int) -> int:
    """Estimate the number of edge pairs to sample per gradient step.

    Allocates 10 % of free GPU memory for link-prediction embeddings,
    rounded down to the nearest power of two (floor of 65 536).
    """
    free = query_gpu_free_memory(device)
    if free == 0:
        return 4_096
    budget = int(free * 0.10)
    # Per edge pair: 2 embedding vectors × hidden_dim × float32 × ~4 (grad + Adam)
    bytes_per_pair = max(2 * hidden_dim * 4 * 4, 1)
    batch = budget // bytes_per_pair
    batch = max(512, min(batch, 65_536))
    # Round down to nearest power of two for memory-allocation efficiency
    return 1 << (batch.bit_length() - 1)


def choose_runtime_plan(
    *,
    repo_root: str | Path = ".",
    requested_device: str = "auto",
    requested_mode: str = "auto",
) -> RuntimePlan:
    repo_root = Path(repo_root)
    device = detect_device() if requested_device == "auto" else requested_device.lower()
    if device not in {"cuda", "mps", "cpu"}:
        raise ValueError("requested_device must be one of auto, cuda, mps, or cpu")

    full_candidates = [
        repo_root / "ETG_MISQ/data/career_rt1_hackersignal.jsonl.gz",
        repo_root / "ETG_MISQ/data/career_rt1_hackersignal.jsonl",
        repo_root / "data/career_rt1_hackersignal.jsonl.gz",
        repo_root / "data/career_rt1_hackersignal.jsonl",
        repo_root / "data/unified_hacker_communities_neurips.jsonl",
        repo_root / "data/unified_hacker_communities_neurips_public.jsonl",
        repo_root / "data/unified_hacker_communities.jsonl",
        repo_root / "data/hackersignal_sample_10k.jsonl",
    ]
    sample_candidates = [
        repo_root / "ETG_MISQ/data/career_rt1_hackersignal_sample_10k.jsonl",
        repo_root / "data/hackersignal_sample_10k.jsonl",
        repo_root / "ETG_MISQ/data/career_rt1_hackersignal.jsonl.gz",
        repo_root / "ETG_MISQ/data/career_rt1_hackersignal.jsonl",
    ]
    sample_path = next((path for path in sample_candidates if path.exists()), sample_candidates[0])
    full_path = next((path for path in full_candidates if path.exists()), sample_path)

    mode = "full" if requested_mode == "auto" and device == "cuda" else requested_mode.lower()
    if requested_mode == "auto" and device != "cuda":
        mode = "smoke"
    if mode not in {"full", "smoke"}:
        raise ValueError("requested_mode must be one of auto, full, or smoke")

    data_path = full_path if mode == "full" else sample_path
    rel_path = data_path.relative_to(repo_root) if data_path.is_relative_to(repo_root) else data_path

    if mode == "full":
        _heads, _layers, _dim = 4, 2, 128
        computed_max_nodes = auto_max_nodes(
            device,
            hidden_dim=_dim,
            heads=_heads,
            layers=_layers,
            n_snapshots=12,
            cap=6_000,
        )
        print(f"[plan] GPU auto-sized dgt_max_nodes={computed_max_nodes} for device={device}")
        return RuntimePlan(
            device=device,
            mode="full",
            data_path=str(rel_path),
            record_limit=None,
            n_spells=12,
            vocab_size=30_000,
            window=4,
            min_edge_weight=10,
            max_edges_per_spell=300_000,
            dgt_max_nodes=computed_max_nodes,
            dgt_hidden_dim=_dim,
            dgt_heads=_heads,
            dgt_layers=_layers,
            dgt_epochs=200,
            lap_pe_k=16,
            min_date="2016-01-01",
            temporal_loss_weight=0.1,
        )
    return RuntimePlan(
        device=device,
        mode="smoke",
        data_path=str(rel_path),
        record_limit=10_000 if device == "mps" else 2_500,
        n_spells=5,
        vocab_size=900 if device == "mps" else 500,
        window=4,
        min_edge_weight=1,
        max_edges_per_spell=50_000,
        dgt_max_nodes=350 if device == "mps" else 250,
        dgt_hidden_dim=64,
        dgt_heads=4,
        dgt_layers=2,
        dgt_epochs=5,
        lap_pe_k=8,
    )


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(text[:10])
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def stem_token(token: str) -> str:
    """Small deterministic stemmer to satisfy RT1.1 without extra dependencies."""

    t = token.lower()
    for suffix in ("ization", "ational", "fulness", "ousness", "iveness", "tional", "ments", "ment", "ingly", "edly", "ing", "ed", "ies", "s"):
        if len(t) > len(suffix) + 3 and t.endswith(suffix):
            if suffix == "ies":
                return t[: -len(suffix)] + "y"
            return t[: -len(suffix)]
    return t


_FIRMWARE_RE = re.compile(r"^v\d+[a-z]\d+[a-z]\d+", re.IGNORECASE)
_HYPHEN_RE = re.compile(r"[a-z0-9]*-[a-z0-9]+-[a-z0-9]+-[a-z0-9]*", re.IGNORECASE)


def _is_noise_token(token: str) -> bool:
    """Return True if *token* is a firmware version code, product SKU, or other non-semantic noise.

    Filters:
    - Firmware/version strings: v1a2b3c pattern (mixed letter/digit runs)
    - Product codes with ≥2 hyphens: spro-abc-xyz, frank-free-code, etc.
    - Tokens where >50 % of characters are digits (serial numbers, hex blobs)
    """
    if _FIRMWARE_RE.match(token):
        return True
    if _HYPHEN_RE.search(token):
        return True
    digit_ratio = sum(ch.isdigit() for ch in token) / max(len(token), 1)
    if digit_ratio > 0.5:
        return True
    return False


def tokenize(text: str) -> list[str]:
    # rstrip removes trailing punctuation TOKEN_RE can include (dots, dashes, etc.)
    # This prevents "expand..." slipping past the "expand" stopword entry.
    tokens = [stem_token(m.group(0)).rstrip('.-+#_,;:!?') for m in TOKEN_RE.finditer(text)]
    return [t for t in tokens if t not in STOPWORDS and not t.startswith("http") and len(t) >= 3 and not _is_noise_token(t)]


def iter_hackersignal_posts(
    path: str | Path,
    *,
    limit: int | None = None,
    layer_filter: set[str] | None = None,
    desc: str = "Reading posts",
    min_date: str | None = None,
    max_date: str | None = None,
) -> Iterable[HackerSignalPost]:
    seen = 0
    _min_ts: datetime | None = parse_timestamp(min_date) if min_date else None
    _max_ts: datetime | None = parse_timestamp(max_date) if max_date else None
    _path = Path(path)
    _open = gzip.open if _path.suffix == ".gz" else _path.open
    _kwargs: dict = {"mode": "rt", "encoding": "utf-8"} if _path.suffix == ".gz" else {"mode": "r", "encoding": "utf-8"}
    with _open(_path, **_kwargs) as handle:
        _pbar = tqdm(handle, desc=desc, unit=" rec", total=limit, leave=False)
        for line_no, line in enumerate(_pbar, 1):
            if limit is not None and seen >= limit:
                _pbar.close()
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
            text = str(rec.get("text") or "").strip()
            ts = parse_timestamp(rec.get("timestamp"))
            source_layer = str(rec.get("source_layer") or "unknown")
            if not text or ts is None:
                continue
            if _min_ts is not None and ts < _min_ts:
                continue
            if _max_ts is not None and ts > _max_ts:
                continue
            if layer_filter and source_layer not in layer_filter:
                continue
            seen += 1
            yield HackerSignalPost(
                id=str(rec.get("unified_id") or rec.get("id") or ""),
                text=text,
                timestamp=ts,
                source_dataset=str(rec.get("source_dataset") or "unknown"),
                source_layer=source_layer,
                forum_id=str(rec.get("forum_id") or "unknown"),
            )


def load_hackersignal_posts(
    path: str | Path,
    *,
    limit: int | None = None,
    layer_filter: set[str] | None = None,
    min_date: str | None = None,
    max_date: str | None = None,
) -> list[HackerSignalPost]:
    return list(iter_hackersignal_posts(path, limit=limit, layer_filter=layer_filter,
                                        desc="Loading posts", min_date=min_date, max_date=max_date))


def corpus_profile(posts: Iterable[HackerSignalPost]) -> pd.DataFrame:
    rows = []
    for post in tqdm(posts, desc="Profiling posts", unit=" post", leave=False):
        rows.append(
            {
                "source_layer": post.source_layer,
                "source_dataset": post.source_dataset,
                "forum_id": post.forum_id,
                "year": post.timestamp.year,
                "tokens": len(tokenize(post.text)),
                "has_cve": bool(CVE_RE.search(post.text)),
            }
        )
    return pd.DataFrame(rows)


def streaming_corpus_profile(
    path: str | Path,
    *,
    limit: int | None = None,
    layer_filter: set[str] | None = None,
    min_date: str | None = None,
    max_date: str | None = None,
) -> pd.DataFrame:
    groups: dict[tuple[str, str, str], dict] = {}
    for post in iter_hackersignal_posts(path, limit=limit, layer_filter=layer_filter, desc="Profiling corpus",
                                        min_date=min_date, max_date=max_date):
        key = (post.source_layer, post.source_dataset, post.forum_id)
        group = groups.setdefault(
            key,
            {
                "source_layer": post.source_layer,
                "source_dataset": post.source_dataset,
                "forum_id": post.forum_id,
                "records": 0,
                "first_year": post.timestamp.year,
                "last_year": post.timestamp.year,
                "token_sum": 0,
                "cve_hits": 0,
            },
        )
        group["records"] += 1
        group["first_year"] = min(group["first_year"], post.timestamp.year)
        group["last_year"] = max(group["last_year"], post.timestamp.year)
        group["token_sum"] += len(tokenize(post.text))
        group["cve_hits"] += int(bool(CVE_RE.search(post.text)))
    df = pd.DataFrame(groups.values())
    if df.empty:
        return df
    df["mean_tokens"] = (df["token_sum"] / df["records"]).round(1)
    df["cve_hit_rate"] = (100 * df["cve_hits"] / df["records"]).round(1)
    return df.drop(columns=["token_sum", "cve_hits"])


def _trim_counter(counter: Counter, keep: int) -> Counter:
    if keep and len(counter) > keep:
        return Counter(dict(counter.most_common(keep)))
    return counter


def _trigram_hashes(term: str, buckets: int = 2**14, max_features: int = 8) -> str:
    grams = [term[i : i + 3] for i in range(max(1, len(term) - 2))]
    hashed = []
    for gram in grams[:max_features]:
        value = int(hashlib.sha256(gram.encode("utf-8")).hexdigest()[:8], 16) % buckets
        hashed.append(str(value))
    return " ".join(hashed)


def _update_period_stat(
    stat: dict,
    post: HackerSignalPost,
    vocab_set: set[str],
    window: int,
    edge_candidate_cap: int,
) -> None:
    stat["posts"] += 1
    stat["start"] = post.timestamp if stat["start"] is None else min(stat["start"], post.timestamp)
    stat["end"] = post.timestamp if stat["end"] is None else max(stat["end"], post.timestamp)
    stat["source_counts"][post.source_dataset] += 1
    stat["forum_counts"][post.forum_id] += 1
    cves = [cve.upper() for cve in CVE_RE.findall(post.text)]
    stat["cve_mentions"].update(cves)

    tokens = [token for token in tokenize(post.text) if token in vocab_set]
    unique_tokens = set(tokens)
    stat["term_counts"].update(tokens)
    stat["term_doc_counts"].update(unique_tokens)
    for i, src in enumerate(tokens):
        for dst in tokens[i + 1 : i + 1 + window]:
            if src != dst:
                stat["edges"][(src, dst)] += 1
    if len(stat["edges"]) > edge_candidate_cap * 2:
        stat["edges"] = _trim_counter(stat["edges"], edge_candidate_cap)


def _empty_period_stat(spell: int) -> dict:
    return {
        "spell": spell,
        "start": None,
        "end": None,
        "posts": 0,
        "term_counts": Counter(),
        "term_doc_counts": Counter(),
        "edges": Counter(),
        "cve_mentions": Counter(),
        "source_counts": Counter(),
        "forum_counts": Counter(),
    }


def _snapshot_from_cumulative(
    spell: int,
    cumulative: dict,
    min_edge_weight: int,
    max_edges_per_spell: int,
) -> dict:
    edges = Counter({edge: weight for edge, weight in cumulative["edges"].items() if weight >= min_edge_weight})
    edges = _trim_counter(edges, max_edges_per_spell)

    in_degree: Counter[str] = Counter()
    out_degree: Counter[str] = Counter()
    weighted_in: Counter[str] = Counter()
    weighted_out: Counter[str] = Counter()
    for (src, dst), weight in edges.items():
        out_degree[src] += 1
        in_degree[dst] += 1
        weighted_out[src] += weight
        weighted_in[dst] += weight

    term_counts = cumulative["term_counts"]
    active_terms = set(term_counts)
    density = len(edges) / max(len(active_terms) * max(len(active_terms) - 1, 1), 1)
    return {
        "spell": spell,
        "start": cumulative["start"].date().isoformat(),
        "end": cumulative["end"].date().isoformat(),
        "posts": cumulative["posts"],
        "active_terms": len(active_terms),
        "active_cves_observed": len(cumulative["cve_mentions"]),
        "edges": len(edges),
        "edge_weight": int(sum(edges.values())),
        "density": float(density),
        "top_terms": term_counts.most_common(12),
        "top_cves": cumulative["cve_mentions"].most_common(8),
        "top_edges": edges.most_common(10),
        "top_sources": cumulative["source_counts"].most_common(10),
        "top_forums": cumulative["forum_counts"].most_common(10),
        "term_counts": dict(term_counts),
        "term_doc_counts": dict(cumulative["term_doc_counts"]),
        "edge_counts": dict(edges),
        "cve_mentions": dict(cumulative["cve_mentions"]),
        "in_degree": dict(in_degree),
        "out_degree": dict(out_degree),
        "weighted_in_degree": dict(weighted_in),
        "weighted_out_degree": dict(weighted_out),
    }


def build_vocabulary(posts: Iterable[HackerSignalPost], vocab_size: int) -> list[str]:
    counts: Counter[str] = Counter()
    for post in tqdm(posts, desc="Building vocab", unit=" post", leave=False):
        counts.update(set(tokenize(post.text)))
    return [term for term, _ in counts.most_common(vocab_size)]


def assign_time_spells(posts: list[HackerSignalPost], n_spells: int) -> list[list[HackerSignalPost]]:
    ordered = sorted(posts, key=lambda p: p.timestamp)
    chunks = np.array_split(np.arange(len(ordered)), n_spells)
    return [[ordered[int(i)] for i in chunk] for chunk in chunks if len(chunk)]


def build_spell_etgs(
    posts: list[HackerSignalPost],
    *,
    n_spells: int,
    vocab_size: int,
    window: int,
    min_edge_weight: int,
    max_edges_per_spell: int,
    cache_path: str | Path | None = None,
    cache_metadata: dict | None = None,
) -> tuple[list[dict], list[str]]:
    """Build cumulative RT1.1 ETGs from in-memory posts."""

    fingerprint = stable_fingerprint(
        {
            "stage": "build_spell_etgs",
            "version": PIPELINE_CACHE_VERSION,
            "post_count": len(posts),
            "first_ts": min((p.timestamp for p in posts), default=None),
            "last_ts": max((p.timestamp for p in posts), default=None),
            "post_ids_digest": hashlib.sha256("|".join(p.id for p in posts[:50_000]).encode("utf-8")).hexdigest()[:16],
            "n_spells": n_spells,
            "vocab_size": vocab_size,
            "window": window,
            "min_edge_weight": min_edge_weight,
            "max_edges_per_spell": max_edges_per_spell,
        }
    )
    if cache_path:
        cached = load_cached_pickle(cache_path, fingerprint)
        if cached is not None:
            print(f"[cache] loaded RT1.1 ETG snapshots from {cache_path}")
            return cached

    vocab = build_vocabulary(posts, vocab_size)
    vocab_set = set(vocab)
    edge_candidate_cap = max_edges_per_spell * 3
    periods = []
    for spell_id, spell_posts in tqdm(enumerate(assign_time_spells(posts, n_spells), 1), desc="Building ETG spells", total=n_spells, unit=" spell"):
        stat = _empty_period_stat(spell_id)
        for post in tqdm(spell_posts, desc=f"  Spell {spell_id}", unit=" post", leave=False):
            _update_period_stat(stat, post, vocab_set, window, edge_candidate_cap)
        periods.append(stat)
    result = _cumulative_snapshots(periods, vocab, min_edge_weight, max_edges_per_spell)
    if cache_path:
        save_cached_pickle(cache_path, result, fingerprint, cache_metadata)
        print(f"[cache] saved RT1.1 ETG snapshots to {cache_path}")
    return result


def build_spell_etgs_streaming(
    path: str | Path,
    *,
    n_spells: int,
    vocab_size: int,
    window: int,
    min_edge_weight: int,
    max_edges_per_spell: int,
    limit: int | None = None,
    layer_filter: set[str] | None = None,
    min_date: str | None = None,
    max_date: str | None = None,
    cache_path: str | Path | None = None,
    cache_metadata: dict | None = None,
) -> tuple[list[dict], list[str]]:
    """Build cumulative RT1.1 ETGs from a full HackerSignal JSONL."""

    fingerprint = stable_fingerprint(
        {
            "stage": "build_spell_etgs_streaming",
            "version": PIPELINE_CACHE_VERSION,
            "data_file": data_file_fingerprint(path),
            "n_spells": n_spells,
            "vocab_size": vocab_size,
            "window": window,
            "min_edge_weight": min_edge_weight,
            "max_edges_per_spell": max_edges_per_spell,
            "limit": limit,
            "layer_filter": layer_filter,
            "min_date": min_date,
            "max_date": max_date,
        }
    )
    if cache_path:
        cached = load_cached_pickle(cache_path, fingerprint)
        if cached is not None:
            print(f"[cache] loaded RT1.1 ETG snapshots from {cache_path}")
            return cached

    vocab_counts: Counter[str] = Counter()
    min_ts: datetime | None = None
    max_ts: datetime | None = None
    n_posts = 0
    for post in iter_hackersignal_posts(path, limit=limit, layer_filter=layer_filter,
                                         desc="Vocab pass (1/2)", min_date=min_date, max_date=max_date):
        n_posts += 1
        vocab_counts.update(set(tokenize(post.text)))
        min_ts = post.timestamp if min_ts is None else min(min_ts, post.timestamp)
        max_ts = post.timestamp if max_ts is None else max(max_ts, post.timestamp)
    if n_posts == 0 or min_ts is None or max_ts is None:
        result = ([], [])
        if cache_path:
            save_cached_pickle(cache_path, result, fingerprint, cache_metadata)
        return result

    vocab = [term for term, _ in vocab_counts.most_common(vocab_size)]
    vocab_set = set(vocab)
    span_seconds = max((max_ts - min_ts).total_seconds(), 1.0)
    periods = [_empty_period_stat(i + 1) for i in range(n_spells)]
    edge_candidate_cap = max_edges_per_spell * 3

    for post in iter_hackersignal_posts(path, limit=limit, layer_filter=layer_filter,
                                         desc="ETG pass (2/2)", min_date=min_date, max_date=max_date):
        offset = (post.timestamp - min_ts).total_seconds() / span_seconds
        spell_idx = min(n_spells - 1, max(0, int(offset * n_spells)))
        _update_period_stat(periods[spell_idx], post, vocab_set, window, edge_candidate_cap)

    periods = [p for p in periods if p["posts"]]
    result = _cumulative_snapshots(periods, vocab, min_edge_weight, max_edges_per_spell)
    if cache_path:
        save_cached_pickle(cache_path, result, fingerprint, cache_metadata)
        print(f"[cache] saved RT1.1 ETG snapshots to {cache_path}")
    return result


def _cumulative_snapshots(
    periods: list[dict],
    vocab: list[str],
    min_edge_weight: int,
    max_edges_per_spell: int,
) -> tuple[list[dict], list[str]]:
    cumulative = _empty_period_stat(0)
    snapshots = []
    edge_candidate_cap = max_edges_per_spell * 3
    for period in tqdm(periods, desc="Accumulating spells", unit=" spell", leave=False):
        cumulative["spell"] = period["spell"]
        cumulative["posts"] += period["posts"]
        cumulative["start"] = period["start"] if cumulative["start"] is None else min(cumulative["start"], period["start"])
        cumulative["end"] = period["end"] if cumulative["end"] is None else max(cumulative["end"], period["end"])
        for key in ("term_counts", "term_doc_counts", "edges", "cve_mentions", "source_counts", "forum_counts"):
            cumulative[key].update(period[key])
        cumulative["edges"] = _trim_counter(cumulative["edges"], edge_candidate_cap)
        snapshots.append(_snapshot_from_cumulative(period["spell"], cumulative, min_edge_weight, max_edges_per_spell))
    return snapshots, vocab


def etg_stats_frame(snapshots: list[dict]) -> pd.DataFrame:
    rows = []
    for snap in snapshots:
        rows.append(
            {
                "spell": snap["spell"],
                "cumulative_date_range": f"{snap['start']} to {snap['end']}",
                "cumulative_posts": snap["posts"],
                "word_nodes": snap["active_terms"],
                "edges": snap["edges"],
                "edge_weight": snap["edge_weight"],
                "density": round(snap["density"], 6),
                "cves_observed": snap["active_cves_observed"],
                "top_words": ", ".join(term for term, _ in snap["top_terms"][:6]),
                "top_cves_observed": ", ".join(cve for cve, _ in snap["top_cves"][:4]),
            }
        )
    return pd.DataFrame(rows)


def temporal_shift_terms(snapshots: list[dict], *, top_k: int = 12, min_count: int = 5) -> pd.DataFrame:
    """Rank words by largest normalized frequency increase between cumulative ETGs."""

    rows = []
    for prev, curr in zip(snapshots, snapshots[1:]):
        prev_counts = Counter(prev["term_counts"])
        curr_counts = Counter(curr["term_counts"])
        prev_total = sum(prev_counts.values()) or 1
        curr_total = sum(curr_counts.values()) or 1
        scored = []
        for term in set(prev_counts) | set(curr_counts):
            if prev_counts[term] + curr_counts[term] < min_count:
                continue
            prev_rate = prev_counts[term] / prev_total
            curr_rate = curr_counts[term] / curr_total
            lift = float(np.log2((curr_rate + 1e-7) / (prev_rate + 1e-7)))
            scored.append((term, lift, prev_counts[term], curr_counts[term]))
        for term, lift, prev_count, curr_count in sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]:
            rows.append(
                {
                    "transition": f"G{prev['spell']} to G{curr['spell']}",
                    "word": term,
                    "log2_lift": round(lift, 3),
                    "previous_cumulative_count": int(prev_count),
                    "current_cumulative_count": int(curr_count),
                }
            )
    return pd.DataFrame(rows)


def write_etg_artifacts(snapshots: list[dict], output_dir: str | Path) -> dict:
    """Write cumulative word-node and word-cooccurrence edge tables."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"graph_definition": "RT1.1 cumulative Graph-of-Words ETG", "spells": []}

    for snap in tqdm(snapshots, desc="Writing ETG artifacts", unit=" spell", leave=False):
        spell = snap["spell"]
        nodes_path = output_dir / f"G_{spell:02d}_word_nodes.csv"
        edges_path = output_dir / f"G_{spell:02d}_word_edges.csv"

        node_rows = []
        for term, count in snap["term_counts"].items():
            out_deg = snap["out_degree"].get(term, 0)
            in_deg = snap["in_degree"].get(term, 0)
            node_rows.append(
                {
                    "word": term,
                    "count": count,
                    "doc_count": snap["term_doc_counts"].get(term, 0),
                    "in_degree": in_deg,
                    "out_degree": out_deg,
                    "degree": in_deg + out_deg,
                    "weighted_in_degree": snap["weighted_in_degree"].get(term, 0),
                    "weighted_out_degree": snap["weighted_out_degree"].get(term, 0),
                    "length": len(term),
                    "has_digit": int(any(ch.isdigit() for ch in term)),
                    "has_symbol": int(any(not ch.isalnum() for ch in term)),
                    "trigram_hashes": _trigram_hashes(term),
                    "spell": spell,
                }
            )

        edge_rows = [
            {"source_word": src, "target_word": dst, "weight": weight, "spell": spell}
            for (src, dst), weight in snap["edge_counts"].items()
        ]

        pd.DataFrame(node_rows).to_csv(nodes_path, index=False)
        pd.DataFrame(edge_rows).to_csv(edges_path, index=False)
        manifest["spells"].append(
            {
                "spell": spell,
                "nodes_path": str(nodes_path),
                "edges_path": str(edges_path),
                "word_nodes": len(node_rows),
                "edges": len(edge_rows),
                "cumulative_posts": snap["posts"],
                "cumulative_date_range": [snap["start"], snap["end"]],
            }
        )

    manifest_path = output_dir / "etg_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _select_dgt_vocab(snapshots: list[dict], max_nodes: int) -> list[str]:
    final_counts = Counter(snapshots[-1]["term_counts"])
    return [term for term, _ in final_counts.most_common(max_nodes)]


def _laplacian_pe(edge_counts: dict, node_to_idx: dict[str, int], k: int) -> np.ndarray:
    n = len(node_to_idx)
    if n == 0 or k <= 0:
        return np.zeros((n, 0), dtype=np.float32)
    rows, cols, data = [], [], []
    for (src, dst), weight in edge_counts.items():
        if src in node_to_idx and dst in node_to_idx:
            i, j = node_to_idx[src], node_to_idx[dst]
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([float(weight), float(weight)])
    if not data:
        return np.zeros((n, k), dtype=np.float32)
    adj = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    degree = np.asarray(adj.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree, dtype=np.float64)
    mask = degree > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])
    norm_adj = sparse.diags(inv_sqrt) @ adj @ sparse.diags(inv_sqrt)
    lap = sparse.eye(n, dtype=np.float64) - norm_adj
    kk = min(k + 1, max(1, n - 1))
    try:
        _, vecs = eigsh(lap, k=kk, which="SM")
        pe = vecs[:, 1 : k + 1] if vecs.shape[1] > 1 else vecs[:, :0]
    except Exception:
        pe = np.zeros((n, 0), dtype=np.float32)
    if pe.shape[1] < k:
        pe = np.pad(pe, ((0, 0), (0, k - pe.shape[1])))
    return pe.astype(np.float32)


def _rwpe(edge_counts: dict, node_to_idx: dict[str, int], k: int) -> np.ndarray:
    """Random Walk Positional Encoding (GraphGPS, NeurIPS 2022).

    For each node i, computes the k-step landing probabilities
    [(D^{-1}A)^s]_{ii} for s=1..k.  These diagonal entries measure how
    likely a random walk starting at i returns to i after exactly s steps,
    capturing multi-scale local structure without an eigenvector solve.
    """
    n = len(node_to_idx)
    if n == 0 or k <= 0:
        return np.zeros((n, 0), dtype=np.float32)
    rows, cols, data = [], [], []
    for (src, dst), weight in edge_counts.items():
        if src in node_to_idx and dst in node_to_idx:
            i, j = node_to_idx[src], node_to_idx[dst]
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([float(weight), float(weight)])
    if not data:
        return np.zeros((n, k), dtype=np.float32)
    adj = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr().astype(np.float64)
    degree = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.zeros_like(degree)
    mask = degree > 0
    inv[mask] = 1.0 / degree[mask]
    # Row-stochastic transition matrix P = D^{-1}A
    P = sparse.diags(inv) @ adj
    pe = np.zeros((n, k), dtype=np.float32)
    Ps = P.copy()
    for s in range(k):
        diag_vals = np.array(Ps.tocsr().diagonal(), dtype=np.float32)
        pe[:, s] = diag_vals
        if s < k - 1:
            Ps = Ps @ P
    return pe


def _mose_features(edge_counts: dict, node_to_idx: dict[str, int]) -> np.ndarray:
    """Motif Structural Encoding (MoSE, ICLR 2025, arxiv:2410.18676).

    Computes three cheap motif counts per node:
      col 0 — star count   = degree (number of incident edges)
      col 1 — triangle count = number of closed triangles through node
      col 2 — path-2 count  = number of length-2 paths through node

    These homomorphism counts capture topology beyond degree alone.
    """
    n = len(node_to_idx)
    if n == 0:
        return np.zeros((n, 3), dtype=np.float32)

    adj_sets: dict[int, set[int]] = {i: set() for i in range(n)}
    for (src, dst) in edge_counts:
        if src in node_to_idx and dst in node_to_idx:
            i, j = node_to_idx[src], node_to_idx[dst]
            adj_sets[i].add(j)
            adj_sets[j].add(i)

    features = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        nbrs = adj_sets[i]
        deg_i = len(nbrs)
        features[i, 0] = float(deg_i)
        tri = 0
        nbr_list = list(nbrs)
        for idx_j, j in enumerate(nbr_list):
            for kk in nbr_list[idx_j + 1:]:
                if kk in adj_sets[j]:
                    tri += 1
        features[i, 1] = float(tri)
        path2 = sum(len(adj_sets[j]) for j in nbrs) - deg_i
        features[i, 2] = float(max(path2, 0))

    return features


def _build_dgt_tensors(
    snapshots: list[dict],
    max_nodes: int,
    lap_pe_k: int,
    vocab: list[str] | None = None,
    pe_type: str = "laplacian",
) -> dict:
    words = vocab if vocab is not None else _select_dgt_vocab(snapshots, max_nodes)
    node_to_idx = {word: i for i, word in enumerate(words)}
    features, adjacencies = [], []
    for snap in tqdm(snapshots, desc="Building DGT tensors", unit=" spell", leave=False):
        if pe_type == "laplacian":
            pe = _laplacian_pe(snap["edge_counts"], node_to_idx, lap_pe_k)
        elif pe_type == "rwpe":
            pe = _rwpe(snap["edge_counts"], node_to_idx, lap_pe_k)
        elif pe_type == "mose":
            pe = _mose_features(snap["edge_counts"], node_to_idx)
        else:  # "none"
            pe = np.zeros((len(words), 0), dtype=np.float32)
        x = []
        for word in words:
            count = snap["term_counts"].get(word, 0)
            doc_count = snap["term_doc_counts"].get(word, 0)
            in_deg = snap["in_degree"].get(word, 0)
            out_deg = snap["out_degree"].get(word, 0)
            win = snap["weighted_in_degree"].get(word, 0)
            wout = snap["weighted_out_degree"].get(word, 0)
            x.append(
                [
                    np.log1p(count),
                    np.log1p(doc_count),
                    np.log1p(in_deg),
                    np.log1p(out_deg),
                    np.log1p(win),
                    np.log1p(wout),
                    len(word) / 32.0,
                    float(any(ch.isdigit() for ch in word)),
                    float(any(not ch.isalnum() for ch in word)),
                ]
            )
        base = np.asarray(x, dtype=np.float32)
        features.append(np.concatenate([base, pe], axis=1))

        adj = np.eye(len(words), dtype=bool)
        for (src, dst), _ in snap["edge_counts"].items():
            if src in node_to_idx and dst in node_to_idx:
                adj[node_to_idx[src], node_to_idx[dst]] = True
                adj[node_to_idx[dst], node_to_idx[src]] = True
        adjacencies.append(adj)
    return {"words": words, "features": features, "adjacencies": adjacencies}


def _sample_edges(edge_counts: dict, node_to_idx: dict[str, int], n_neg: int, rng: np.random.Generator, *, max_pos: int = 4_096):
    positives = [(node_to_idx[s], node_to_idx[d]) for (s, d), _ in edge_counts.items() if s in node_to_idx and d in node_to_idx]
    if not positives:
        return [], []
    if len(positives) > max_pos:
        keep = rng.choice(len(positives), size=max_pos, replace=False)
        positives = [positives[int(i)] for i in keep]
    max_neg = max_pos * n_neg
    pos_set = set(positives)
    negatives = []
    n_nodes = len(node_to_idx)
    attempts = 0
    while len(negatives) < min(len(positives) * n_neg, max_neg) and attempts < len(positives) * n_neg * 20:
        src = int(rng.integers(0, n_nodes))
        dst = int(rng.integers(0, n_nodes))
        attempts += 1
        if src != dst and (src, dst) not in pos_set:
            negatives.append((src, dst))
    return positives, negatives


def run_dgt_pipeline(
    snapshots: list[dict],
    *,
    max_nodes: int,
    hidden_dim: int,
    heads: int,
    layers: int,
    epochs: int,
    lap_pe_k: int,
    device: str = "auto",
    output_dir: str | Path | None = None,
    seed: int = 1729,
    use_cache: bool = True,
    edge_batch: int | None = None,
    temporal_loss_weight: float = 0.1,
) -> dict:
    """Train the RT1.2 Diachronic Graph Transformer approximation.

    The stage implements the proposal components in a compact, scalable form:
    Laplacian positional encodings, masked multi-head self-attention over ETG
    neighborhoods, temporal projection/context loss, and semantic-shift
    detection/prediction from diachronic embeddings.
    """

    cache_fingerprint = stable_fingerprint(
        {
            "stage": "run_dgt_pipeline",
            "version": PIPELINE_CACHE_VERSION,
            "snapshots": snapshot_fingerprint(snapshots),
            "max_nodes": max_nodes,
            "hidden_dim": hidden_dim,
            "heads": heads,
            "layers": layers,
            "epochs": epochs,
            "lap_pe_k": lap_pe_k,
            "device_request": device,
            "seed": seed,
            "temporal_loss_weight": temporal_loss_weight,
        }
    )
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    if out is not None and use_cache:
        manifest_path = out / "dgt_manifest.json"
        required = [
            out / "dgt_training_history.csv",
            out / "dgt_semantic_shifts.csv",
            out / "dgt_shift_predictions.csv",
            out / "dgt_embeddings.npz",
            out / "dgt_metrics.json",
        ]
        if manifest_path.exists() and all(path.exists() for path in required):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("fingerprint") == cache_fingerprint:
                    emb_npz = np.load(out / "dgt_embeddings.npz", allow_pickle=True)
                    keys = sorted(key for key in emb_npz.files if key.startswith("G_"))
                    print(f"[cache] loaded RT1.2 DGT outputs from {out}")
                    return {
                        "words": list(emb_npz["words"]),
                        "embeddings": [emb_npz[key] for key in keys],
                        "history": pd.read_csv(out / "dgt_training_history.csv").to_dict(orient="records"),
                        "shifts": pd.read_csv(out / "dgt_semantic_shifts.csv"),
                        "predictions": pd.read_csv(out / "dgt_shift_predictions.csv"),
                        "metrics": json.loads((out / "dgt_metrics.json").read_text(encoding="utf-8")),
                    }
            except Exception:
                pass

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    chosen = detect_device() if device == "auto" else device
    if chosen == "cuda" and torch.cuda.is_available():
        torch_device = torch.device("cuda")
    elif chosen == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch_device = torch.device("mps")
    else:
        torch_device = torch.device("cpu")

    rng = np.random.default_rng(seed)
    tensors = _build_dgt_tensors(snapshots, max_nodes=max_nodes, lap_pe_k=lap_pe_k)
    words = tensors["words"]
    node_to_idx = {w: i for i, w in enumerate(words)}
    if not words:
        raise ValueError("DGT requires at least one ETG node")

    input_dim = tensors["features"][0].shape[1]

    class MaskedGraphTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(input_dim, hidden_dim)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 2,
                dropout=0.2,          # v4: increased from 0.1 for better generalization
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.time_embedding = nn.Embedding(len(snapshots), hidden_dim)
            self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            # v4: learnable residual bypass — alpha=sigmoid(0)=0.5 initially.
            # Allows the model to interpolate between full-attention output and
            # the skip-connected projected input, reducing embedding volatility.
            self.residual_alpha = nn.Parameter(torch.tensor(0.0))

        def forward(self, x, adjacency, t):
            h = self.input(x) + self.time_embedding(torch.tensor(t, device=x.device)).view(1, -1)
            attn_mask = torch.tensor(~adjacency, dtype=torch.bool, device=x.device)
            z_enc = self.encoder(h.unsqueeze(0), mask=attn_mask).squeeze(0)
            alpha = torch.sigmoid(self.residual_alpha)
            z = alpha * z_enc + (1.0 - alpha) * h   # bypass reduces inter-spell embedding jumps
            return F.normalize(self.proj(z), dim=1)

    model = MaskedGraphTransformer().to(torch_device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Warmup + cosine annealing: 10 % linear warmup, then cosine decay to eta_min=1e-5
    _warmup_epochs = max(1, int(epochs * 0.10))
    _eta_min_ratio = 1e-5 / 1e-3  # eta_min / base_lr

    def _lr_lambda(epoch: int) -> float:
        if epoch < _warmup_epochs:
            return float(epoch + 1) / float(_warmup_epochs)
        progress = (epoch - _warmup_epochs) / max(epochs - _warmup_epochs, 1)
        return _eta_min_ratio + 0.5 * (1.0 - _eta_min_ratio) * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
    bce = nn.BCEWithLogitsLoss()
    feat_tensors = [torch.tensor(x, dtype=torch.float32, device=torch_device) for x in tensors["features"]]

    # ── Auto edge-batch size ─────────────────────────────────────────────────
    actual_edge_batch = edge_batch if edge_batch is not None else auto_edge_batch(chosen, hidden_dim)
    print(f"[DGT] device={torch_device}  nodes={len(words)}  edge_batch={actual_edge_batch}  epochs={epochs}")

    # ── Epoch-level checkpoint helpers ───────────────────────────────────────
    ckpt_pt   = out / "dgt_checkpoint.pt"   if out is not None else None
    ckpt_meta = out / "dgt_checkpoint.json" if out is not None else None
    ckpt_hist = out / "dgt_checkpoint_history.json" if out is not None else None

    history: list[dict] = []

    def _save_ckpt(epoch_done: int) -> None:
        if ckpt_pt is None:
            return
        tmp = ckpt_pt.with_suffix(".pt.tmp")
        torch.save({
            "epoch": epoch_done,
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "sched_state": scheduler.state_dict(),
        }, tmp)
        tmp.replace(ckpt_pt)
        _atomic_write_text(ckpt_hist, json.dumps(history, indent=2))
        _atomic_write_text(ckpt_meta, json.dumps({"fingerprint": cache_fingerprint, "epoch": epoch_done, "epochs_total": epochs}, indent=2))

    # ── Resume from checkpoint if fingerprint matches ─────────────────────────
    start_epoch = 0
    if ckpt_pt is not None and ckpt_pt.exists() and ckpt_meta is not None and ckpt_meta.exists():
        try:
            meta = json.loads(ckpt_meta.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == cache_fingerprint:
                ckpt = torch.load(ckpt_pt, map_location=torch_device, weights_only=False)
                model.load_state_dict(ckpt["model_state"])
                opt.load_state_dict(ckpt["opt_state"])
                if "sched_state" in ckpt:
                    scheduler.load_state_dict(ckpt["sched_state"])
                    # fast-forward cosine schedule to match resumed epoch
                else:
                    for _ in range(start_epoch):
                        scheduler.step()
                if ckpt_hist is not None and ckpt_hist.exists():
                    history.extend(json.loads(ckpt_hist.read_text(encoding="utf-8")))
                start_epoch = int(meta.get("epoch", 0))
                print(f"[checkpoint] Resuming DGT from epoch {start_epoch}/{epochs}")
        except Exception as exc:
            print(f"[checkpoint] Load failed ({exc!r}), restarting from epoch 0")
            history.clear()
            start_epoch = 0

    _epoch_bar = tqdm(range(start_epoch, epochs), desc="DGT training", unit=" epoch",
                      initial=start_epoch, total=epochs)
    for epoch in _epoch_bar:
        opt.zero_grad()
        embeddings = []
        loss = torch.tensor(0.0, device=torch_device)
        for t, snap in enumerate(snapshots):
            emb = model(feat_tensors[t], tensors["adjacencies"][t], t)
            embeddings.append(emb)
            pos, neg = _sample_edges(snap["edge_counts"], node_to_idx, n_neg=1, rng=rng, max_pos=actual_edge_batch)
            if pos and neg:
                pos_idx = torch.tensor(pos, dtype=torch.long, device=torch_device)
                neg_idx = torch.tensor(neg, dtype=torch.long, device=torch_device)
                pos_score = (emb[pos_idx[:, 0]] * emb[pos_idx[:, 1]]).sum(dim=1)
                neg_score = (emb[neg_idx[:, 0]] * emb[neg_idx[:, 1]]).sum(dim=1)
                scores = torch.cat([pos_score, neg_score])
                labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
                loss = loss + bce(scores, labels)
        temporal_loss = torch.tensor(0.0, device=torch_device)
        for t in range(1, len(embeddings)):
            current = embeddings[t]
            past = torch.stack(embeddings[:t], dim=0)          # shape (t, N, D)
            n_past = past.shape[0]
            # Recency-weighted EMA: exponential decay so the most-recent snapshot
            # contributes the most; decay factor 0.5 per step back in time.
            decay = torch.exp(
                -0.5 * torch.arange(n_past - 1, -1, -1, dtype=torch.float32, device=torch_device)
            )  # shape (t,)  — largest weight at index -1 (most recent past)
            ema_weights = (decay / decay.sum()).view(-1, 1, 1)
            context = (ema_weights * past).sum(dim=0)           # weighted mean of past
            temporal_loss = temporal_loss + F.mse_loss(current, context.detach())
        total = loss + temporal_loss_weight * temporal_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()
        entry = {"epoch": epoch + 1, "loss": float(total.detach().cpu()), "temporal_loss": float(temporal_loss.detach().cpu())}
        history.append(entry)
        _save_ckpt(epoch + 1)
        _epoch_bar.set_postfix(
            loss=f"{entry['loss']:.4f}",
            temporal=f"{entry['temporal_loss']:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

    model.eval()
    with torch.no_grad():
        final_embeddings = [
            model(feat_tensors[t], tensors["adjacencies"][t], t).detach().cpu().numpy()
            for t in range(len(snapshots))
        ]

    shift_rows = []
    shift_series_by_word: dict[str, list[float]] = {w: [] for w in words}
    for t in range(1, len(final_embeddings)):
        prev, curr = final_embeddings[t - 1], final_embeddings[t]
        cos = np.sum(prev * curr, axis=1) / (np.linalg.norm(prev, axis=1) * np.linalg.norm(curr, axis=1) + 1e-9)
        dist = 1.0 - cos
        for idx, value in enumerate(dist):
            shift_series_by_word[words[idx]].append(float(value))
        top_idx = np.argsort(dist)[-50:][::-1]
        for idx in top_idx:
            shift_rows.append({"transition": f"G{t} to G{t+1}", "word": words[idx], "cosine_shift": float(dist[idx])})
    shifts = pd.DataFrame(shift_rows)

    def _ema_predict(shift_series: list[float], decay: float = 0.5) -> float:
        """Predict next cosine-shift value using a recency-weighted EMA.

        Replaces linear extrapolation (np.polyfit) which produced negative
        predictions (invalid for cosine distance) and poor R².  EMA gives
        smooth, non-negative forecasts anchored to the recent trend.
        """
        n = len(shift_series)
        if n == 0:
            return 0.0
        weights = np.array([decay ** (n - 1 - i) for i in range(n)], dtype=np.float64)
        weights /= weights.sum()
        return float(np.dot(weights, shift_series))

    pred_rows = []
    for word, series in shift_series_by_word.items():
        if len(series) < 2:
            continue
        actual = series[-1]
        # v4: EMA prediction from the training window (all but last spell).
        # Clipped to [0, inf) — cosine distances are non-negative by definition.
        pred = max(0.0, _ema_predict(series[:-1]))
        pred_rows.append({"word": word, "predicted_next_shift": float(pred), "heldout_shift": float(actual), "abs_error": abs(float(pred) - float(actual))})
    predictions = pd.DataFrame(pred_rows).sort_values("predicted_next_shift", ascending=False)
    if not predictions.empty:
        _preds = predictions["predicted_next_shift"].to_numpy()
        _actual = predictions["heldout_shift"].to_numpy()
        _ss_res = float(np.sum((_preds - _actual) ** 2))
        _ss_tot = float(np.sum((_actual - _actual.mean()) ** 2))
        _r2 = 1.0 - _ss_res / _ss_tot if _ss_tot > 0 else float("nan")
        _rmse = float(np.sqrt(np.mean((_preds - _actual) ** 2)))
    else:
        _r2 = _rmse = None
    metrics = {
        "embedding_words": len(words),
        "epochs": epochs,
        "device": str(torch_device),
        "final_loss": history[-1]["loss"] if history else None,
        "shift_prediction_mae": float(predictions["abs_error"].mean()) if not predictions.empty else None,
        "shift_prediction_rmse": _rmse,
        "shift_prediction_r2": _r2,
    }

    if out is not None:
        pd.DataFrame(history).to_csv(out / "dgt_training_history.csv", index=False)
        shifts.to_csv(out / "dgt_semantic_shifts.csv", index=False)
        predictions.to_csv(out / "dgt_shift_predictions.csv", index=False)
        np.savez_compressed(out / "dgt_embeddings.npz", words=np.array(words), **{f"G_{i+1:02d}": emb for i, emb in enumerate(final_embeddings)})
        (out / "dgt_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _atomic_write_text(
            out / "dgt_manifest.json",
            json.dumps(
                {
                    "cache_version": PIPELINE_CACHE_VERSION,
                    "fingerprint": cache_fingerprint,
                    "stage": "run_dgt_pipeline",
                    "snapshots": snapshot_fingerprint(snapshots),
                    "params": {
                        "max_nodes": max_nodes,
                        "hidden_dim": hidden_dim,
                        "heads": heads,
                        "layers": layers,
                        "epochs": epochs,
                        "lap_pe_k": lap_pe_k,
                        "device": device,
                        "seed": seed,
                        "edge_batch": actual_edge_batch,
                        "temporal_loss_weight": temporal_loss_weight,
                    },
                },
                indent=2,
            ),
        )
        print(f"[cache] saved RT1.2 DGT outputs to {out}")
        # Remove epoch-level checkpoint now that full artifacts are committed
        for _p in (ckpt_pt, ckpt_meta, ckpt_hist):
            if _p is not None and _p.exists():
                try:
                    _p.unlink()
                except OSError:
                    pass

    return {
        "words": words,
        "embeddings": final_embeddings,
        "history": history,
        "shifts": shifts,
        "predictions": predictions,
        "metrics": metrics,
    }


def build_rt13_review_packet(
    snapshots: list[dict],
    dgt_result: dict,
    output_path: str | Path,
    *,
    top_k: int = 100,
) -> pd.DataFrame:
    """Create an RT1.3 partner-review packet for emerging semantics."""

    freq_shift = temporal_shift_terms(snapshots, top_k=top_k, min_count=5)
    dgt_shift = dgt_result["shifts"].head(top_k).copy()
    rows = []
    for rank, row in enumerate(dgt_shift.itertuples(index=False), 1):
        rows.append({"rank": rank, "signal_type": "dgt_semantic_shift", "term": row.word, "transition": row.transition, "score": row.cosine_shift})
    for rank, row in enumerate(freq_shift.head(top_k).itertuples(index=False), 1):
        rows.append({"rank": rank, "signal_type": "frequency_lift", "term": row.word, "transition": row.transition, "score": row.log2_lift})
    packet = pd.DataFrame(rows)
    packet.to_csv(output_path, index=False)
    return packet


def run_rt13_baseline_evaluation(
    snapshots: list[dict],
    dgt_result: dict,
    output_dir: str | Path,
    *,
    max_nodes: int = 350,
    dim: int = 64,
    top_k: int = 50,
    use_cache: bool = True,
) -> dict:
    """Evaluate DGT shifts against RT1.3-style diachronic baselines."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_fingerprint = stable_fingerprint(
        {
            "stage": "run_rt13_baseline_evaluation",
            "version": PIPELINE_CACHE_VERSION,
            "snapshots": snapshot_fingerprint(snapshots),
            "dgt_metrics": dgt_result.get("metrics", {}),
            "dgt_words": dgt_result.get("words", [])[: max_nodes],
            "max_nodes": max_nodes,
            "dim": dim,
            "top_k": top_k,
        }
    )
    manifest_path = output_dir / "rt13_baseline_manifest.json"
    required = [
        output_dir / "svd_procrustes_semantic_shifts.csv",
        output_dir / "frequency_lift_semantic_shifts.csv",
        output_dir / "rt13_baseline_metrics.json",
    ]
    if use_cache and manifest_path.exists() and all(path.exists() for path in required):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("fingerprint") == cache_fingerprint:
                print(f"[cache] loaded RT1.3 quick baselines from {output_dir}")
                return {
                    "metrics": json.loads((output_dir / "rt13_baseline_metrics.json").read_text(encoding="utf-8")),
                    "svd_shifts": pd.read_csv(output_dir / "svd_procrustes_semantic_shifts.csv"),
                    "frequency_shifts": pd.read_csv(output_dir / "frequency_lift_semantic_shifts.csv"),
                }
        except Exception:
            pass
    words = _select_dgt_vocab(snapshots, max_nodes)
    node_to_idx = {w: i for i, w in enumerate(words)}
    svd_embeddings = []

    for snap in tqdm(snapshots, desc="SVD baselines", unit=" spell", leave=False):
        rows, cols, data = [], [], []
        for (src, dst), weight in snap["edge_counts"].items():
            if src in node_to_idx and dst in node_to_idx:
                rows.append(node_to_idx[src])
                cols.append(node_to_idx[dst])
                data.append(float(weight))
        mat = sparse.csr_matrix((data, (rows, cols)), shape=(len(words), len(words)), dtype=np.float64)
        if mat.nnz == 0:
            svd_embeddings.append(np.zeros((len(words), dim), dtype=np.float32))
            continue
        total = mat.sum()
        row_sum = np.asarray(mat.sum(axis=1)).ravel() + 1e-9
        col_sum = np.asarray(mat.sum(axis=0)).ravel() + 1e-9
        coo = mat.tocoo()
        ppmi_values = np.maximum(np.log((coo.data * total) / (row_sum[coo.row] * col_sum[coo.col])), 0.0)
        ppmi = sparse.csr_matrix((ppmi_values, (coo.row, coo.col)), shape=mat.shape)
        n_comp = min(dim, max(2, min(ppmi.shape) - 1))
        emb = TruncatedSVD(n_components=n_comp, random_state=1729).fit_transform(ppmi)
        if emb.shape[1] < dim:
            emb = np.pad(emb, ((0, 0), (0, dim - emb.shape[1])))
        svd_embeddings.append(emb.astype(np.float32))

    aligned = [svd_embeddings[0]]
    for prev, curr in zip(aligned, svd_embeddings[1:]):
        try:
            transform, _ = orthogonal_procrustes(curr, prev)
            aligned.append(curr @ transform)
        except Exception:
            aligned.append(curr)

    svd_rows = []
    for t in range(1, len(aligned)):
        prev, curr = aligned[t - 1], aligned[t]
        cos = np.sum(prev * curr, axis=1) / (np.linalg.norm(prev, axis=1) * np.linalg.norm(curr, axis=1) + 1e-9)
        dist = 1.0 - cos
        for idx in np.argsort(dist)[-top_k:][::-1]:
            svd_rows.append({"transition": f"G{t} to G{t+1}", "word": words[idx], "svd_procrustes_shift": float(dist[idx])})
    svd_shifts = pd.DataFrame(svd_rows)

    freq_shifts = temporal_shift_terms(snapshots, top_k=top_k, min_count=5)
    dgt_shifts = dgt_result["shifts"].head(top_k)
    dgt_terms = set(dgt_shifts["word"]) if not dgt_shifts.empty else set()
    svd_terms = set(svd_shifts.head(top_k)["word"]) if not svd_shifts.empty else set()
    freq_terms = set(freq_shifts.head(top_k)["word"]) if not freq_shifts.empty else set()

    dgt_pred = dgt_result["predictions"]
    naive_mae = None
    if not dgt_pred.empty:
        naive_mae = float(np.mean(np.abs(dgt_pred["heldout_shift"] - dgt_pred["heldout_shift"].mean())))

    metrics = {
        "top_k": top_k,
        "dgt_svd_topk_overlap": len(dgt_terms & svd_terms) / max(len(dgt_terms), 1),
        "dgt_frequency_topk_overlap": len(dgt_terms & freq_terms) / max(len(dgt_terms), 1),
        "dgt_shift_prediction_mae": dgt_result["metrics"].get("shift_prediction_mae"),
        "mean_shift_naive_mae": naive_mae,
    }
    svd_shifts.to_csv(output_dir / "svd_procrustes_semantic_shifts.csv", index=False)
    freq_shifts.to_csv(output_dir / "frequency_lift_semantic_shifts.csv", index=False)
    (output_dir / "rt13_baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _atomic_write_text(
        manifest_path,
        json.dumps(
            {
                "cache_version": PIPELINE_CACHE_VERSION,
                "fingerprint": cache_fingerprint,
                "stage": "run_rt13_baseline_evaluation",
            },
            indent=2,
        ),
    )
    print(f"[cache] saved RT1.3 quick baselines to {output_dir}")
    return {"metrics": metrics, "svd_shifts": svd_shifts, "frequency_shifts": freq_shifts}
