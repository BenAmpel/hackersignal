from etg.rt1.etg_builder import _spell_cooccurrences
from etg.rt1.vocab import build_vocab
from etg.data.schemas import ForumPost
from datetime import datetime, timezone


class _Cfg:
    min_token_len = 2
    vocab_size_cap = 100
    window_size = 2


def _mk(text: str, ts: datetime) -> ForumPost:
    return ForumPost(id="x", text=text, timestamp=ts, forum_id="f", author_hash="a")


def test_etg_persistence(vocab_and_snapshots):
    _, snaps = vocab_and_snapshots
    # Edge persistence: the weighted edge set at t should have >= weights of t-1 on shared edges.
    for t in range(1, len(snaps)):
        prev = dict(
            zip(
                [(int(s), int(d)) for s, d in zip(*snaps[t - 1]["edge_index"].tolist())],
                snaps[t - 1]["edge_weight"].tolist(),
            )
        )
        curr = dict(
            zip(
                [(int(s), int(d)) for s, d in zip(*snaps[t]["edge_index"].tolist())],
                snaps[t]["edge_weight"].tolist(),
            )
        )
        for e, w in prev.items():
            assert curr.get(e, 0.0) >= w - 1e-6


def test_cooccurrence_window():
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    posts = [_mk("alpha beta gamma", ts)]
    vocab = build_vocab(posts, _Cfg())
    out = _spell_cooccurrences(posts, vocab, window=1, min_len=2)
    # window=1: alpha→beta, beta→gamma (directed, j in (i, i+1])
    assert out[(vocab.id("alpha"), vocab.id("beta"))] == 1
    assert out[(vocab.id("beta"), vocab.id("gamma"))] == 1
    # alpha→gamma should NOT appear at window=1
    assert (vocab.id("alpha"), vocab.id("gamma")) not in out
