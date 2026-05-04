"""Shared test fixtures — tiny config so the whole suite runs in ~1 minute."""

from __future__ import annotations

import pytest

from etg.config import Config
from etg.data.synthetic_ev import SyntheticEV
from etg.data.synthetic_forum import SyntheticForum
from etg.data.time_spells import TimeSpellIndex
from etg.rt1.etg_builder import build_etg_sequence, group_posts_by_spell
from etg.rt1.laplacian_pe import attach_lap_pe
from etg.rt1.vocab import build_vocab
from etg.seeding import set_global_seed


@pytest.fixture(scope="session")
def tiny_config() -> Config:
    return Config(
        seed=13,
        n_posts=200,
        n_spells=3,
        n_ev_pairs=50,
        n_ev_negatives_per_positive=2,
        vocab_size_cap=200,
        trigram_buckets=2**12,
        dgt_hidden_dim=32,
        dgt_num_heads=2,
        dgt_num_layers=1,
        dgt_epochs=1,
        dgt_batch_size_edges=64,
        lap_pe_k=4,
        cte_emb_dim=32,
        cte_lstm_hidden=32,
        cte_cnn_channels=32,
        cte_attn_dim=32,
        cte_max_len=16,
        cte_pretrain_epochs=1,
        cte_finetune_epochs=1,
        cte_batch_size=16,
        inject_n_shift_words=4,
    )


@pytest.fixture(scope="session")
def synthetic_forum_posts(tiny_config):
    set_global_seed(tiny_config.seed)
    return list(SyntheticForum(tiny_config))


@pytest.fixture(scope="session")
def synthetic_ev_pairs(tiny_config):
    set_global_seed(tiny_config.seed)
    return list(SyntheticEV(tiny_config))


@pytest.fixture(scope="session")
def vocab_and_snapshots(tiny_config, synthetic_forum_posts):
    vocab = build_vocab(synthetic_forum_posts, tiny_config)
    ts = TimeSpellIndex.from_posts(synthetic_forum_posts, tiny_config.n_spells)
    by_spell = group_posts_by_spell(synthetic_forum_posts, ts)
    snapshots = build_etg_sequence(
        synthetic_forum_posts, by_spell, vocab, tiny_config
    )
    attach_lap_pe(snapshots, tiny_config)
    return vocab, snapshots
