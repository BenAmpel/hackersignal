import torch

from etg.rt1.laplacian_pe import attach_lap_pe


def test_lap_pe_shape(tiny_config, vocab_and_snapshots):
    _, snaps = vocab_and_snapshots
    # Already attached in fixture; verify shape
    for snap in snaps:
        assert snap["lap_pe"].shape == (snap["x"].shape[0], tiny_config.lap_pe_k)
        assert torch.isfinite(snap["lap_pe"]).all()


def test_lap_pe_is_deterministic_under_canonical_sign(tiny_config, vocab_and_snapshots):
    _, snaps = vocab_and_snapshots
    first = [s["lap_pe"].clone() for s in snaps]
    # Recompute: should be identical (canonical signs at inference).
    attach_lap_pe(snaps, tiny_config)
    for a, s in zip(first, snaps):
        assert torch.allclose(a, s["lap_pe"], atol=1e-4)
