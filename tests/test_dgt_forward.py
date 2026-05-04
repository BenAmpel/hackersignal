import torch

from etg.device import auto_device
from etg.rt1.dgt_model import DGT


def test_dgt_forward(tiny_config, vocab_and_snapshots):
    _, snaps = vocab_and_snapshots
    feature_dim = snaps[0]["x"].shape[1]
    device = auto_device()
    model = DGT(tiny_config, feature_dim).to(device)
    outs = model(snaps, device=device, train=True)
    assert len(outs) == len(snaps)
    for h in outs:
        assert h.shape == (snaps[0]["x"].shape[0], tiny_config.dgt_hidden_dim)
        assert torch.isfinite(h).all()
