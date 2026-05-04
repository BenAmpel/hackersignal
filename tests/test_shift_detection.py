import numpy as np
import torch

from etg.rt1.shift_detection import detect_shifts_pairwise


def test_shift_detection_finds_outlier():
    V = 50
    d = 16
    torch.manual_seed(0)
    base = torch.randn(V, d)
    embeddings = [base, base.clone(), base.clone()]
    # Inject a big drift on word 7 between spell 1 and 2
    embeddings[2][7] += 5.0 * torch.randn(d)
    masks = [torch.ones(V, dtype=torch.bool) for _ in range(3)]
    results = detect_shifts_pairwise(embeddings, masks, top_quantile=0.05)
    assert len(results) == 2
    # The second pair should pick up word 7.
    assert 7 in set(results[1].shifted_ids.tolist())


def test_shift_detection_handles_empty():
    V = 10
    emb = [torch.zeros(V, 4) for _ in range(2)]
    masks = [torch.zeros(V, dtype=torch.bool) for _ in range(2)]
    out = detect_shifts_pairwise(emb, masks, top_quantile=0.05)
    assert len(out) == 1
    assert out[0].shifted_ids.size == 0
