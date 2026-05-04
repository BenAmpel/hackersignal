import torch

from etg.device import auto_device
from etg.rt2.cte_model import CTE
from etg.rt2.trigram_hasher import tokenize_cte


def test_cte_mlm_forward(tiny_config):
    model = CTE(tiny_config).to(auto_device())
    texts = ["sqli exploit apache drop shell", "rce on nginx via base64 payload"]
    ids = [
        tokenize_cte(t, tiny_config.trigram_buckets, tiny_config.cte_max_len)
        for t in texts
    ]
    ids_t = torch.tensor(ids, dtype=torch.long, device=auto_device())
    logits, mask = model.forward_mlm(ids_t)
    assert logits.shape == (2, tiny_config.cte_max_len, tiny_config.trigram_buckets)
    assert mask.shape == (2, tiny_config.cte_max_len)
    assert torch.isfinite(logits).all()


def test_cte_sentence_shape(tiny_config):
    model = CTE(tiny_config).to(auto_device())
    ids = [[2] + [5] * (tiny_config.cte_max_len - 1), [2] + [7] * (tiny_config.cte_max_len - 1)]
    ids_t = torch.tensor(ids, dtype=torch.long, device=auto_device())
    z = model.encode_sentence(ids_t)
    assert z.shape == (2, tiny_config.cte_attn_dim)
    assert torch.isfinite(z).all()
