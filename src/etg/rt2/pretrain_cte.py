"""Self-supervised pretraining of the CTE.

Loss = λ_mlm · L_mlm + λ_cont · L_contrastive

L_mlm          : cross-entropy on masked positions (predicting hash-bucket ids).
L_contrastive  : InfoNCE between two augmented views of the same sample.
                 The two views arrive as different token ids (text-level
                 augmentation) *plus* the encoder's intrinsic dropout provides
                 the SimCSE-style implicit view variance.
"""

from __future__ import annotations

import random

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..config import Config
from ..logging_utils import get_logger
from .cte_model import CTE
from .ev_dataset import (
    EVPretrainDataset,
    collate_contrastive,
    collate_mlm,
)


def _info_nce(z_a: Tensor, z_b: Tensor, temp: float) -> Tensor:
    z_a = torch.nn.functional.normalize(z_a, dim=-1)
    z_b = torch.nn.functional.normalize(z_b, dim=-1)
    sim = z_a @ z_b.T / temp  # [B, B]
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    return torch.nn.functional.cross_entropy(sim, labels)


def pretrain_cte(
    model: CTE,
    pairs,
    config: Config,
    device: torch.device,
) -> dict[str, list[float]]:
    logger = get_logger("etg.pretrain_cte")
    rng = random.Random(config.seed + 101)
    dataset = EVPretrainDataset(pairs, config)
    loader_mlm = DataLoader(
        dataset,
        batch_size=config.cte_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_mlm(b, config, rng),
    )
    loader_cont = DataLoader(
        dataset,
        batch_size=config.cte_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_contrastive(b, config, rng),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.cte_lr)
    model.to(device)
    model.train()
    history = {"loss": [], "mlm": [], "contrastive": []}
    for epoch in range(config.cte_pretrain_epochs):
        ep_loss = 0.0
        ep_mlm = 0.0
        ep_cont = 0.0
        n_batches = 0
        for batch_mlm, batch_cont in zip(loader_mlm, loader_cont):
            input_ids = batch_mlm["input_ids"].to(device)
            labels = batch_mlm["labels"].to(device)
            mlm_mask = batch_mlm["mlm_mask"].to(device)
            logits, _ = model.forward_mlm(input_ids)
            V = logits.shape[-1]
            mlm_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, V),
                labels.view(-1),
                reduction="none",
            )
            mlm_mask_flat = mlm_mask.view(-1)
            if mlm_mask_flat.sum() > 0:
                mlm_loss = (mlm_loss * mlm_mask_flat.float()).sum() / mlm_mask_flat.float().sum()
            else:
                mlm_loss = mlm_loss.mean() * 0.0

            view_a = batch_cont["view_a"].to(device)
            view_b = batch_cont["view_b"].to(device)
            z_a = model.forward_contrastive(view_a)
            z_b = model.forward_contrastive(view_b)
            cont_loss = _info_nce(z_a, z_b, config.cte_contrastive_temp)

            total = config.cte_mlm_weight * mlm_loss + config.cte_contrastive_weight * cont_loss
            optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            ep_loss += float(total.item())
            ep_mlm += float(mlm_loss.item())
            ep_cont += float(cont_loss.item())
            n_batches += 1

        n_batches = max(1, n_batches)
        history["loss"].append(ep_loss / n_batches)
        history["mlm"].append(ep_mlm / n_batches)
        history["contrastive"].append(ep_cont / n_batches)
        logger.info(
            f"CTE pretrain epoch {epoch + 1}/{config.cte_pretrain_epochs} "
            f"loss={ep_loss / n_batches:.4f} "
            f"mlm={ep_mlm / n_batches:.4f} cont={ep_cont / n_batches:.4f}"
        )
    return history
