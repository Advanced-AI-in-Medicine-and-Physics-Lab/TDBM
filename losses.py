"""Vessel-aware contrastive learning (VCL) loss.

The physics-based data engine can render two XCA frames that share an identical
background but contain different vascular trees.  Passing both through the
denoising U-Net gives two feature maps whose only systematic difference is the
vasculature, which turns them into a clean supervision signal:

* **anchors**   - features of image 1 inside the vessel region ``Omega_v``
  (the union of the two vessel masks);
* **positives** - the spatially corresponding features of image 2;
* **negatives** - features of image 2 sampled from the background ``Omega_b``.

Minimising the resulting InfoNCE term amplifies the network's response to vessel
patterns, so that residual vessel signal is suppressed more completely when the
bridge maps back to the contrast-free domain.
"""

from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VesselAwareContrastiveLoss(nn.Module):
    """InfoNCE over vessel/background feature pairs.

    Args:
        num_anchors: Number of anchor locations sampled per image.
        patch_size: One or more average-pooling window sizes.  ``[1]`` (the paper
            setting) contrasts individual feature vectors; larger values contrast
            pooled patches, and the losses over all sizes are summed.
        num_negatives: Background features drawn per anchor.
        temperature: InfoNCE temperature ``tau``.
    """

    def __init__(
        self,
        num_anchors: int = 256,
        patch_size: Union[int, Sequence[int]] = (1,),
        num_negatives: int = 100,
        temperature: float = 0.05,
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.patch_size = list(patch_size) if isinstance(patch_size, (list, tuple)) else [patch_size]
        self.num_negatives = num_negatives
        self.temperature = temperature

    def forward(
        self,
        feat_view1: Union[torch.Tensor, Iterable[torch.Tensor]],
        feat_view2: Union[torch.Tensor, Iterable[torch.Tensor]],
        vessel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the VCL loss.

        Args:
            feat_view1: ``[B, C, H, W]`` features of image 1, or an iterable of
                such tensors (one per U-Net level).
            feat_view2: Matching features of image 2.
            vessel_mask: ``[B, 1, H_m, W_m]`` union of both vessel masks, resized
                internally to the feature resolution.  If ``None``, the loss falls
                back to plain patch-NCE with in-batch negatives.

        Returns:
            Scalar loss averaged over levels and batch elements.
        """
        if isinstance(feat_view1, torch.Tensor):
            return self._loss_single_level(feat_view1, feat_view2, vessel_mask)
        return self._loss_multi_level(list(feat_view1), list(feat_view2), vessel_mask)

    # --------------------------------------------------------------- helpers

    def _loss_multi_level(
        self,
        feats1: List[torch.Tensor],
        feats2: List[torch.Tensor],
        vessel_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        total, count = 0.0, 0
        for feat1, feat2 in zip(feats1, feats2):
            if feat1 is None or feat2 is None:
                continue
            total = total + self._loss_single_level(feat1, feat2, vessel_mask)
            count += 1
        return total / max(count, 1)

    def _loss_single_level(
        self,
        feat_view1: torch.Tensor,
        feat_view2: torch.Tensor,
        vessel_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if feat_view1.shape != feat_view2.shape:
            raise ValueError(f"Feature shapes do not match: {feat_view1.shape} vs {feat_view2.shape}")

        _, _, height, width = feat_view1.shape
        if vessel_mask is not None and vessel_mask.shape[-2:] != (height, width):
            vessel_mask = F.interpolate(vessel_mask.float(), size=(height, width), mode="nearest")

        total = 0.0
        for size in self.patch_size:
            total = total + self._loss_for_patch_size(feat_view1, feat_view2, vessel_mask, size)
        return total

    def _loss_for_patch_size(
        self,
        feat_view1: torch.Tensor,
        feat_view2: torch.Tensor,
        vessel_mask: Optional[torch.Tensor],
        patch_size: int,
    ) -> torch.Tensor:
        if patch_size > 1:
            pooling = nn.AvgPool2d(kernel_size=(patch_size, patch_size))
            feat_view1 = pooling(feat_view1)
            feat_view2 = pooling(feat_view2)
            if vessel_mask is not None:
                vessel_mask = pooling(vessel_mask.float())

        feat_view1 = F.normalize(feat_view1, dim=1)
        feat_view2 = F.normalize(feat_view2, dim=1)
        return self._batched_infonce(feat_view1, feat_view2, vessel_mask)

    def _batched_infonce(
        self,
        feat_view1: torch.Tensor,
        feat_view2: torch.Tensor,
        vessel_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, channels, height, width = feat_view1.shape
        device = feat_view1.device

        feat1_flat = feat_view1.permute(0, 2, 3, 1).reshape(batch, -1, channels)  # [B, N, C]
        feat2_flat = feat_view2.permute(0, 2, 3, 1).reshape(batch, -1, channels)
        mask_flat = vessel_mask.reshape(batch, -1) if vessel_mask is not None else None

        losses = []
        for b in range(batch):
            anchor_ids = self._sample_anchor_ids(mask_flat[b] if mask_flat is not None else None,
                                                 height * width, device)
            anchor = feat1_flat[b, anchor_ids, :]
            positive = feat2_flat[b, anchor_ids, :]

            if mask_flat is not None:
                loss = self._vessel_aware_nce(anchor, positive, feat2_flat[b], mask_flat[b])
            else:
                loss = self._patch_nce(anchor, positive)
            losses.append(loss.mean())

        return torch.stack(losses).mean()

    def _sample_anchor_ids(
        self,
        mask: Optional[torch.Tensor],
        num_locations: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Pick anchor locations, preferring the vessel region when a mask is given."""
        if mask is None:
            num = min(self.num_anchors, num_locations)
            return torch.from_numpy(np.random.permutation(num_locations)[:num]).to(device).long()

        vessel_ids = torch.where(mask > 0)[0]
        if len(vessel_ids) == 0:
            # Frame without any vessel at this resolution: fall back to random anchors.
            return torch.randperm(num_locations, device=device)[: self.num_anchors]
        if len(vessel_ids) > self.num_anchors:
            keep = torch.randperm(len(vessel_ids), device=device)[: self.num_anchors]
            return vessel_ids[keep]
        return vessel_ids

    def _vessel_aware_nce(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        all_feat_view2: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """InfoNCE with negatives drawn from the background region of image 2."""
        num_anchors = anchor.shape[0]
        device = anchor.device

        pos_logits = torch.einsum("nc,nc->n", anchor, positive).unsqueeze(1)  # [A, 1]

        background_ids = torch.where(mask <= 0)[0]
        if len(background_ids) == 0:
            # Degenerate case (mask covers everything): use all locations as candidates.
            background_ids = torch.arange(all_feat_view2.shape[0], device=device)

        # Sampled with replacement, which also covers the case of very small
        # background regions.
        rand_ids = torch.randint(0, len(background_ids), (num_anchors, self.num_negatives), device=device)
        neg_feats = all_feat_view2[background_ids[rand_ids]]  # [A, K, C]
        neg_logits = torch.bmm(neg_feats, anchor.unsqueeze(2)).squeeze(2)  # [A, K]

        logits = torch.cat([pos_logits, neg_logits], dim=1) / self.temperature
        labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        return F.cross_entropy(logits, labels, reduction="none")

    def _patch_nce(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        """Plain patch-NCE with in-batch negatives (used when no mask is available).

        Follows the PatchNCE formulation of CUT (Park et al., ECCV 2020).
        """
        num_patches, dim = anchor.shape
        device = anchor.device

        l_pos = torch.bmm(anchor.view(num_patches, 1, -1), positive.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        anchor_b = anchor.view(1, -1, dim)
        positive_b = positive.view(1, -1, dim)
        l_neg = torch.bmm(anchor_b, positive_b.transpose(2, 1))
        diagonal = torch.eye(num_patches, device=device, dtype=torch.bool)[None, :, :]
        l_neg.masked_fill_(diagonal, -10.0)  # exclude the positive from the negatives
        l_neg = l_neg.view(-1, num_patches)

        logits = torch.cat((l_pos, l_neg), dim=1) / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
        return F.cross_entropy(logits, labels, reduction="none")
