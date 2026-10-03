from __future__ import annotations

import random
import os

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GatedFusionSelector(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_clusters: int,
        dropout: float = 0.1,
        gate_hidden_dim: int = 128,
        fusion_mode: str = "full",
    ) -> None:
        super().__init__()
        if fusion_mode not in {"full", "static_gate", "no_interaction"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.image_branch = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.text_branch = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.interaction_branch = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 3, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.interaction_scale = nn.Parameter(torch.tensor(0.1))
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_clusters),
        )

    def fused_features(
        self, image: torch.Tensor, text: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        interaction = image * text
        image_hidden = self.image_branch(image)
        text_hidden = self.text_branch(text)
        if self.fusion_mode == "static_gate":
            gate = torch.full(
                (image.shape[0], 1), 0.5, dtype=image.dtype, device=image.device
            )
        else:
            gate = self.gate(torch.cat((image, text, interaction), dim=-1))
        fused = gate * image_hidden + (1.0 - gate) * text_hidden
        if self.fusion_mode != "no_interaction":
            fused = fused + self.interaction_scale * self.interaction_branch(interaction)
        return fused, gate.squeeze(-1)

    def forward_with_gate(
        self, image: torch.Tensor, text: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused, gate = self.fused_features(image, text)
        return self.output(fused), gate

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return self.forward_with_gate(image, text)[0]
