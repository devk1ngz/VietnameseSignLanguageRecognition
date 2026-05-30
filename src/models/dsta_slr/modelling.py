import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from .configuration import DSTASLRConfig
from transformers import FeatureExtractionMixin, PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput


class DSTASLRFeatureExtractor(FeatureExtractionMixin):
    def __init__(self, config: DSTASLRConfig = DSTASLRConfig(), **kwargs) -> None:
        super().__init__(**kwargs)
        self.arch = config.arch
        self.num_frames = config.num_frames
        self.num_points = config.num_points
        self.in_channels = config.in_channels
        self.is_vector = config.is_vector
        self.window_size = config.window_size
        self.num_people = config.num_people
        self.bone_stream = config.bone_stream
        self.motion_stream = config.motion_stream


class SpatialAttention(nn.Module):
    """
    Spatial attention module that models relationships between skeleton joints
    at each time step independently.
    """

    def __init__(
        self,
        in_channels: int,
        num_points: int,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (in_channels // num_heads) ** -0.5

        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.proj = nn.Linear(in_channels, in_channels)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, V, C)
        N, T, V, C = x.shape
        h = self.num_heads

        qkv = self.qkv(x).reshape(N, T, V, 3, h, C // h).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (N, h, T, V, d)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (N, h, T, V, V)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(2, 3).reshape(N, T, V, C)
        out = self.proj(out)
        return self.norm(x + out)


class TemporalAttention(nn.Module):
    """
    Temporal attention module that models relationships across time steps
    for each joint independently, using a sliding window.
    """

    def __init__(
        self,
        in_channels: int,
        window_size: int = 120,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.scale = in_channels ** -0.5

        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.proj = nn.Linear(in_channels, in_channels)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, V, C)
        N, T, V, C = x.shape

        # Reshape to process joints independently: (N*V, T, C)
        x_r = x.permute(0, 2, 1, 3).contiguous().view(N * V, T, C)

        qkv = self.qkv(x_r).reshape(N * V, T, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (N*V, T, C)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (N*V, T, T)
        attn = attn.softmax(dim=-1)

        out = (attn @ v)  # (N*V, T, C)
        out = self.proj(out)

        out = out.view(N, V, T, C).permute(0, 2, 1, 3)  # (N, T, V, C)
        return self.norm(x + out)


class DSTABlock(nn.Module):
    """
    Decoupled Spatial-Temporal Attention block.
    Applies spatial attention followed by temporal attention with
    a feed-forward network.
    """

    def __init__(
        self,
        in_channels: int,
        num_points: int,
        s_num_heads: int = 1,
        window_size: int = 120,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spatial_attn = SpatialAttention(in_channels, num_points, s_num_heads)
        self.temporal_attn = TemporalAttention(in_channels, window_size)

        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * 4, in_channels),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_attn(x)
        x = self.temporal_attn(x)
        x = self.norm(x + self.ffn(x))
        return x


class DSTASLR(nn.Module):
    """
    Decoupled Spatial-Temporal Attention for Sign Language Recognition.
    Uses stacked DSTA blocks to process skeleton sequences.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        num_points: int = 27,
        inner_dim: int = 64,
        depth: int = 4,
        s_num_heads: int = 1,
        window_size: int = 120,
        drop_layers: int = 2,
    ) -> None:
        super().__init__()
        # Input embedding
        self.input_proj = nn.Linear(in_channels, inner_dim)
        self.data_bn = nn.BatchNorm1d(in_channels * num_points)

        # DSTA blocks
        self.blocks = nn.ModuleList([
            DSTABlock(
                in_channels=inner_dim,
                num_points=num_points,
                s_num_heads=s_num_heads,
                window_size=window_size,
                dropout=0.1 if i < drop_layers else 0.0,
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(inner_dim)
        self.fc = nn.Linear(inner_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V, M) where M is num_people
        N, C, T, V, M = x.size()
        # Merge person dim: (N*M, C, T, V)
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        # Data batch normalization
        x = x.permute(0, 3, 1, 2).contiguous().view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0, 3, 1, 2).contiguous()
        # x: (N*M, T, V, C)

        # Project to inner dimension
        x = self.input_proj(x)  # (N*M, T, V, inner_dim)

        # Apply DSTA blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Global average pooling over T and V
        x = x.mean(dim=2).mean(dim=1)  # (N*M, inner_dim)

        # Merge back person dim
        x = x.view(N, M, -1).mean(dim=1)  # (N, inner_dim)

        return self.fc(x)


class DSTASLRForGraphClassification(PreTrainedModel):
    config_class = DSTASLRConfig

    def __init__(
        self,
        config: DSTASLRConfig = DSTASLRConfig(),
        label2id: dict = None,
        id2label: dict = None,
    ) -> None:
        super().__init__(config=config)
        self.label2id = label2id if label2id is not None else config.label2id
        self.id2label = id2label if id2label is not None else config.id2label
        self.num_classes = len(self.label2id)
        self.model = DSTASLR(
            num_classes=self.num_classes,
            in_channels=config.in_channels,
            num_points=config.num_points,
            inner_dim=config.inner_dim,
            depth=config.depth,
            s_num_heads=config.s_num_heads,
            window_size=config.window_size,
            drop_layers=config.drop_layers,
        )

        # Load pretrained weights if path exists
        if config.pretrained is not None and Path(config.pretrained).exists():
            state_dict = torch.load(config.pretrained, map_location="cpu")
            for key in list(state_dict.keys()):
                if key.startswith("model."):
                    state_dict[key[6:]] = state_dict.pop(key)
            self.model.load_state_dict(state_dict, strict=False)

        # Freeze layers
        for i, param in enumerate(self.model.parameters()):
            if i >= config.num_frozen_layers:
                break
            param.requires_grad = False

    def forward(
        self,
        poses: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> torch.Tensor:
        logits = self.model(poses)
        if labels is not None:
            labels = labels.to(logits.device, dtype=torch.long)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return ImageClassifierOutput(loss=loss, logits=logits)
        return ImageClassifierOutput(logits=logits)
