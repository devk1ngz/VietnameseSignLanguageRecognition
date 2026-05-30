import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from .configuration import SLGCNConfig
from transformers import FeatureExtractionMixin, PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput


class SLGCNFeatureExtractor(FeatureExtractionMixin):
    def __init__(self, config: SLGCNConfig = SLGCNConfig(), **kwargs) -> None:
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


def get_adjacency_matrix(num_points: int, labeling_mode: str = "spatial") -> np.ndarray:
    """
    Build a simple adjacency matrix for the skeleton graph.
    Uses a chain-like connectivity for the body + hand landmarks.
    """
    adjacency = np.eye(num_points, dtype=np.float32)
    # Connect consecutive joints as a baseline
    for i in range(num_points - 1):
        adjacency[i, i + 1] = 1
        adjacency[i + 1, i] = 1
    return adjacency


class SpatialGraphConvolution(nn.Module):
    """
    Spatial Graph Convolution layer.
    Performs graph convolution on skeleton joints using the adjacency matrix.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: np.ndarray,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_subsets = 1
        self.register_buffer(
            "adjacency",
            torch.from_numpy(adjacency).float(),
        )
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * self.num_subsets,
            kernel_size=1,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        N, C, T, V = x.size()
        # Graph convolution: aggregate neighbor features
        support = torch.einsum("nctv,vw->nctw", x, self.adjacency)
        out = self.conv(support)
        out = self.bn(out)
        out = self.relu(out)
        return out


class TemporalConvolution(nn.Module):
    """
    Temporal Convolution layer using 1D convolution along the time axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class STGCNBlock(nn.Module):
    """
    Spatial-Temporal Graph Convolution Block.
    Combines a spatial graph convolution with a temporal convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: np.ndarray,
        stride: int = 1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.sgcn = SpatialGraphConvolution(in_channels, out_channels, adjacency)
        self.tgcn = TemporalConvolution(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.sgcn(x)
        out = self.tgcn(out)
        return self.relu(out + res)


class SLGCN(nn.Module):
    """
    Sign Language Graph Convolutional Network.
    Applies spatial-temporal graph convolutions to skeleton sequences
    for sign language recognition.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        num_points: int = 27,
        labeling_mode: str = "spatial",
    ) -> None:
        super().__init__()
        adjacency = get_adjacency_matrix(num_points, labeling_mode)

        self.data_bn = nn.BatchNorm1d(in_channels * num_points)

        self.layers = nn.Sequential(
            STGCNBlock(in_channels, 64, adjacency, residual=False),
            STGCNBlock(64, 64, adjacency),
            STGCNBlock(64, 64, adjacency),
            STGCNBlock(64, 128, adjacency, stride=2),
            STGCNBlock(128, 128, adjacency),
            STGCNBlock(128, 128, adjacency),
            STGCNBlock(128, 256, adjacency, stride=2),
            STGCNBlock(256, 256, adjacency),
            STGCNBlock(256, 256, adjacency),
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V, M) where M is num_people
        N, C, T, V, M = x.size()
        # Merge person dim: (N*M, C, T, V)
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        # Data batch normalization
        x = x.permute(0, 3, 1, 2).contiguous().view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()

        # ST-GCN layers
        x = self.layers(x)

        # Global average pooling
        x = x.mean(dim=-1).mean(dim=-1)  # (N*M, C')

        # Merge back person dim
        x = x.view(N, M, -1).mean(dim=1)  # (N, C')

        return self.fc(x)


class SLGCNForGraphClassification(PreTrainedModel):
    config_class = SLGCNConfig

    def __init__(
        self,
        config: SLGCNConfig = SLGCNConfig(),
        label2id: dict = None,
        id2label: dict = None,
    ) -> None:
        super().__init__(config=config)
        self.label2id = label2id if label2id is not None else config.label2id
        self.id2label = id2label if id2label is not None else config.id2label
        self.num_classes = len(self.label2id)
        self.model = SLGCN(
            num_classes=self.num_classes,
            in_channels=config.in_channels,
            num_points=config.num_points,
            labeling_mode=config.labeling_mode,
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
