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


# Anatomical skeleton edges for the 27-joint layout (see SLGCN_JOINTS[27] in
# src/utils/constants.py). Local indices after joint selection:
#   Body (0-6):   0=nose, 1=L-shoulder, 2=R-shoulder, 3=L-elbow, 4=R-elbow,
#                 5=L-wrist, 6=R-wrist
#   Left hand (7-16):  7=wrist, 8=thumb-tip, 9=index-mcp, 10=index-tip,
#                 11=middle-mcp, 12=middle-tip, 13=ring-mcp, 14=ring-tip,
#                 15=pinky-mcp, 16=pinky-tip
#   Right hand (17-26): same order as left, offset by +10
SLGCN_SKELETON_EDGES = {
    27: [
        # body / torso + arms
        (0, 1), (0, 2), (1, 2),
        (1, 3), (3, 5), (2, 4), (4, 6),
        # body wrist -> hand wrist
        (5, 7), (6, 17),
        # left hand: wrist -> finger bases, base -> tip, palm arch
        (7, 8),
        (7, 9), (9, 10),
        (7, 11), (11, 12),
        (7, 13), (13, 14),
        (7, 15), (15, 16),
        (9, 11), (11, 13), (13, 15),
        # right hand (offset +10 from left)
        (17, 18),
        (17, 19), (19, 20),
        (17, 21), (21, 22),
        (17, 23), (23, 24),
        (17, 25), (25, 26),
        (19, 21), (21, 23), (23, 25),
    ],
}


def _build_edges(num_points: int) -> list:
    """Return the skeleton edges for ``num_points`` (anatomical or chain)."""
    edges = SLGCN_SKELETON_EDGES.get(num_points)
    if edges is not None:
        return list(edges)
    # Baseline chain connectivity for unsupported joint counts
    return [(i, i + 1) for i in range(num_points - 1)]


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """Symmetric normalization: D^-1/2 (A + I) D^-1/2."""
    adjacency = adjacency + np.eye(adjacency.shape[0], dtype=adjacency.dtype)
    degree = adjacency.sum(axis=1)
    d_inv_sqrt = np.power(degree, -0.5, where=degree > 0)
    d_inv_sqrt[degree == 0] = 0.0
    d_mat = np.diag(d_inv_sqrt)
    return (d_mat @ adjacency @ d_mat).astype(np.float32)


def get_hop_distance(num_points: int, edges: list, max_hop: int = 1) -> np.ndarray:
    """Pairwise hop (graph) distance between joints, capped at ``max_hop``."""
    adjacency = np.zeros((num_points, num_points))
    for i, j in edges:
        adjacency[i, j] = 1
        adjacency[j, i] = 1
    hop_dis = np.full((num_points, num_points), np.inf)
    transfer = [np.linalg.matrix_power(adjacency, d) for d in range(max_hop + 1)]
    arrive = np.stack(transfer) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive[d]] = d
    return hop_dis


def normalize_digraph(adjacency: np.ndarray) -> np.ndarray:
    """Asymmetric normalization A D^-1 (column-normalized)."""
    degree = adjacency.sum(axis=0)
    num = adjacency.shape[0]
    d_inv = np.zeros((num, num))
    for i in range(num):
        if degree[i] > 0:
            d_inv[i, i] = degree[i] ** -1
    return adjacency @ d_inv


def get_spatial_adjacency(num_points: int, edges: list) -> np.ndarray:
    """
    ST-GCN "spatial" partitioning into 3 subsets per the original paper:
    root (self), centripetal (neighbours closer to the graph centre) and
    centrifugal (neighbours farther from the centre). Returns (3, V, V).

    The graph centre is chosen automatically as the joint that minimises the
    total hop distance to every other joint (a body joint for our layouts).
    """
    max_hop = 1
    hop_dis = get_hop_distance(num_points, edges, max_hop=max_hop)
    finite = np.where(np.isinf(hop_dis), 0.0, hop_dis)
    center = int(np.argmin(finite.sum(axis=1)))

    valid_hop = range(max_hop + 1)
    adjacency = np.zeros((num_points, num_points))
    for hop in valid_hop:
        adjacency[hop_dis == hop] = 1
    norm_adj = normalize_digraph(adjacency)

    subsets = []
    for hop in valid_hop:
        a_root = np.zeros((num_points, num_points))
        a_close = np.zeros((num_points, num_points))
        a_further = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(num_points):
                if hop_dis[j, i] == hop:
                    if hop_dis[j, center] == hop_dis[i, center]:
                        a_root[j, i] = norm_adj[j, i]
                    elif hop_dis[j, center] > hop_dis[i, center]:
                        a_close[j, i] = norm_adj[j, i]
                    else:
                        a_further[j, i] = norm_adj[j, i]
        if hop == 0:
            subsets.append(a_root)
        else:
            subsets.append(a_root + a_close)
            subsets.append(a_further)
    return np.stack(subsets).astype(np.float32)


def get_adjacency_matrix(num_points: int, labeling_mode: str = "spatial") -> np.ndarray:
    """
    Build the skeleton adjacency tensor for the graph convolution.

    Returns a stack of normalized adjacency matrices with shape (K, V, V):
      - "spatial": K=3 ST-GCN partitions (root / centripetal / centrifugal)
      - any other mode: K=1 symmetrically normalized adjacency (D^-1/2 Â D^-1/2)

    For the 27-joint layout an anatomically correct skeleton graph is used
    (body + both hands); other joint counts fall back to chain connectivity.
    """
    edges = _build_edges(num_points)
    if labeling_mode == "spatial":
        return get_spatial_adjacency(num_points, edges)
    adjacency = np.zeros((num_points, num_points), dtype=np.float32)
    for i, j in edges:
        adjacency[i, j] = 1
        adjacency[j, i] = 1
    return normalize_adjacency(adjacency)[None]  # (1, V, V)


class SpatialGraphConvolution(nn.Module):
    """
    Spatial Graph Convolution layer with multi-subset partitioning and a
    learnable edge-importance mask (adaptive graph). The adjacency has shape
    (K, V, V); a separate 1x1 conv branch is learned per subset and the
    partitions are aggregated by the (masked) adjacency.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: np.ndarray,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_subsets = adjacency.shape[0]
        self.register_buffer(
            "adjacency",
            torch.from_numpy(adjacency).float(),
        )
        # Learnable per-edge importance, initialised to 1 (identity behaviour).
        self.edge_importance = nn.Parameter(torch.ones_like(self.adjacency))
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
        adjacency = self.adjacency * self.edge_importance
        out = self.conv(x)  # (N, out*K, T, V)
        out = out.view(N, self.num_subsets, -1, T, V)
        # Aggregate neighbour features per partition, then sum over subsets.
        out = torch.einsum("nkctv,kvw->nctw", out, adjacency)
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
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.sgcn = SpatialGraphConvolution(in_channels, out_channels, adjacency)
        self.tgcn = TemporalConvolution(out_channels, out_channels, stride=stride)
        self.dropout = nn.Dropout(dropout)
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
        out = self.dropout(out)
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
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        adjacency = get_adjacency_matrix(num_points, labeling_mode)

        self.data_bn = nn.BatchNorm1d(in_channels * num_points)

        self.layers = nn.Sequential(
            STGCNBlock(in_channels, 64, adjacency, residual=False, dropout=dropout),
            STGCNBlock(64, 64, adjacency, dropout=dropout),
            STGCNBlock(64, 64, adjacency, dropout=dropout),
            STGCNBlock(64, 128, adjacency, stride=2, dropout=dropout),
            STGCNBlock(128, 128, adjacency, dropout=dropout),
            STGCNBlock(128, 128, adjacency, dropout=dropout),
            STGCNBlock(128, 256, adjacency, stride=2, dropout=dropout),
            STGCNBlock(256, 256, adjacency, dropout=dropout),
            STGCNBlock(256, 256, adjacency, dropout=dropout),
        )

        self.dropout = nn.Dropout(dropout)
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

        x = self.dropout(x)
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
            dropout=config.dropout,
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
