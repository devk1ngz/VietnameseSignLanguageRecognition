import copy
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
from transformers import PreTrainedModel, FeatureExtractionMixin
from transformers.modeling_outputs import ImageClassifierOutput
from .configuration import SPOTERConfig


class SPOTERFeatureExtractor(FeatureExtractionMixin):
    def __init__(self, config: SPOTERConfig = SPOTERConfig(), **kwargs) -> None:
        super().__init__(**kwargs)
        self.arch = config.arch
        self.mean = 0.0
        self.std = 0.001
        self.num_frames = config.num_frames
        self.num_points = config.num_points
        self.in_channels = config.in_channels


class SPOTERTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    Edited TransformerDecoderLayer implementation omitting the redundant self-attention operation as opposed to the
    standard implementation.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super(SPOTERTransformerDecoderLayer, self).__init__(
            d_model, nhead, dim_feedforward, dropout, activation
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class SPOTER(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int) -> None:
        super().__init__()
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim))
        self.pos = nn.Parameter(
            (
                torch
                .cat([self.row_embed[0].unsqueeze(0).repeat(1, 1, 1)], dim=-1)
                .flatten(0, 1)
                .unsqueeze(0)
            )
        )
        self.class_query = nn.Parameter(torch.rand(1, hidden_dim))
        self.transformer = nn.Transformer(hidden_dim, 9, 6, 6)
        self.linear_class = nn.Linear(hidden_dim, num_classes)

        # Deactivate the initial attention decoder mechanism
        custom_decoder_layer = SPOTERTransformerDecoderLayer(
            d_model=self.transformer.d_model,
            nhead=self.transformer.nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
        )
        self.transformer.decoder.layers = nn.ModuleList([
            copy.deepcopy(custom_decoder_layer)
            for _ in range(self.transformer.decoder.num_layers)
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = torch.unsqueeze(inputs.squeeze(0).flatten(start_dim=1), 1).float()
        h = self.transformer(self.pos + h, self.class_query.unsqueeze(0)).transpose(0, 1)
        res = self.linear_class(h)
        return res


class SPOTERForGraphClassification(PreTrainedModel):
    config_class = SPOTERConfig

    def __init__(
        self,
        config: SPOTERConfig = SPOTERConfig(),
        label2id: dict = None,
        id2label: dict = None,
    ) -> None:
        super().__init__(config=config)
        self.label2id = label2id if label2id is not None else config.label2id
        self.id2label = id2label if id2label is not None else config.id2label
        self.num_classes = len(self.label2id)
        self.model = SPOTER(
            num_classes=self.num_classes,
            hidden_dim=self.config.hidden_dim,
        )

        if Path(self.config.pretrained).exists():
            state_dict = torch.load(self.config.pretrained)
            for key in list(state_dict.keys()):
                if key.startswith("model."):
                    state_dict[key[6:]] = state_dict.pop(key)
            self.model.load_state_dict(state_dict)

    def forward(
        self,
        poses: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> torch.Tensor:
        logits = self.model(poses).squeeze(0)
        if labels is not None:
            labels = labels.to(logits.device, dtype=torch.long)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return ImageClassifierOutput(loss=loss, logits=logits)
        return ImageClassifierOutput(logits=logits)
