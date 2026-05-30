from transformers import PretrainedConfig


class SPOTERConfig(PretrainedConfig):
    model_type = "spoter"

    def __init__(
        self,
        arch: str = "spoter",
        pretrained: str = None,
        num_frozen_layers: int = 0,
        num_frames: int = 150,
        hidden_dim: int = 108,
        id2label: dict = None,
        label2id: dict = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.arch = arch
        self.pretrained = pretrained
        self.num_frozen_layers = num_frozen_layers
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.id2label = id2label
        self.label2id = label2id
        self.num_points = 54
        self.in_channels = 2
