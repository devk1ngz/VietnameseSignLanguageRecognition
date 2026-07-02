from transformers import PretrainedConfig


class SLGCNConfig(PretrainedConfig):
    model_type = "sl_gcn"

    def __init__(
        self,
        arch: str = "sl_gcn",
        pretrained: str = None,
        num_frozen_layers: int = 0,
        num_frames: int = 150,
        num_points: int = 27,
        groups: int = 8,
        block_size: int = 41,
        in_channels: int = 3,
        labeling_mode: str = "spatial",
        dropout: float = 0.0,
        is_vector: bool = False,
        bone_stream: bool = False,
        motion_stream: bool = False,
        id2label: dict = None,
        label2id: dict = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.arch = arch
        self.pretrained = pretrained
        self.num_frozen_layers = num_frozen_layers
        self.num_frames = num_frames
        self.num_points = num_points
        self.groups = groups
        self.block_size = block_size
        self.in_channels = in_channels
        self.labeling_mode = labeling_mode
        self.dropout = dropout
        self.is_vector = is_vector
        self.bone_stream = bone_stream
        self.motion_stream = motion_stream
        self.id2label = id2label
        self.label2id = label2id
        self.num_people = 1
        self.window_size = num_frames
