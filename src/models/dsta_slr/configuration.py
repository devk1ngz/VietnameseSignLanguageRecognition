from transformers import PretrainedConfig


class DSTASLRConfig(PretrainedConfig):
    model_type = "dsta_slr"

    def __init__(
        self,
        arch: str = "dsta_slr",
        pretrained: str = None,
        num_frozen_layers: int = 0,
        num_frames: int = 150,
        num_points: int = 27,
        in_channels: int = 3,
        is_vector: bool = False,
        bone_stream: bool = False,
        motion_stream: bool = False,
        graph: str = "wlasl",
        inner_dim: int = 64,
        drop_layers: int = 2,
        depth: int = 4,
        s_num_heads: int = 1,
        window_size: int = 120,
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
        self.in_channels = in_channels
        self.is_vector = is_vector
        self.bone_stream = bone_stream
        self.motion_stream = motion_stream
        self.graph = graph
        self.inner_dim = inner_dim
        self.drop_layers = drop_layers
        self.depth = depth
        self.s_num_heads = s_num_heads
        self.window_size = window_size
        self.id2label = id2label
        self.label2id = label2id
        self.num_people = 1
