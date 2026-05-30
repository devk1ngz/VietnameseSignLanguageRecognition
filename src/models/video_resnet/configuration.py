from transformers import PretrainedConfig


class VideoResNetConfig(PretrainedConfig):
    model_type = "video_resnet"

    def __init__(
        self,
        arch: str = "r3d_18",
        pretrained: str = "DEFAULT",
        num_frozen_layers: int = 0,
        num_frames: int = 16,
        id2label: dict = None,
        label2id: dict = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.arch = arch
        self.pretrained = pretrained
        self.num_frozen_layers = num_frozen_layers
        self.num_frames = num_frames
        self.id2label = id2label
        self.label2id = label2id
