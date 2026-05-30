from transformers import PretrainedConfig


class Swin3DConfig(PretrainedConfig):
    model_type = "swin3d"

    def __init__(
        self,
        arch: str = "swin3d_t",
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
