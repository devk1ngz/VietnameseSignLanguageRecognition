from transformers import PretrainedConfig
from transformers import VideoMAEConfig as HFVideoMAEConfig


class VideoMAEConfig(PretrainedConfig):
    model_type = "videomae"

    def __init__(
        self,
        arch: str = "videomae",
        pretrained: str = "MCG-NJU/videomae-small-finetuned-kinetics",
        num_frozen_layers: int = 0,
        id2label: dict = None,
        label2id: dict = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.arch = arch
        self.pretrained = pretrained
        self.num_frozen_layers = num_frozen_layers

        config = HFVideoMAEConfig.from_pretrained(pretrained)
        self.num_frames = config.num_frames

        self.id2label = id2label
        self.label2id = label2id
