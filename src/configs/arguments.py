from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
from utils import MODELS, VIDEO_EXTENSIONS
import argparse

class ProcessRecordedVideosArguments():
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            conflict_handler="resolve"
        )
        self.init_general_args()
        self.init_normalize_args()    
        self.init_get_cut_time_args()
        self.init_cut_crop_video_args()
        
    def init_general_args(self) -> None:
        self.parser.add_argument("--log_file", type=str, default=None, help="path to logging file (.txt) which store logged information")
        
        self.parser.add_argument("--input_video", type=str, default=None, help="path to the video need be processed")
        self.parser.add_argument("--output_dir", type=str, default=None, help="path to the directory containing the processed videos, will be in the same folder with the input video and named by input video's name if not specified")
        self.parser.add_argument("--normalized_video", type=str, default=None , help="path to the video after normalizing, will be the input video with '_normalized' if not specified")
        self.parser.add_argument("--cut_time_file", type=str, default=None, help="path to cut time file (.csv) which store logged start and end time of chunks,  will be the input video with '_cut_time' if not specified")
        
        self.parser.add_argument("--normalize_quality", action='store_true', help="whether normalize video quality or not")
        self.parser.add_argument("--get_cut_time", action='store_true', help="whether get cut time file or not")
        self.parser.add_argument("--cut_crop_video", action='store_true', help="whether cut and crop videos or not")
        
        self.parser.add_argument("--visualize", action='store_true', help="whether display video and notations or not")
        self.parser.add_argument("--overwrite", action='store_true', help="whether overwrite output or not")
        
        self.parser.add_argument("--process_all", action='store_true', help="whether process all the data or just part of it (get_cut_time, visualize, cut_crop_video)")
        
        self.parser.add_argument("--from_second", type=int, default=None, help="the second you want to start from, it will be equal 0 if you do not select process all and do not enter this second (get_cut_time, visualize)")
        self.parser.add_argument("--to_second", type=int, default=None, help="the second you want to end with, it will be equal length of data you process if you do not select process all and do not enter this second (get_cut_time, visualize)")
        
    def init_normalize_args(self) -> None:
        self.parser.add_argument("--fps", type=int, default=30, help="fps of the video after processing")
        self.parser.add_argument("--resolution", type=str, default='1920:1080', help="width:height of the video after processing")
        
    def init_get_cut_time_args(self) -> None:
        self.parser.add_argument("--threshold", type=int, default=160, help="angle threshold (degrees) to determine whether the arm is up or down")
        self.parser.add_argument("--min_up_frame", type=int, default=20, help="minimum number of frames to determine hand up")
        self.parser.add_argument("--min_down_frame", type=int, default=20, help="minimum number of frames to determine hand down")
        self.parser.add_argument("--delay", type=int, default=400, help="the number of miliseconds that the video border adds")
        
    def init_cut_crop_video_args(self) -> None:
        self.parser.add_argument("--from_index", type=int, default=None, help="the index you want to start from, it will be equal 0 if you do not select process all and do not enter this index (from cut time file)")
        self.parser.add_argument("--to_index", type=int, default=None, help="the index you want to end with, it will be equal length of data you process if you do not select process all and do not enter this index (from cut time file)")
        self.parser.add_argument("--crop_dimensions", type=str, default='1080:1080:420:0', help="'width:height:x:y' with width and height of the cropped video, x,y-coordinate of the top-left corner of the cropped video")   

    def parse(self) -> object:
        args = self.parser.parse_args()
        return args

@dataclass
class TransformConfig:
    # RGB specific
    horizontal_flip_prob: float = 0.5
    aug_type: str = "augmix"
    aug_paras: dict = field(
        default_factory=lambda: {
            "magnitude": 3,
            "alpha": 1.0,
            "width": 5,
            "depth": -1,
        }
    )
    sample_rate: int = 4

    # Pose specific
    aug_prob: float = 0.5

    # SL-GCN, DSTA-SLR specific
    rotation_std: float = 0.2
    shear_std: float = 0.2
    scale_std: float = 0.2

    # SPOTER specific
    add_gaussian_noise: bool = False
    gaussian_noise_mean: float = 0.0
    gaussian_noise_std: float = 0.001

    def __post_init__(self):
        assert self.aug_type in ["augmix", "mixup"], \
            "Only AugMix and MixUp are supported for now"


@dataclass
class DataConfig:
    dataset: str = "vsl"
    modality: str = "rgb"
    subset: str = None
    data_dir: str = "data/processed/vsl"
    transform: Any = None
    fps: int = 30
    debug: bool = False
    transform: TransformConfig = TransformConfig()

    def __post_init__(self):
        assert self.dataset in ["visl_98", "visl_400"], \
            "Only VSL dataset is supported for now"
        assert self.modality in ["rgb", "pose"], \
            "Only RGB and Pose modalities are supported for now"


@dataclass
class ModelConfig:
    arch: str = "swin3d_t"
    pretrained: str = "DEFAULT"
    num_frozen_layers: int = 0
    ignored_weights: list = field(default_factory=lambda: [])
    num_frames: int = 16

    # SL-GCN specific
    num_points: int = 27
    groups: int = 8
    block_size: int = 41
    in_channels: int = 3
    labeling_mode: str = "spatial"
    is_vector: bool = False
    bone_stream: bool = False
    motion_stream: bool = False

    # DSTA-SLR specific
    graph: str = "wlasl"
    inner_dim: int = 64
    drop_layers: int = 2
    depth: int = 4
    s_num_heads: int = 1
    window_size: int = 120

    # SPOTER specific
    hidden_dim: int = 108

    def __post_init__(self):
        assert self.arch in MODELS, f"Model {self.arch} is not supported"


@dataclass
class TrainingConfig:
    output_dir: str = "experiments"
    remove_unused_columns: bool = False
    do_train: bool = True
    use_cpu: bool = False

    eval_strategy: str = "epoch"
    logging_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 1
    save_steps: int = 1
    eval_steps: int = 1
    save_total_limit: int = 10

    learning_rate: float = 5e-5
    weight_decay: float = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    warmup_ratio: float = 0.1

    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    dataloader_num_workers: int = 0

    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    resume_from_checkpoint: str = None

    run_name: str = "swin3d"
    report_to: str = None
    push_to_hub: bool = False
    hub_model_id: str = None
    hub_strategy: str = "checkpoint"
    hub_private_repo: bool = True

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if str(self.output_dir) == "experiments":
            self.output_dir = self.output_dir / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.hub_model_id is not None:
            self.push_to_hub = True
            if len(self.hub_model_id.split("/")) == 1:
                self.hub_model_id = f"{self.hub_model_id}/{self.run_name}"


@dataclass
class EvaluationConfig:
    arch: str = None
    pretrained: str = None
    output_dir: str = "experiments"
    eval_set: str = "test"
    push_to_hub: bool = False
    batch_size: int = 8

    def __post_init__(self):
        assert self.arch is not None, \
            "Model architecture is required for evaluation"
        assert self.pretrained is not None, \
            "Pretrained model path is required for evaluation"
        assert self.eval_set in ["train", "validation", "test"], \
            "Evaluation set must be either 'train', 'validation', or 'test'"
        self.output_dir = Path(self.output_dir)
        if str(self.output_dir) == "experiments":
            self.output_dir = self.output_dir / self.pretrained.split("/")[-1]
        self.output_dir = self.output_dir / self.eval_set
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class InferenceConfig:
    source: str = "webcam"
    output_dir: str = "demo"
    use_onnx: bool = False
    device: str = "cpu"
    cache_dir: str = "models/huggingface"

    visualize: bool = False
    show_skeleton: bool = False

    visibility: float = 0.5
    angle_threshold: int = 140
    min_num_up_frames: int = 10
    min_num_down_frames: int = 10
    delay: int = 400

    top_k: int = 3
    # SL-GCN, DSTA-SLR specific
    bone_stream: bool = False
    motion_stream: bool = False

    def __post_init__(self):
        self.source = Path(self.source)
        assert any((
            str(self.source) == "webcam",
            (self.source.exists() and str(self.source).endswith(VIDEO_EXTENSIONS))
        )), \
            f"Only Webcam and Video sources are supported for now (got {self.source})"
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
