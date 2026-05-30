import torch
import logging
import onnxruntime as ort
from time import time
from typing import Union
from configs import ModelConfig, InferenceConfig, EvaluationConfig
from utils import (
    POSE_BASED_MODELS,
    RGB_BASED_MODELS,
    HUGGINGFACE_RGB_BASED_MODELS,
    TORCHHUB_RGB_BASED_MODELS,
)
from transformers import (
    ImageProcessingMixin,
    FeatureExtractionMixin,
    AutoModelForVideoClassification,
    AutoModel,
    Pipeline,
    pipeline,
)
from transformers.pipelines import PIPELINE_REGISTRY
from visualization import draw_text_on_image
from utils import exists_on_hf
from models import (
    Swin3DConfig, Swin3DImageProcessor, Swin3DForVideoClassification,
    S3DConfig, S3DImageProcessor, S3DForVideoClassification,
    VideoResNetConfig, VideoResNetImageProcessor, VideoResNetForVideoClassification,
    MViTConfig, MViTImageProcessor, MViTForVideoClassification,
    SLGCNConfig, SLGCNFeatureExtractor, SLGCNForGraphClassification,
    SPOTERConfig, SPOTERFeatureExtractor, SPOTERForGraphClassification,
    DSTASLRConfig, DSTASLRFeatureExtractor, DSTASLRForGraphClassification,
    VideoMAEConfig, VideoMAEImageProcessor, VideoMAEForVideoClassification
)
from pipelines import (
    VideoClassificationPipeline,
    SLGCNGraphClassificationPipeline,
    SPOTERGraphClassificationPipeline,
)


def load_model(
    model_config: Union[ModelConfig, EvaluationConfig],
    label2id: dict = None,
    id2label: dict = None,
    do_train: bool = False,
) -> tuple:
    '''
    '''
    if isinstance(model_config, EvaluationConfig):
        do_train = False

    if do_train:
        if model_config.arch in POSE_BASED_MODELS:
            return load_pose_model_for_training(model_config, label2id, id2label)
        return load_rgb_model_for_training(model_config, label2id, id2label)

    if model_config.arch in POSE_BASED_MODELS:
        processor = FeatureExtractionMixin.from_pretrained(
            model_config.pretrained,
            trust_remote_code=True,
            cache_dir="models/huggingface",
        )
        model = AutoModel.from_pretrained(
            model_config.pretrained,
            trust_remote_code=True,
            cache_dir="models/huggingface",
        )
    else:
        processor = ImageProcessingMixin.from_pretrained(
            model_config.pretrained,
            trust_remote_code=True,
            cache_dir="models/huggingface",
        )
        model = AutoModelForVideoClassification.from_pretrained(
            model_config.pretrained,
            trust_remote_code=True,
            cache_dir="models/huggingface",
        )
    model.eval()
    return model.config, processor, model


def load_rgb_model_for_training(
    model_config: ModelConfig,
    label2id: dict = None,
    id2label: dict = None,
) -> tuple:
    '''
    '''
    if model_config.arch in HUGGINGFACE_RGB_BASED_MODELS:
        if model_config.arch == "videomae":
            config_class = VideoMAEConfig
            processor_class = VideoMAEImageProcessor
            model_class = VideoMAEForVideoClassification
    elif exists_on_hf(model_config.pretrained):
        processor = ImageProcessingMixin.from_pretrained(
            model_config.pretrained,
            trust_remote_code=True,
            cache_dir="models/huggingface",
        )
        model = AutoModelForVideoClassification.from_pretrained(
            model_config.pretrained,
            label2id,
            id2label,
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
            cache_dir="models/huggingface",
        )
        return model.config, processor, model
    elif model_config.arch in TORCHHUB_RGB_BASED_MODELS:
        if model_config.arch in ['swin3d_t', 'swin3d_s', 'swin3d_b']:
            config_class = Swin3DConfig
            processor_class = Swin3DImageProcessor
            model_class = Swin3DForVideoClassification
        elif model_config.arch in ['r3d_18', 'mc3_18', 'r2plus1d_18']:
            config_class = VideoResNetConfig
            processor_class = VideoResNetImageProcessor
            model_class = VideoResNetForVideoClassification
        elif model_config.arch in ['s3d']:
            config_class = S3DConfig
            processor_class = S3DImageProcessor
            model_class = S3DForVideoClassification
        elif model_config.arch in ['mvit_v1_b', 'mvit_v2_s']:
            config_class = MViTConfig
            processor_class = MViTImageProcessor
            model_class = MViTForVideoClassification
    else:
        logging.error(f"Model {model_config.arch} is not supported")
        exit(1)

    config_class.register_for_auto_class()
    processor_class.register_for_auto_class("AutoImageProcessor")
    model_class.register_for_auto_class("AutoModel")
    model_class.register_for_auto_class("AutoModelForVideoClassification")
    logging.info(f"{model_config.arch} classes registered")

    config = config_class(**vars(model_config))
    processor = processor_class(config=config)
    model = model_class(config=config, label2id=label2id, id2label=id2label)

    return config, processor, model


def load_pose_model_for_training(
    model_config: ModelConfig,
    label2id: dict = None,
    id2label: dict = None,
) -> tuple:
    '''
    '''
    if exists_on_hf(model_config.pretrained):
        processor = FeatureExtractionMixin.from_pretrained(
            model_config.pretrained,
            trust_remote_code=True,
            cache_dir="models/huggingface",
        )
        model = AutoModel.from_pretrained(
            model_config.pretrained,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
            cache_dir="models/huggingface",
        )
        return model.config, processor, model
    elif model_config.arch in POSE_BASED_MODELS:
        if model_config.arch == "spoter":
            config_class = SPOTERConfig
            processor_class = SPOTERFeatureExtractor
            model_class = SPOTERForGraphClassification
        elif model_config.arch == "sl_gcn":
            config_class = SLGCNConfig
            processor_class = SLGCNFeatureExtractor
            model_class = SLGCNForGraphClassification
        elif model_config.arch == "dsta_slr":
            config_class = DSTASLRConfig
            processor_class = DSTASLRFeatureExtractor
            model_class = DSTASLRForGraphClassification
    else:
        logging.error(f"Model {model_config.arch} is not supported")
        exit(1)

    config_class.register_for_auto_class()
    processor_class.register_for_auto_class("AutoFeatureExtractor")
    model_class.register_for_auto_class("AutoModel")
    logging.info(F"Registering {model_config.arch} classes")

    config = config_class(**vars(model_config))
    processor = processor_class(config=config)
    model = model_class(config=config, label2id=label2id, id2label=id2label)

    return config, processor, model


class Predictions:
    def __init__(
        self,
        predictions: list[dict] = None,
        inference_time: float = 0,
        start_time: float = 0,
        end_time: float = 0,
    ) -> None:
        self.predictions = predictions
        self.inference_time = inference_time
        self.start_time = start_time
        self.end_time = end_time

    def visualize(
        self,
        frame: torch.Tensor,
        position: tuple = (20, 100),
        prefix: str = "Predictions",
        color: tuple = (0, 0, 255),
    ) -> None:
        text = prefix + ": " + self.get_pred_message()
        return draw_text_on_image(
            image=frame,
            text=text,
            position=position,
            color=color,
            font_size=20,
        )

    def get_pred_message(self) -> str:
        if not any((
            self.start_time,
            self.end_time,
            self.inference_time,
            self.predictions
        )):
            return ""

        return ', '.join(
            [
                f"{pred['gloss']} ({pred['score']*100:.2f}%)"
                for pred in self.predictions
            ]
        )

    def __str__(self) -> str:
        if not any((
            self.start_time,
            self.end_time,
            self.inference_time,
            self.predictions
        )):
            return ""

        predictions = self.get_pred_message()
        message = "Sample start: {:.2f}s - end: {:.2f}s | Runtime: {:.2f}s | Predictions: {}"
        return message.format(self.start_time, self.end_time, self.inference_time, predictions)

    def merge_results(self, results: dict = None) -> dict:
        if results is None:
            results = {
                "start_time": [],
                "end_time": [],
                "inference_time": [],
                "prediction": [],
            }
        results["start_time"].append(self.start_time)
        results["end_time"].append(self.end_time)
        results["inference_time"].append(self.inference_time)
        results["prediction"].append(self.predictions)
        return results


def get_predictions(
    inputs: torch.Tensor,
    model: Union[ort.InferenceSession, AutoModel],
    id2gloss: dict,
    k: int = 3,
) -> Predictions:
    '''
    Get the top-k predictions.
    Parameters
    ----------
    inputs : torch.Tensor
        Model inputs (Time, Height, Width, Channels).
    model : Union[ort.InferenceSession, AutoModel]
        Model to get predictions from.
    id2gloss : dict
        Mapping of class indices to glosses.
    k : int, optional
        Number of predictions to return, by default 3.
    Returns
    -------
    tuple
        List of top-k predictions and inference time.
    '''
    if inputs is None:
        return Predictions()

    # Get logits
    start_time = time()
    if isinstance(model, ort.InferenceSession):
        inputs = inputs.cpu().numpy()
        logits = torch.from_numpy(model.run(None, {"pixel_values": inputs})[0])
    else:
        logits = model(inputs.to(model.device)).logits
    inference_time = time() - start_time

    # Get top-3 predictions
    topk_scores, topk_indices = torch.topk(logits, k, dim=1)
    topk_scores = torch.nn.functional.softmax(topk_scores, dim=1).squeeze().detach().numpy()
    topk_indices = topk_indices.squeeze().detach().numpy()
    predictions = [
        {
            'gloss': id2gloss[str(topk_indices[i])],
            'score': topk_scores[i],
        }
        for i in range(k)
    ]

    return Predictions(predictions=predictions, inference_time=inference_time)


def register_pipeline(model_config: ModelConfig) -> Pipeline:
    '''
    '''
    _, processor, model = load_model(model_config)

    if model_config.arch == "spoter":
        PIPELINE_REGISTRY.register_pipeline(
            "video-classification",
            pipeline_class=SPOTERGraphClassificationPipeline,
            pt_model=AutoModel,
            type="multimodal",
        )
        return SPOTERGraphClassificationPipeline(
            model=model,
            feature_extractor=processor,
        )

    if model_config.arch in ["sl_gcn", "dsta_slr"]:
        PIPELINE_REGISTRY.register_pipeline(
            "video-classification",
            pipeline_class=SLGCNGraphClassificationPipeline,
            pt_model=AutoModel,
            type="multimodal",
        )
        return SLGCNGraphClassificationPipeline(
            model=model,
            feature_extractor=processor,
        )

    PIPELINE_REGISTRY.register_pipeline(
        "video-classification",
        pipeline_class=VideoClassificationPipeline,
        pt_model=AutoModelForVideoClassification,
        type="multimodal",
    )
    return VideoClassificationPipeline(
        model=model,
        image_processor=processor,
    )


def load_pipeline(
    model_config: ModelConfig,
    inference_config: InferenceConfig,
) -> Pipeline:
    '''
    '''
    if model_config.arch in POSE_BASED_MODELS:
        return pipeline(
            "video-classification",
            model=model_config.pretrained,
            feature_extractor=model_config.pretrained,
            device=inference_config.device,
            model_kwargs={
                "cache_dir": inference_config.cache_dir,
            },
            trust_remote_code=True,
            use_onnx=inference_config.use_onnx,
            top_k=inference_config.top_k,
            bone_stream=inference_config.bone_stream,
            motion_stream=inference_config.motion_stream,
        )

    return pipeline(
        "video-classification",
        model=model_config.pretrained,
        image_processor=model_config.pretrained,
        device=inference_config.device,
        model_kwargs={
            "cache_dir": inference_config.cache_dir,
        },
        trust_remote_code=True,
        use_onnx=inference_config.use_onnx,
        top_k=inference_config.top_k,
    )


def get_input_shape(
    arch: str,
    processor: Union[ImageProcessingMixin, FeatureExtractionMixin],
    batch_size: int = 1,
) -> tuple:
    '''
    Get the input shape for the model.
    Parameters
    ----------
    processor : Union[ImageProcessingMixin, FeatureExtractionMixin]
        Model processor.
    batch_size : int, optional
        Batch size, by default 1.
    Returns
    -------
    tuple
        Input shape.
    '''
    if arch in RGB_BASED_MODELS:
        return (
            batch_size,
            processor.num_frames,
            3,
            processor.size["height"],
            processor.size["width"]
        )
    elif arch in POSE_BASED_MODELS:
        if arch == "spoter":
            return (
                batch_size,
                processor.num_frames,
                processor.num_points,
                processor.in_channels,
            )
        elif arch in ["sl_gcn", "dsta_slr"]:
            return (
                batch_size,
                processor.in_channels,
                processor.window_size,
                processor.num_points,
                processor.num_people,
            )
        else:
            logging.error(f"Model {arch} is not supported")
            exit(1)
    else:
        logging.error(f"Model {arch} is not supported")
        exit(1)
