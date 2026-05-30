from configs import TransformConfig
from transformers import ImageProcessingMixin, FeatureExtractionMixin
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample,
    create_video_transform,
    Div255,
)
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    CenterCrop,
)
from .augmentations import (
    SPOTERRandomAugment,
    SPOTERGaussianNoise,
    SLGCNAugment,
)
from .transforms import (
    SPOTERShift,
    SPOTERJointSelect,
    SPOTERTensorToDict,
    SPOTERSingleBodyDictNormalize,
    SPOTERSingleHandDictNormalize,
    SPOTERDictToTensor,
    SLGCNJointSelect,
    SLGCNPad,
    SLGCNNormalize,
    SLGCNBoneStream,
    SLGCNMotionStream,
    NumPyToTensor,
    PoseExtract,
)


def get_rgb_transforms(
    split: str,
    processor: ImageProcessingMixin,
    transform_config: TransformConfig,
) -> Compose:
    num_frames = processor.num_frames
    mean = processor.mean
    std = processor.std
    max_resize_size = processor.max_resize_size
    min_resize_size = processor.min_resize_size
    crop_size = (processor.size["height"], processor.size["width"])
    horizontal_flip_prob = transform_config.horizontal_flip_prob
    aug_type = transform_config.aug_type
    aug_paras = transform_config.aug_paras

    if split == "train":
        transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            Div255(),
                            create_video_transform(
                                mode="train",
                                num_samples=num_frames,
                                convert_to_float=False,
                                video_mean=mean,
                                video_std=std,
                                max_size=max_resize_size,
                                min_size=min_resize_size,
                                crop_size=crop_size,
                                horizontal_flip_prob=horizontal_flip_prob,
                                aug_type=aug_type,
                                aug_paras=aug_paras
                            )
                        ]
                    ),
                ),
            ]
        )
        clip_sampler_type = "random"
    else:
        transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames),
                            Div255(),
                            Normalize(mean, std),
                            Resize(min_resize_size),
                            CenterCrop(crop_size),
                        ]
                    ),
                ),
            ]
        )
        clip_sampler_type = "uniform"

    return transform, clip_sampler_type


def get_pose_transforms(
    split: str,
    processor: FeatureExtractionMixin,
    transform_config: TransformConfig,
) -> Compose:
    if processor.arch == "spoter":
        return _get_spoter_transforms(split, processor, transform_config)
    if processor.arch in ["sl_gcn", "dsta_slr"]:
        return _get_sl_gcn_transforms(split, processor, transform_config)
    return Compose([])


def _get_spoter_transforms(
    split: str,
    processor: FeatureExtractionMixin,
    transform_config: TransformConfig,
) -> Compose:
    transforms = [
        PoseExtract(),
        SPOTERJointSelect(),
        SPOTERTensorToDict(),
    ]

    if split == "train" and transform_config.aug_prob > 0:
        transforms.append(SPOTERRandomAugment(transform_config.aug_prob))

    transforms.extend([
        SPOTERSingleBodyDictNormalize(),
        SPOTERSingleHandDictNormalize(),
        SPOTERDictToTensor(),
        SPOTERShift()
    ])

    if split == "train" and transform_config.add_gaussian_noise:
        transforms.append(
            SPOTERGaussianNoise(
                transform_config.gaussian_noise_mean,
                transform_config.gaussian_noise_std,
            )
        )

    return Compose(transforms)


def _get_sl_gcn_transforms(
    split: str,
    processor: FeatureExtractionMixin,
    transform_config: TransformConfig,
) -> Compose:
    transforms = [PoseExtract()]

    if split == "train":
        transforms.append(
            SLGCNAugment(
                aug_prob=transform_config.aug_prob,
                rotation_std=transform_config.rotation_std,
                shear_std=transform_config.shear_std,
                scale_std=transform_config.scale_std,
            )
        )

    transforms.extend(
        [
            SLGCNJointSelect(processor.num_points),
            SLGCNPad(processor.num_frames),
        ]
    )

    if processor.bone_stream:
        transforms.append(SLGCNBoneStream())
    if processor.motion_stream:
        transforms.append(SLGCNMotionStream())

    transforms.extend(
        [
            SLGCNNormalize(processor.is_vector),
            NumPyToTensor(),
        ]
    )
    return Compose(transforms)
