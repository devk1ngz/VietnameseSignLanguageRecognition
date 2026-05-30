import torch
from configs import DataConfig
from features import BaseDataset, VISL98Dataset, VISL400Dataset


def load_dataset(data_config: DataConfig) -> BaseDataset:
    '''
    '''
    datasets = {
        "visl_98": VISL98Dataset,
        "visl_400": VISL400Dataset,
    }
    return datasets[data_config.dataset](data_config)


def rgb_collate_fn(examples) -> dict:
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def pose_collate_fn(examples) -> dict:
    # permute to (num_frames, num_channels, height, width)
    poses = torch.stack([example["pose"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"poses": poses, "labels": labels}
