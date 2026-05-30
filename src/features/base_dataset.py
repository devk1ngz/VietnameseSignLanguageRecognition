import os
import pandas as pd
from typing import Union
from configs import DataConfig
from datasets import load_dataset
from utils import exists_on_hf
from huggingface_hub import hf_hub_download
from .utils import get_rgb_transforms, get_pose_transforms
from transformers import ImageProcessingMixin, FeatureExtractionMixin
from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler
from .pose_dataset import PoseDataset


class BaseDataset:
    def __init__(
        self,
        data_config: DataConfig,
        **kwargs,
    ) -> None:
        self.data_config = data_config
        self.dataset, self.gloss2id, self.id2gloss = self._load()
        if self.data_config.debug:
            for split in self.dataset.keys():
                self.dataset[split] = self.dataset[split].take(10)

    def _load(self) -> tuple:
        if exists_on_hf(self.data_config.data_dir, "dataset"):
            dataset, gloss2id, id2gloss = self._load_from_hf(
                repo_id=self.data_config.data_dir,
                subset=self.data_config.subset,
            )
        else:
            dataset, gloss2id, id2gloss = self._load_from_local(
                data_dir=self.data_config.data_dir,
                subset=self.data_config.subset,
            )
        dataset = dataset.select_columns(
            ["video_id", "resolution", "gloss_id", "video", "pose"]
        )
        dataset = dataset.shuffle(seed=42)
        return dataset, gloss2id, id2gloss

    def _load_from_local(self, data_dir: str, subset: str = None) -> tuple:
        raise NotImplementedError

    def _load_from_hf(
        self,
        repo_id: str,
        subset: str = None,
        renamed_columns: dict = None,
    ) -> tuple:
        """
        Load dataset from HuggingFace.

        Parameters
        ----------
        repo_id : str
            Dataset identifier on HuggingFace.
        subset : str, optional
            Subset name, by default None.
        renamed_columns : dict, optional
            Dictionary of renamed columns, by default None.

        Returns
        -------
        datasets.Dataset
            Dataset.
        """
        dataset = load_dataset(
            repo_id, subset,
            num_proc=os.cpu_count(),
            cache_dir="data/external/huggingface",
            trust_remote_code=True,
        )

        if renamed_columns is not None:
            dataset = dataset.rename_columns(renamed_columns)

        gloss2id_file = hf_hub_download(
            repo_id=repo_id,
            filename="gloss.csv",
            repo_type="dataset",
            cache_dir="data/external/huggingface",
        )
        gloss2id = pd.read_csv(
            gloss2id_file,
            delimiter=",",
            names=["id", "gloss"],
            index_col="gloss",
        )
        gloss2id = gloss2id.to_dict()["id"]
        id2gloss = {v: k for k, v in gloss2id.items()}

        return dataset, gloss2id, id2gloss

    def get_split(
        self,
        split: str,
        processor: Union[ImageProcessingMixin, FeatureExtractionMixin],
    ) -> LabeledVideoDataset:
        if self.data_config.modality == "rgb":
            return self.__get_rgb_split(split, processor)
        return self.__get_pose_split(split, processor)

    def __get_rgb_split(
        self,
        split: str,
        processor: ImageProcessingMixin,
    ) -> LabeledVideoDataset:
        transform, clip_sampler_type = get_rgb_transforms(split, processor, self.data_config.transform)

        labeled_video_paths = [
            (sample["video"], {'label': sample["gloss_id"]})
            for sample in self.dataset[split]
        ]

        sample_rate = self.data_config.transform.sample_rate
        fps = self.data_config.fps
        clip_duration = processor.num_frames * sample_rate / fps

        return LabeledVideoDataset(
            labeled_video_paths=labeled_video_paths,
            clip_sampler=make_clip_sampler(clip_sampler_type, clip_duration),
            transform=transform,
            decode_audio=False,
        )

    def __get_pose_split(
        self,
        split: str,
        processor: FeatureExtractionMixin,
    ) -> PoseDataset:
        transform = get_pose_transforms(split, processor, self.data_config.transform)
        return PoseDataset(
            dataset=self.dataset[split],
            transforms=transform,
        )
