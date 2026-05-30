# Copyright 2023 Thinh T. Duong
import datasets
import pandas as pd
from pathlib import Path
from typing import Generator, Dict, Tuple, Any


_CITATION = """
"""
_DESCRIPTION = """
"""
_HOMEPAGE = """
"""

_REPO_URL = "https://huggingface.co/datasets/vsltranslation/vsl-400/resolve/main"
_URLS = {
    "meta": _REPO_URL + "/cam_{cam}.json",
    "data": _REPO_URL + "/cam_{cam}.zip",
    "gloss2id": _REPO_URL + "/gloss.csv",
}


def load_visl_400(
    data_dict: Dict[str, Dict[str, Path]],
    gloss2id_file: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    gloss2id = pd.read_csv(
        gloss2id_file,
        delimiter=",",
        names=["id", "gloss"],
        index_col="gloss",
    )
    gloss2id = gloss2id.to_dict()["id"]

    dfs = []
    for cam, file_dict in data_dict.items():
        metadata_file = file_dict["meta"]
        data_dir = file_dict["data"] / cam
        df = pd.read_json(
            metadata_file,
            encoding='utf-8',
            dtype={
                "video_id": "string",
                "signer_id": "string",
                "fps": "int",
                "resolution": "int",
                # "num_of_frames": "int",
                "length": "float",
                "gloss": "string",
                # "english_gloss": "string",
            }
        )
        df["cam_id"] = cam[-1]
        df["gloss_id"] = df["gloss"].map(gloss2id)
        df["video"] = df["video_id"].apply(lambda x: str(data_dir / f"{x}.mp4"))
        df["pose"] = df["video_id"].apply(lambda x: str(data_dir / f"{x}.pose"))
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    common_signer_ids = {
        "020": ("1", "2", "3"),
        "014": ("2", "3", "1"),
        "015": ("3", "1", "2"),
    }
    val_unique_signer_ids = ["007"]
    test_unique_signer_ids = ["024"]
    val_test_common_signer_ids = ["009"]
    train_not_unique_signer_ids = (
        val_unique_signer_ids
        + test_unique_signer_ids
        + val_test_common_signer_ids
        + list(common_signer_ids.keys())
    )
    cam_ids = list(df["cam_id"].unique())

    val_test_df = df[df["signer_id"].isin(val_test_common_signer_ids)]
    val_df = (
        val_test_df
        .groupby(["gloss_id", "cam_id"], group_keys=False)
        .apply(lambda x: x.sample(frac=0.5, random_state=42))
    )
    test_df = val_test_df[~val_test_df.index.isin(val_df.index)]

    train_df = df[~df["signer_id"].isin(train_not_unique_signer_ids)]
    val_df = pd.concat(
        [
            df[df["signer_id"].isin(val_unique_signer_ids)],
            val_df,
        ],
        ignore_index=True,
    )
    test_df = pd.concat(
        [
            df[df["signer_id"].isin(test_unique_signer_ids)],
            test_df,
        ],
        ignore_index=True,
    )

    for signer_id, (train_cam, val_cam, test_cam) in common_signer_ids.items():
        if train_cam in cam_ids:
            train_df = pd.concat(
                [
                    df[(df["signer_id"] == signer_id) & (df["cam_id"] == train_cam)],
                    train_df,
                ],
                ignore_index=True,
            )
        if val_cam in cam_ids:
            val_df = pd.concat(
                [
                    df[(df["signer_id"] == signer_id) & (df["cam_id"] == val_cam)],
                    val_df,
                ],
                ignore_index=True,
            )
        if test_cam in cam_ids:
            test_df = pd.concat(
                [
                    df[(df["signer_id"] == signer_id) & (df["cam_id"] == test_cam)],
                    test_df,
                ],
                ignore_index=True,
            )

    return train_df, val_df, test_df, gloss2id


class VISL400Config(datasets.BuilderConfig):
    """VISL-400 configuration."""

    def __init__(self, name, **kwargs) -> None:
        """
        Parameters
        ----------
        name : str
            Name of subset.
        kwargs : dict
            Keyword arguments.
        """
        super(VISL400Config, self).__init__(
            name=name,
            version=datasets.Version("1.0.0"),
            description=_DESCRIPTION,
            **kwargs,
        )


class VISL400(datasets.GeneratorBasedBuilder):
    """VISL-400 dataset."""
    BUILDER_CONFIGS = [
        VISL400Config(name="cam_1"),
        VISL400Config(name="cam_2"),
        VISL400Config(name="cam_3"),
        VISL400Config(name="cam_1_2"),
        VISL400Config(name="cam_1_3"),
        VISL400Config(name="cam_2_3"),
        VISL400Config(name="cam_1_2_3"),
    ]
    DEFAULT_CONFIG_NAME = "cam_1"

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "video_id": datasets.Value("string"),
            "signer_id": datasets.Value("string"),
            "fps": datasets.Value("int16"),
            "resolution": datasets.Value("int16"),
            # "num_of_frames": datasets.Value("int16"),
            "length": datasets.Value("float32"),
            "gloss": datasets.Value("string"),
            # "english_gloss": datasets.Value("string"),
            "cam_id": datasets.Value("string"),
            "gloss_id": datasets.Value("int16"),
            "video": datasets.Value("string"),
            "pose": datasets.Value("string"),
        })

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(
        self,
        dl_manager: datasets.DownloadManager,
    ) -> list[datasets.SplitGenerator]:
        """
        Get splits.
        Parameters
        ----------
        dl_manager : datasets.DownloadManager
            Download manager.
        Returns
        -------
        list[datasets.SplitGenerator]
            Split generators.
        """
        cams = self.config.name.split("_")[1:]
        gloss2id_file = Path(dl_manager.download(_URLS["gloss2id"]))

        data_dict = {}
        for cam in cams:
            data_dict[f"cam_{cam}"] = {
                "meta": Path(dl_manager.download(_URLS["meta"].format(cam=cam))),
                "data": Path(dl_manager.download_and_extract(_URLS["data"].format(cam=cam))),
            }
        train_df, val_df, test_df, _ = load_visl_400(data_dict, gloss2id_file)

        split_dict = {
            datasets.Split.TRAIN: train_df,
            datasets.Split.VALIDATION: val_df,
            datasets.Split.TEST: test_df,
        }

        return [
            datasets.SplitGenerator(name=name, gen_kwargs={"split_df": split_df})
            for name, split_df in split_dict.items()
        ]

    def _generate_examples(
        self,
        split_df: pd.DataFrame,
    ) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        """
        Generate examples from metadata.
        Parameters
        ----------
        split_df : str
            Split dataframe.
        video_dirs : list[str]
            List of video directories.
        Yields
        ------
        tuple[int, dict]
            Sample.
        """
        for i, sample in enumerate(split_df.itertuples()):
            if Path(sample.video).exists() and Path(sample.pose).exists():
                yield i, {
                    "video_id": sample.video_id,
                    "signer_id": sample.signer_id,
                    "fps": sample.fps,
                    "resolution": sample.resolution,
                    # "num_of_frames": sample.num_of_frames,
                    "length": sample.length,
                    "gloss": sample.gloss,
                    # "english_gloss": sample.english_gloss,
                    "cam_id": sample.cam_id,
                    "gloss_id": sample.gloss_id,
                    "video": sample.video,
                    "pose": sample.pose,
                }
