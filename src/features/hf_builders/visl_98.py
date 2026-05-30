# Copyright 2023 Thinh T. Duong
import datasets
import pandas as pd
from pathlib import Path
from typing import Generator


logger = datasets.logging.get_logger(__name__)


_CITATION = """
"""
_DESCRIPTION = """
"""
_HOMEPAGE = ""
_REPO_URL = "https://huggingface.co/datasets/vsltranslation/visl_98/resolve/main"
_URLS = {
    "meta": f"{_REPO_URL}/meta.json",
    "gloss2id": f"{_REPO_URL}/gloss.csv",
    "data": f"{_REPO_URL}/data.zip",
}


def load_visl_98(metadata_file: Path, gloss2id_file: Path, data_dir: Path) -> tuple:
    gloss2id = pd.read_csv(
        gloss2id_file,
        delimiter=",",
        names=["id", "gloss"],
        index_col="gloss",
    )
    gloss2id = gloss2id.to_dict()["id"]

    df = pd.read_json(
        metadata_file,
        encoding='utf-8',
        dtype={
            "video_id": "string",
            "signer_id": "string",
            "fps": "int",
            "resolution": "int",
            "num_of_frames": "int",
            "length": "float",
            "gloss": "string",
            "english_gloss": "string",
        }
    )
    df["gloss_id"] = df["gloss"].map(gloss2id)
    df["video"] = df["video_id"].apply(lambda x: str(data_dir / f"{x}.mp4"))
    df["pose"] = df["video_id"].apply(lambda x: str(data_dir / f"{x}.pose"))

    test_set_signer_ids = ["001", "009", "014"]
    train_df = df[~df["signer_id"].isin(test_set_signer_ids)]
    test_df = df[df["signer_id"].isin(test_set_signer_ids)]

    return train_df, test_df, gloss2id


class VISL98Config(datasets.BuilderConfig):
    """VISL-98 configuration."""

    def __init__(self, name, **kwargs):
        """
        Parameters
        ----------
        name : str
            Name of subset.
        kwargs : dict
            Keyword arguments.
        """
        super(VISL98Config, self).__init__(
            name=name,
            version=datasets.Version("1.0.0"),
            description=_DESCRIPTION,
            **kwargs,
        )


class VISL98(datasets.GeneratorBasedBuilder):
    """VISL-98 dataset."""
    BUILDER_CONFIGS = [
        VISL98Config(name="vsl"),
    ]
    DEFAULT_CONFIG_NAME = "vsl"

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "video_id": datasets.Value("string"),
            "signer_id": datasets.Value("string"),
            "fps": datasets.Value("int16"),
            "resolution": datasets.Value("int16"),
            "num_of_frames": datasets.Value("int16"),
            "length": datasets.Value("float32"),
            "gloss": datasets.Value("string"),
            "english_gloss": datasets.Value("string"),
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
        self, dl_manager: datasets.DownloadManager
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
        meta_file = Path(dl_manager.download(_URLS["meta"]))
        gloss2id_file = Path(dl_manager.download(_URLS["gloss2id"]))
        data_dir = Path(dl_manager.download_and_extract(_URLS["data"])) / "data"

        train_df, test_df, _ = load_visl_98(meta_file, gloss2id_file, data_dir)

        split_dict = {
            datasets.Split.TRAIN: train_df,
            datasets.Split.TEST: test_df,
        }

        return [
            datasets.SplitGenerator(name=name, gen_kwargs={"split_df": split_df})
            for name, split_df in split_dict.items()
        ]

    def _generate_examples(
        self, split_df: pd.DataFrame,
    ) -> Generator[tuple[int, dict], None, None]:
        """
        Generate examples from metadata.

        Parameters
        ----------
        split_df : str
            Split dataframe.

        Yields
        ------
        tuple[int, dict]
            Sample.
        """
        for i, sample in enumerate(split_df.itertuples()):
            if Path(sample.video).exists() and Path(sample.pose).exists():
                yield i, {
                    "video_id": str(sample.video_id).zfill(6),
                    "signer_id": str(sample.signer_id).zfill(3),
                    "fps": sample.fps,
                    "resolution": sample.resolution,
                    "num_of_frames": sample.num_of_frames,
                    "length": sample.length,
                    "gloss": sample.gloss,
                    "english_gloss": sample.english_gloss,
                    "gloss_id": sample.gloss_id,
                    "video": sample.video,
                    "pose": sample.pose,
                }
