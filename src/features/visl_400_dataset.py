from pathlib import Path
from typing import Tuple
from datasets import DatasetDict, Dataset
from .base_dataset import BaseDataset
from .hf_builders import load_visl_400


class VISL400Dataset(BaseDataset):
    def _load_from_local(
        self,
        data_dir: str,
        subset: str,
    ) -> Tuple[DatasetDict, dict, dict]:
        data_dir = Path(data_dir)
        cams = subset.split("_")[1:]
        gloss2id_file = data_dir / "gloss.csv"

        data_dict = {}
        for cam in cams:
            data_dict[f"cam_{cam}"] = {
                "meta": data_dir / f"cam_{cam}.json",
                "data": data_dir,
            }

        train_df, val_df, test_df, gloss2id = load_visl_400(data_dict, gloss2id_file)
        id2gloss = {v: k for k, v in gloss2id.items()}

        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df),
            "test": Dataset.from_pandas(test_df),
        })

        return dataset, gloss2id, id2gloss
