"""
features/visl_98_dataset.py
---------------------------
Dataset class for the VISL-98 dataset (98-class Vietnamese Sign Language).

The VISL-98 dataset follows the same directory layout as VISL-400 but
contains 98 gloss categories.  Loading is delegated to the shared
``load_visl_400`` HuggingFace builder because the JSON / CSV schema is
identical; only the gloss vocabulary differs.
"""

from pathlib import Path
from typing import Tuple
from datasets import DatasetDict, Dataset
from .base_dataset import BaseDataset
from .hf_builders import load_visl_400


class VISL98Dataset(BaseDataset):
    """
    Dataset wrapper for the local VISL-98 split.

    Expected directory layout::

        <data_dir>/
          cam_1/          # video files for camera 1
          cam_2/
          cam_3/
          cam_1.json      # metadata JSON for camera 1
          cam_2.json
          cam_3.json
          gloss.csv       # gloss-id mapping (98 rows)

    The ``subset`` field in :class:`~configs.DataConfig` controls which
    cameras are used (e.g. ``"vsl_1_2_3"`` → cameras 1, 2 and 3).
    """

    def _load_from_local(
        self,
        data_dir: str,
        subset: str,
    ) -> Tuple[DatasetDict, dict, dict]:
        """
        Load VISL-98 data from a local directory.

        Parameters
        ----------
        data_dir : str
            Root directory that contains the camera sub-folders and JSON
            metadata files.
        subset : str
            Camera subset string in the form ``"vsl_<cam_ids…>"``
            (e.g. ``"vsl_1_2_3"``).  The camera IDs are extracted from the
            trailing underscore-separated tokens.

        Returns
        -------
        tuple[DatasetDict, dict, dict]
            ``(dataset, gloss2id, id2gloss)`` where *dataset* is a
            HuggingFace :class:`~datasets.DatasetDict` with ``"train"``,
            ``"validation"`` and ``"test"`` splits.
        """
        data_dir = Path(data_dir)
        # Parse camera IDs from the subset string (e.g. "vsl_1_2_3" → ["1","2","3"])
        cams = subset.split("_")[1:] if subset is not None else ["1"]
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
