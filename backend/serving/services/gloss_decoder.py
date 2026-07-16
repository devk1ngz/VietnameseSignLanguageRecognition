"""Chuyen class index -> gloss string, loc theo confidence.

Nhan doc tu recognition/experiments/gloss.csv (dinh dang `id,gloss`, khong header, 400 dong).
"""

import csv
from pathlib import Path

from serving.config import settings
from serving.utils.logger import get_logger

logger = get_logger(__name__)

UNKNOWN = "<UNK>"


def dedupe_consecutive(glosses: list[str]) -> list[str]:
    """Bo cac tu trung LIEN KE (cung mot ky hieu bi nhan dang nhieu lan lien tiep)."""
    return [g for i, g in enumerate(glosses) if i == 0 or g != glosses[i - 1]]


class GlossDecoder:
    def __init__(self, gloss_csv_path: Path | None = None):
        self._path = gloss_csv_path or settings.gloss_csv_path
        self._map: dict[int, str] = {}

    def load(self) -> None:
        with open(self._path, newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if len(row) >= 2:
                    self._map[int(row[0])] = row[1]
        logger.info("Da nap %d nhan gloss tu %s", len(self._map), self._path)

    def decode(self, class_idx: int, confidence: float) -> str | None:
        """Tra ve gloss neu confidence dat nguong, nguoc lai None."""
        if confidence < settings.spoter_confidence_threshold:
            return None
        return self._map.get(class_idx, UNKNOWN)

    def label(self, class_idx: int) -> str:
        """Tra ve gloss tho (khong loc confidence) — nguoi goi tu ap nguong rieng (vd video)."""
        return self._map.get(class_idx, UNKNOWN)
