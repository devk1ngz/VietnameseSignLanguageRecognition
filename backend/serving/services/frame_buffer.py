"""Sliding window buffer cho keypoint frames (mot buffer moi phien).

Giu song song frame SPOTER (54 khop x,y) va frame SL-GCN (27 khop x,y,confidence) de
phuc vu late-fusion o luong realtime. Frame SL-GCN la TUY CHON: client cu chi gui SPOTER
van chay duoc (fusion tu lui ve SPOTER-only).
"""

from collections import deque

import numpy as np

from serving.models.schemas import NUM_JOINTS, SLGCN_NUM_JOINTS


class FrameBuffer:
    """Cua so truot cho keypoint.

    - maxlen = seq_len (mac dinh 70) cho ca hai buffer
    - stride: so frame moi tich luy truoc khi kich hoat inference tiep theo
    - Buffer SL-GCN chay song song, chi day khi co frame SL-GCN.
    """

    def __init__(self, seq_len: int, stride: int):
        self.seq_len = seq_len
        self.stride = stride
        self._buf: deque[np.ndarray] = deque(maxlen=seq_len)
        self._slgcn_buf: deque[np.ndarray | None] = deque(maxlen=seq_len)
        self._frames_since_last = 0

    def push(
        self, frame: np.ndarray, slgcn_frame: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None] | None:
        """Them 1 frame SPOTER ([54,2] hoac [108]) + (tuy chon) 1 frame SL-GCN ([27,3] hoac [81]).

        Tra ve (snapshot SPOTER (seq_len,54,2), snapshot SL-GCN (T,27,3) hoac None) khi buffer
        day va du stride frame moi; nguoc lai None. Snapshot SL-GCN = None neu chua co du frame
        SL-GCN nao (client khong gui) -> luong goi fusion se tu lui ve SPOTER-only.
        """
        self._buf.append(frame.reshape(NUM_JOINTS, 2))
        self._slgcn_buf.append(
            slgcn_frame.reshape(SLGCN_NUM_JOINTS, 3) if slgcn_frame is not None else None
        )
        self._frames_since_last += 1

        if len(self._buf) == self.seq_len and self._frames_since_last >= self.stride:
            self._frames_since_last = 0
            spoter_snap = np.array(self._buf)
            slgcn_frames = [f for f in self._slgcn_buf if f is not None]
            # Chi fuse khi co du frame SL-GCN cho toan cua so (dam bao dong bo thoi gian).
            slgcn_snap = (
                np.array(slgcn_frames) if len(slgcn_frames) == len(self._buf) else None
            )
            return spoter_snap, slgcn_snap
        return None

    def reset(self) -> None:
        self._buf.clear()
        self._slgcn_buf.clear()
        self._frames_since_last = 0

    @property
    def fill_ratio(self) -> float:
        return len(self._buf) / self.seq_len
