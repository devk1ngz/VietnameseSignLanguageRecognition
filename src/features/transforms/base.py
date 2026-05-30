from pathlib import Path
from pose_format import Pose
from pose_format.utils.holistic import load_holistic
from typing import Dict, Any, Union


class PoseExtract:
    def __call__(self, inputs: Union[Dict[str, Any], str, Path]) -> Pose:
        if isinstance(inputs, (str, Path)):
            with open(inputs, "rb") as f:
                pose = Pose.read(f.read())
        else:
            pose = load_holistic(
                frames=inputs["frames"],
                fps=inputs["fps"],
                width=inputs["width"],
                height=inputs["height"],
                progress=False,
            )
        return pose
