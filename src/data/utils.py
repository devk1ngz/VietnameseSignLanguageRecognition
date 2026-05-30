import numpy as np
from mediapipe.python.solutions import pose
from visualization import draw_text_on_image


class Arm:
    def __init__(
        self,
        side: str,
        visibility: float = 0.5,
    ) -> None:
        if side == "left":
            self.shoulde_idx = pose.PoseLandmark.LEFT_SHOULDER.value
            self.elbow_idx = pose.PoseLandmark.LEFT_ELBOW.value
            self.wrist_idx = pose.PoseLandmark.LEFT_WRIST.value
        elif side == "right":
            self.shoulde_idx = pose.PoseLandmark.RIGHT_SHOULDER.value
            self.elbow_idx = pose.PoseLandmark.RIGHT_ELBOW.value
            self.wrist_idx = pose.PoseLandmark.RIGHT_WRIST.value
        else:
            raise ValueError("Side must be either 'left' or 'right'")
        self.visibility = visibility

        self.is_up = False
        self.num_up_frames = 0
        self.num_down_frames = 0
        self.start_time = 0
        self.end_time = 0
        self.shoulder = None
        self.elbow = None
        self.wrist = None
        self.angle = 0

    def reset_state(self) -> None:
        self.is_up = False
        self.num_up_frames = 0
        self.num_down_frames = 0
        self.start_time = 0
        self.end_time = 0
        self.shoulder = None
        self.elbow = None
        self.wrist = None
        self.angle = 0

    def set_pose(self, landmarks) -> bool:
        if landmarks[self.shoulde_idx].visibility < self.visibility:
            return False
        self.shoulder = (
            landmarks[self.shoulde_idx].x,
            landmarks[self.shoulde_idx].y,
        )

        if landmarks[self.elbow_idx].visibility < self.visibility:
            return False
        self.elbow = (
            landmarks[self.elbow_idx].x,
            landmarks[self.elbow_idx].y,
        )

        if landmarks[self.wrist_idx].visibility < self.visibility:
            return False
        self.wrist = (
            landmarks[self.wrist_idx].x,
            landmarks[self.wrist_idx].y,
        )

        self.angle = calculate_angle(self.shoulder, self.elbow, self.wrist)
        return True

    def visualize(
        self,
        frame: np.ndarray,
        position: tuple = (20, 50),
        prefix: str = "Angle",
        color: tuple = (0, 0, 255),
    ) -> np.ndarray:
        text = prefix + ": " + str(round(self.angle, 2))
        return draw_text_on_image(
            image=frame,
            text=text,
            position=position,
            color=color,
            font_size=20,
        )


def get_sample_timestamp(left_arm: Arm, right_arm: Arm) -> tuple:
    start_time, end_time = 0, 0
    left_arm_available = left_arm.start_time > 0 and left_arm.end_time > 0
    right_arm_available = right_arm.start_time > 0 and right_arm.end_time > 0

    if left_arm_available and right_arm.start_time == 0:
        start_time = left_arm.start_time
        end_time = left_arm.end_time
    if right_arm_available and left_arm.start_time == 0:
        start_time = right_arm.start_time
        end_time = right_arm.end_time
    if all((
        left_arm_available, not left_arm.is_up,
        right_arm_available, not right_arm.is_up,
    )):
        start_time = min(left_arm.start_time, right_arm.start_time)
        end_time = max(left_arm.end_time, right_arm.end_time)

    # Convert seconds to milliseconds
    start_time /= 1000
    end_time /= 1000
    return start_time, end_time


def calculate_angle(a: tuple, b: tuple, c: tuple) -> float:
    a = np.array(a)     # First
    b = np.array(b)     # Mid
    c = np.array(c)     # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    return 360 - angle if angle > 180 else angle


def ok_to_get_frame(
    arm: Arm,
    angle_threshold: int,
    min_num_up_frames: int,
    min_num_down_frames: int,
    current_time: int,
    delay: int,
) -> bool:
    if 0 < arm.angle < angle_threshold:
        if arm.is_up:
            arm.num_down_frames = 0
            arm.end_time = 0
        else:
            if arm.num_up_frames == min_num_up_frames:
                arm.is_up = True
                arm.num_up_frames = 0
            else:
                if arm.num_up_frames == 0:
                    arm.start_time = current_time - delay
                arm.num_up_frames += 1
                return False
    else:
        if arm.is_up:
            if arm.num_down_frames == min_num_down_frames:
                arm.is_up = False
                arm.num_down_frames = 0
            else:
                if arm.num_down_frames == 0:
                    arm.end_time = current_time + delay
                arm.num_down_frames += 1
                return True
        else:
            arm.num_up_frames = 0
            arm.start_time = 0

    return arm.is_up
