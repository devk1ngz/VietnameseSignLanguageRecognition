import cv2
import shutil
import logging
import numpy as np
import pandas as pd
from time import time
from traceback import format_exc
from argparse import Namespace
from transformers import Pipeline
from simple_parsing import ArgumentParser
from visualization import draw_text_on_image
from configs import ModelConfig, InferenceConfig
from utils import config_logger, POSE_BASED_MODELS
from data import Arm, get_sample_timestamp, ok_to_get_frame
from mediapipe.python.solutions import pose, drawing_utils, holistic
from tools import load_pipeline, Predictions


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Train a model on VSL",
        add_config_path_arg=True,
    )
    parser.add_arguments(ModelConfig, "model")
    parser.add_arguments(InferenceConfig, "inference")
    return parser.parse_args()


def inference(config: InferenceConfig, pipeline: Pipeline) -> None:
    # Load video
    source = str(config.source) if config.source.is_file() else 0
    cap = cv2.VideoCapture(source)
    if config.output_dir is not None:
        writer = cv2.VideoWriter(
            str(config.output_dir / "output.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            cap.get(cv2.CAP_PROP_FPS),
            (int(cap.get(3)), int(cap.get(4))),
        )

    # Init Mediapipe
    keypoints_detector = holistic.Holistic(
        model_complexity=0,
        min_detection_confidence=0.9,
    )

    # Init variables
    right_arm = Arm("right", config.visibility)
    left_arm = Arm("left", config.visibility)
    data = []
    results = None
    predictions = Predictions()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Recolor image to RGB, because mp processes on RGB image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False

        # Make detections
        detection_results = keypoints_detector.process(frame)

        # Recolor image back to BGR, because cv2 processes on BGR image
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = detection_results.pose_landmarks.landmark
        except Exception:
            continue

        left_arm.set_pose(landmarks)
        right_arm.set_pose(landmarks)

        # Check if arms are up or down
        left_arm_ok_to_get_frame = ok_to_get_frame(
            arm=left_arm,
            angle_threshold=config.angle_threshold,
            min_num_up_frames=config.min_num_up_frames,
            min_num_down_frames=config.min_num_down_frames,
            current_time=cap.get(cv2.CAP_PROP_POS_MSEC),
            delay=config.delay,
        )
        right_arm_ok_to_get_frame = ok_to_get_frame(
            arm=right_arm,
            angle_threshold=config.angle_threshold,
            min_num_up_frames=config.min_num_up_frames,
            min_num_down_frames=config.min_num_down_frames,
            current_time=cap.get(cv2.CAP_PROP_POS_MSEC),
            delay=config.delay,
        )
        if left_arm_ok_to_get_frame or right_arm_ok_to_get_frame:
            # logging.info("Frame added to the list")
            predictions = Predictions()
            data.append(detection_results if config.use_pose_model else frame)

        # Calculate the start and end time of sign
        start_time, end_time = get_sample_timestamp(left_arm, right_arm)

        # Convert from miliseconds to seconds
        start_time /= 1_000
        end_time /= 1_000

        # logging.info(f"start_time: {start_time} - end_time: {end_time}")
        # logging.info(f"\tLeft arm: {left_arm.start_time} - {left_arm.end_time} - {left_arm.is_up}")
        # logging.info(f"\tRight arm: {right_arm.start_time} - {right_arm.end_time} - {right_arm.is_up}")

        if start_time != 0 and end_time != 0:
            # Render waiting screen
            if config.visualize:
                wait_frame = draw_text_on_image(
                    np.zeros_like(frame),
                    text="Please wait for the prediction...",
                    position=(20, 20),
                    color=(255, 255, 255),
                    font_size=20,
                )
                cv2.imshow("Video Visualization", wait_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            start_inference_time = time()
            predictions = Predictions(predictions=pipeline(np.array(data)))
            predictions.inference_time = time() - start_inference_time

            predictions.start_time = start_time
            predictions.end_time = end_time
            logging.info(str(predictions))
            results = predictions.merge_results(results)

            # Reset variables
            start_time = 0
            end_time = 0
            left_arm.reset_state()
            right_arm.reset_state()
            data = []

        # Render detections
        frame = left_arm.visualize(frame, (20, 10), "Left arm angle")
        frame = right_arm.visualize(frame, (20, 40), "Right arm angle")
        frame = predictions.visualize(frame, (20, 70))
        if config.show_skeleton:
            drawing_utils.draw_landmarks(
                frame,
                detection_results.pose_landmarks,
                pose.POSE_CONNECTIONS
            )

        if config.output_dir is not None:
            writer.write(frame)

        if config.visualize:
            cv2.imshow("Video Visualization", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    if config.output_dir is not None:
        writer.release()
        logging.info(f"Video is recorded and saved to {config.output_dir / 'output.avi'}")
        pd.DataFrame(results).to_csv(config.output_dir / "results.csv", index=False)
        logging.info(f"Results saved to {config.output_dir / 'results.csv'}")


def main(args: Namespace) -> None:
    model_config = args.model
    logging.info(model_config)
    inference_config = args.inference
    logging.info(inference_config)

    if model_config.arch in POSE_BASED_MODELS:
        inference_config.use_pose_model = True
    else:
        inference_config.use_pose_model = False

    pipeline = load_pipeline(model_config, inference_config)
    logging.info("Pipeline loaded")

    inference(inference_config, pipeline)
    logging.info("Inference completed")


if __name__ == "__main__":
    try:
        args = get_args()

        config_logger(args.inference.output_dir / "inference.log")
        logging.info(f"Config file loaded from {args.config_path[0]}")

        shutil.copy(args.config_path[0], args.inference.output_dir / "inference.yaml")
        logging.info(f"Config file saved to {args.inference.output_dir}")

        main(args=args)
    except Exception:
        print(format_exc())
