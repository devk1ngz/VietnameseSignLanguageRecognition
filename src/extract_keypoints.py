import logging
import subprocess
from tqdm import tqdm
from glob import glob
from pathlib import Path
from argparse import Namespace, ArgumentParser
from utils import config_logger, VIDEO_EXTENSIONS


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--video_dir",
        type=str,
        help="Directory containing videos to extract keypoints from",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing keypoints",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse the order of the keypoints",
    )
    return parser.parse_args()


def main(args: Namespace) -> None:
    logging.info(f"Extracting keypoints from videos in {args.video_dir}")
    logging.info(f"Overwrite existing keypoints: {args.overwrite}")

    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(glob(f"{args.video_dir}/*{ext}"))
    videos = sorted(videos, reverse=args.reverse)
    num_videos = len(videos)
    logging.info(f"Found {num_videos} videos")

    for video in tqdm(videos):
        video = Path(video)
        pose = video.with_suffix(".pose")

        if args.overwrite or not pose.exists():
            subprocess.run(
                [
                    "video_to_pose",
                    "--format",
                    "mediapipe",
                    "-i",
                    str(video),
                    "-o",
                    str(pose),
                ]
            )

    logging.info("Finished extracting keypoints")


if __name__ == "__main__":
    args = get_args()
    config_logger()
    main(args=args)
