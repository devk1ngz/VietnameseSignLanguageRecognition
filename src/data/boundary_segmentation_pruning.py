import sys
import logging
from pathlib import Path
import pandas as pd
import cv2

sys.path.append(str(Path().cwd() / 'src'))
from utils import config_logger

sys.path.append(str(Path().cwd() / 'src/configs'))
from arguments import ProcessRecordedVideosArguments

logger = logging.getLogger(__name__)

def log_video_info(input_video):
    logger.info('--VIDEO INFO')
    # Load video
    cap = cv2.VideoCapture(str(input_video))
    
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4)) 
    resolution = '{}:{}'.format(frame_width, frame_height) 
    
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info("Video resolution: {}.".format(resolution))
    logger.info('FPS: {}.'.format(fps))
    logger.info('Number of frame: {}.'.format(num_frame))

def cut_crop_video(video_path, output_path, start_time, end_time, crop_dimensions):
    """
    Cut and crop video segment based on temporal boundaries and spatial parameters.
    
    Args:
        video_path: Path to input video
        output_path: Path to output video
        start_time: Start time in seconds
        end_time: End time in seconds
        crop_dimensions: String format "w:h:x:y" where:
            - w, h: target width and height (default: 1080x1080)
            - x, y: cropping origin coordinates (default: x=420, y=0)
    """
    w, h, x, y = map(int, crop_dimensions.split(':'))
    cap = cv2.VideoCapture(str(video_path))
    
    # Get frame rate of video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate start and end frame positions
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Set starting position for reading video
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Create VideoWriter object to write video
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    # Read and write frames from start_frame to end_frame
    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        # Crop frame according to specified position
        cropped_frame = frame[y:y+h, x:x+w]

        out.write(cropped_frame)
        current_frame += 1
    
    # Release resources
    cap.release()
    out.release()
    logger.info("Video segment has been cut and cropped successfully.")

def process_cutting_cropping_video(input_video, cut_time_file, output_dir, process_all, from_index, to_index, crop_dimensions, overwrite):
    """
    Boundary-Guided Segmentation & Pruning (BGSP) - Performs temporal segmentation,
    spatial cropping, and resolution normalization on video segments.
    
    Takes candidate boundaries from TBL and produces final segmented videos with:
    - Temporal segmentation based on cut times
    - Spatial cropping to remove redundant background
    - Resolution normalization (1920x1080 -> 1080x1080)
    """
    data = pd.read_csv(cut_time_file)
    logger.info("Loaded cut time from {}.".format(cut_time_file))

    # Choose the process range
    if process_all:
        process_range = range(len(data))
        logger.info("Will process all data, {} samples.".format(len(data)))
    else:
        if from_index == None:
            from_index = 0
        if to_index == None:
            to_index = len(data)
        process_range = range(from_index, to_index)
        logger.info("Will process data from index {} to index {}.".format(from_index, to_index))

    for i in process_range:
        output_path = Path(output_dir, "{}.mp4".format(i))
        if output_path.exists() and not overwrite:
            logger.info('{} | Video already exists at {}.'.format(i, output_path))
        else:
            logger.info('{} | start time: {}, end time: {}.'.format(i, data['start_time'][i], data['end_time'][i]))
            cut_crop_video(input_video, str(output_path), data['start_time'][i], data['end_time'][i], crop_dimensions)
            logger.info('Saved video at {}.'.format(output_path))

def main():
    # Get arguments
    args = ProcessRecordedVideosArguments()
    args = args.parse()

    # Config logger
    config_logger(args.log_file)
    
    logger.info("------------------------- RUNNING BGSP PROCESS -------------------------")
    
    input_video = Path(args.input_video)
      
    # Check input video
    if not input_video.exists():
        logger.error("Not found {}.".format(input_video))
        return
    else:
        logger.info('Processing video at {}.'.format(input_video))
    
    log_video_info(input_video)
    
    # Check cut time file
    if args.cut_time_file is None:
        cut_time_file = input_video.with_name(input_video.stem + '_cut_time.csv')
    else:
        cut_time_file = Path(args.cut_time_file)
        
    if args.cut_crop_video:
        logger.info('--CUTTING AND CROPPING VIDEO (BGSP)')
        # Check input data
        if not cut_time_file.exists():
            logger.error("Not found {}.".format(cut_time_file))
        else:
            # Check output dir
            if args.output_dir is None:
                output_dir = input_video.parent / input_video.stem
            else:
                output_dir = Path(args.output_dir)
            if not output_dir.exists():
                output_dir.mkdir(parents=True)
            logger.info("Save processed videos at {}.".format(output_dir))
            process_cutting_cropping_video(input_video, cut_time_file, output_dir, args.process_all, args.from_index, args.to_index, args.crop_dimensions, args.overwrite)
        
if __name__ == "__main__":
    main()