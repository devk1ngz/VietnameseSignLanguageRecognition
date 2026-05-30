import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp

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

def normalize_video(input_file, output_file, resolution='1920:1080', fps=30):
    # Open the video file
    cap = cv2.VideoCapture(str(input_file))

    # Get the video's original width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Get the desired width and height
    desired_width, desired_height = map(int, resolution.split(':'))

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_file), fourcc, fps, (desired_width, desired_height))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Resize the frame
        frame = cv2.resize(frame, (desired_width, desired_height))

        # Write the frame into the file 'output_file'
        out.write(frame)

    # Release everything after the job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
def process_normalizing_quality(input_video, normalized_video, standard_resolution, standard_fps):
    # Load video
    cap = cv2.VideoCapture(str(input_video))
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4)) 
    resolution = '{}:{}'.format(frame_width, frame_height) 
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    
    if frame_width != int(standard_resolution.split(':')[0]) or frame_height!=int(standard_resolution.split(':')[1]) or standard_fps != fps:
        logger.info('Change the video resolution: {} -> {}.'.format(resolution, standard_resolution))
        logger.info('Change the video fps: {} -> {}.'.format(fps, standard_fps))
        normalize_video(input_video, normalized_video, standard_resolution, standard_fps)
        logger.info('Normalized video, saved at {}.'.format(normalized_video))
        normalized = True
    else:
        logger.info('The video is already normalized.')
        normalized = False
    return normalized
        
def calculate_angle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

def get_start_end_time(left_start_time, left_end_time, left_status, right_start_time, right_end_time, right_status):
    start_time = 0
    end_time = 0 
    
    if left_start_time != 0 and left_end_time != 0 and right_start_time == 0:
        start_time = left_start_time
        end_time = left_end_time
    elif right_start_time != 0 and right_end_time != 0 and left_start_time == 0:
        start_time = right_start_time
        end_time = right_end_time
    elif (left_start_time != 0 and left_end_time != 0 and left_status == 'down') and (right_start_time != 0 and right_end_time != 0 and right_status == 'down'):
        start_time = min(left_start_time, right_start_time) 
        end_time = max(left_end_time, right_end_time) 
        
    return start_time, end_time

def save_to_csv(output_file, data):
    df = pd.DataFrame(data, columns =['start_time', 'end_time'])
    df.to_csv(output_file, index=True)
    logger.info('Saved cut time file at {}'.format(output_file))

def process_getting_cut_time(input_video, cut_time_file, process_all, from_second, to_second, threshold, delay, min_up_frame, min_down_frame, visualize):
    """
    Temporal Boundary Localization (TBL) - Detects candidate temporal boundaries
    for sign language gestures based on arm angles and hand visibility.
    """
    # Load video
    cap = cv2.VideoCapture(str(input_video))
    
    # Init Mediapipe
    mp_pose= mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Init variables
    left_status = 'down'
    left_up_frame = 0
    left_down_frame = 0
    left_start_time_temp = 0
    left_end_time_temp = 0
    left_start_time = 0
    left_end_time = 0
    
    right_status = 'down'
    right_up_frame = 0
    right_down_frame = 0
    right_start_time_temp = 0
    right_end_time_temp = 0
    right_start_time = 0
    right_end_time = 0

    cut_time = []
        
    visibility_threshold = 0.6

    # Choose the process range
    if process_all:
        from_frame = 0
        to_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        logger.info("Will process all video, {} frames.".format(to_frame ))
    else:
        if from_second == None:
            from_frame = 0
        else:
            from_frame = round(from_second * cap.get(cv2.CAP_PROP_FPS)) 
            
        if to_second == None:
            to_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        else:
            to_frame = round(to_second * cap.get(cv2.CAP_PROP_FPS))
            
        logger.info("Will process data from frame {} to frame {}.".format(from_frame, to_frame))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, from_frame)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_landmarks=True) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Recolor image to RGB, because mp processes on RGB image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detections
            results = pose.process(image)
            
            # Recolor image back to BGR, because cv2 processes on BGR image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            landmarks = None
            try:
                landmarks = results.pose_landmarks.landmark
            except:
                continue 
            
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,   landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility]
            
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility]
            
            # Calculate angles
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Check if left hand up or down
            if left_angle < threshold and left_wrist[2]>visibility_threshold and left_status == 'down':
                if left_up_frame == 0:
                    left_start_time_temp = cap.get(cv2.CAP_PROP_POS_MSEC) - delay
                    left_up_frame += 1       
                elif left_up_frame == min_up_frame:
                    left_status = 'up'
                    left_start_time = left_start_time_temp
                    left_up_frame = 0
                    left_start_time_temp = 0
                else:
                    left_up_frame += 1 
            
            if ((left_angle > threshold and left_wrist[2]>visibility_threshold) or left_wrist[2]<visibility_threshold) and left_status == 'down':
                left_up_frame = 0
                left_start_time_temp = 0
            
            if ((left_angle > threshold and left_wrist[2]>visibility_threshold) or left_wrist[2]<visibility_threshold) and left_status == 'up':
                if left_down_frame == 0:
                    left_end_time_temp = cap.get(cv2.CAP_PROP_POS_MSEC) + delay
                    left_down_frame += 1       
                elif left_down_frame == min_down_frame:
                    left_status = 'down'
                    left_end_time = left_end_time_temp
                    left_down_frame = 0
                    left_end_time_temp = 0
                else:
                    left_down_frame += 1 
            
            if left_angle < threshold and left_wrist[2]>visibility_threshold and left_status == 'up':
                left_down_frame = 0
                left_end_time_temp = 0   
            
            # Check if right hand up or down
            if right_angle < threshold and right_wrist[2]>visibility_threshold and right_status == 'down':
                if right_up_frame == 0:
                    right_start_time_temp = cap.get(cv2.CAP_PROP_POS_MSEC) - delay
                    right_up_frame += 1       
                elif right_up_frame == min_up_frame:
                    right_status = 'up'
                    right_start_time = right_start_time_temp
                    right_up_frame = 0
                    right_start_time_temp = 0
                else:
                    right_up_frame += 1 
            
            if ((right_angle > threshold and right_wrist[2]>visibility_threshold) or right_wrist[2]<visibility_threshold) and right_status == 'down':
                right_up_frame = 0
                right_start_time_temp = 0
            
            if ((right_angle > threshold and right_wrist[2]>visibility_threshold) or right_wrist[2]<visibility_threshold) and right_status == 'up':
                if right_down_frame == 0:
                    right_end_time_temp = cap.get(cv2.CAP_PROP_POS_MSEC) + delay
                    right_down_frame += 1       
                elif right_down_frame == min_down_frame:
                    right_status = 'down'
                    right_end_time = right_end_time_temp
                    right_down_frame = 0
                    right_end_time_temp = 0
                else:
                    right_down_frame += 1 
            
            if right_angle < threshold and right_wrist[2]>visibility_threshold and right_status == 'up':
                right_down_frame = 0
                right_end_time_temp = 0
                
            # Calculate the start and end time of sign
            start_time, end_time = get_start_end_time(left_start_time, left_end_time, left_status, right_start_time, right_end_time, right_status)
            if start_time !=0 and end_time != 0:
                # Convert seconds to milliseconds
                start_time /= 1000
                end_time /= 1000
                
                logger.info('{} | frame: {}/{} | start time: {} - end time: {}.'.format(len(cut_time), cap.get(cv2.CAP_PROP_POS_FRAMES), cap.get(cv2.CAP_PROP_FRAME_COUNT), start_time, end_time))
                cut_time.append([start_time, end_time])
                
                # Reset variables
                start_time = 0
                end_time = 0
                left_start_time = 0
                right_start_time = 0
                left_end_time = 0
                right_end_time = 0
            
            # Show image
            if visualize:
                # Render angles
                cv2.putText(image, str(left_angle), (round(cap.get(3) / 2) + 300, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                cv2.putText(image, str(right_angle), (300, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
                cv2.putText(image, str(round(left_wrist[2], 2)),(300, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                cv2.putText(image, str(round(right_wrist[2], 2)),(500, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                image = cv2.resize(image, (540, 540))
                cv2.imshow("Video Visualization", image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == to_frame:
                break
        cap.release()
        cv2.destroyAllWindows()
            
        save_to_csv(cut_time_file, cut_time)

def process_visualization(input_video, process_all, from_second, to_second):
    # Load video
    cap = cv2.VideoCapture(str(input_video))
    
    # Init Mediapipe
    mp_pose= mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Choose the process range
    if process_all:
        from_frame = 0
        to_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        logger.info("Will process all video, {} frames.".format(to_frame ))
    else:
        if from_second == None:
            from_frame = 0
        else:
            from_frame = round(from_second * cap.get(cv2.CAP_PROP_FPS)) 
            
        if to_second == None:
            to_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        else:
            to_frame = round(to_second * cap.get(cv2.CAP_PROP_FPS))
            
        logger.info("Will process data from frame {} to frame {}.".format(from_frame, to_frame))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, from_frame)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_landmarks=True) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Recolor image to RGB, because mp processes on RGB image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detections
            results = pose.process(image)
            
            # Recolor image back to BGR, because cv2 processes on BGR image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            landmarks = None
            try:
                landmarks = results.pose_landmarks.landmark
            except:
                continue 
            
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,   landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility]
            
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility]
            # Calculate angles
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Render angles
            cv2.putText(image, str(left_angle), (round(cap.get(3) / 2) + 300, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.putText(image, str(right_angle), (300, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            cv2.putText(image, str(round(left_wrist[2], 2)),(300, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.putText(image, str(round(right_wrist[2], 2)),(500, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            image = cv2.resize(image, (540, 540))
            cv2.imshow("Video Visualization", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == to_frame:
                break
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Get arguments
    args = ProcessRecordedVideosArguments()
    args = args.parse()

    # Config logger
    config_logger(args.log_file)
    
    logger.info("------------------------- RUNNING TBL PROCESS -------------------------")
    
    input_video = Path(args.input_video)
      
    # Check input video
    if not input_video.exists():
        logger.error("Not found {}.".format(input_video))
        return
    else:
        logger.info('Processing video at {}.'.format(input_video))
    
    log_video_info(input_video)
    
    if args.normalize_quality:
        logger.info('--NORMALIZING QUALITY')
        # Check output path
        if args.normalized_video is None:
            normalized_video = input_video.with_name(input_video.stem + '_normalized.mp4')
        else:
            normalized_video = Path(args.normalized_video)
            
        if normalized_video.exists() and not args.overwrite:
            logger.error('Normalized video already exists.')
        else:
            normalized = process_normalizing_quality(input_video, normalized_video, args.resolution, args.fps)
            # Change input source
            if normalized:
                input_video = normalized_video
                log_video_info(input_video)
                
    # Check cut time file
    if args.cut_time_file is None:
        cut_time_file = input_video.with_name(input_video.stem + '_cut_time.csv')
    else:
        cut_time_file = Path(args.cut_time_file)
    
    if args.get_cut_time:
        logger.info('--GETTING CUT TIME FILE (TBL)')
        # Check output path
        if cut_time_file.exists() and not args.overwrite:
            logger.error('Cut time file already exists.')
        else:
            if not cut_time_file.parent.exists():
                cut_time_file.parent.mkdir(parents=True)
            process_getting_cut_time(input_video, cut_time_file, args.process_all, args.from_second, args.to_second, args.threshold, args.delay, args.min_up_frame, args.min_down_frame, args.visualize)
            
    if not args.get_cut_time and args.visualize:
        logger.info('--VISUALIZATION')
        process_visualization(input_video, args.process_all, args.from_second, args.to_second)
        
if __name__ == "__main__":
    main()