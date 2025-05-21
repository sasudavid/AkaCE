import os
import math
import cv2
import pandas as pd
import subprocess
from tqdm import tqdm
import numpy as np
import torch
from facenet_pytorch import MTCNN

# Read the CSV file.
data_csv = pd.read_csv('../Akan Speech Emotion Dataset cleaned.csv')

videos_directory = '../movie_segments/'
output_directory = '../extracted_frames/'
max_faces = 30  # Maximum number of face images to extract per video segment
fps = 5         # Frames per second to sample
interval = 1.0 / fps

# Set up the device and MTCNN face detector.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)  # Advanced face detector

def get_video_duration(filename):
    """
    Returns the duration of the video in seconds using ffprobe.
    """
    try:
        command = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            filename
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        print(f"An error occurred while getting duration: {e}")
        return None

for index, row in tqdm(data_csv.iterrows(), total=len(data_csv), desc="Processing videos"):
    # Clean up start and end time strings
    start_time_str = row['Start Time'].replace(':', '')
    end_time_str = row['End Time'].replace(':', '')
    if start_time_str == end_time_str:
        print(f"Skipping {row['Movie Title']}: Start and end times are the same.")
        continue

    # Construct the video filename.
    video_name = f"{row['Movie Title']}_{start_time_str}_{end_time_str}.mp4"
    video_directory = os.path.join(videos_directory, row['Movie Title'])
    video_path = os.path.join(video_directory, video_name.capitalize())
    
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        continue

    # Create output directory for faces for this video segment
    faces_output_dir = os.path.join(output_directory, row['Movie Title'], video_name.split('.')[0] + '_faces')
    os.makedirs(faces_output_dir, exist_ok=True)
    
    # Get video duration using ffprobe
    duration = get_video_duration(video_path)
    if duration is None:
        print(f"Skipping {video_path} due to duration extraction error.")
        continue

    # Generate timestamps at 5fps: 0, 0.2, 0.4, ... up to the video duration.
    timestamps = np.arange(0, duration, interval)
    
    face_count = 0  # Counter for total face images extracted for this video

    for t in timestamps:
        if face_count >= max_faces:
            print(f"Reached maximum of {max_faces} faces for video: {video_path}")
            break

        # Create a temporary filename for the extracted frame.
        frame_filename = os.path.join(faces_output_dir, f"frame_{t:.2f}.jpg")
        if os.path.exists(frame_filename):
            frame = cv2.imread(frame_filename)
        else:
            # Extract frame using ffmpeg
            command = [
                'ffmpeg',
                '-ss', str(t),
                '-i', video_path,
                '-frames:v', '1',
                '-q:v', '2',
                frame_filename,
                '-y'
            ]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print(f"Warning: ffmpeg could not extract frame at {t:.2f}s for {video_path}")
                print(result.stderr)
                continue
            frame = cv2.imread(frame_filename)
            if frame is None:
                print(f"Warning: Could not read frame from {frame_filename}")
                continue

        # Use MTCNN to detect faces. MTCNN expects a PIL Image.
        # Convert the frame (BGR) to RGB and then to a PIL Image.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # MTCNN's detect method returns bounding boxes and probabilities.
        boxes, probs = mtcnn.detect(frame_rgb)
        
        # Check if faces were detected with reasonable confidence.
        if boxes is not None:
            for box in boxes:
                if face_count >= max_faces:
                    break
                # The box is in the format [x1, y1, x2, y2]
                x1, y1, x2, y2 = [int(coord) for coord in box]
                # Ensure the box is within frame boundaries.
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                face_img = frame[y1:y2, x1:x2]
                # Construct the output filename for the face.
                face_filename = os.path.join(faces_output_dir, f"face_{face_count:02d}.jpg")
                cv2.imwrite(face_filename, face_img)
                face_count += 1
        
    print(f"Extracted {face_count} face images from {video_path}")
