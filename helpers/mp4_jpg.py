import cv2
import os
import zipfile
from datetime import datetime

def create_zip(folder_path, zip_name):
    """
    Create a ZIP file from a folder
    
    Args:
        folder_path: Path to folder to zip
        zip_name: Name of the output ZIP file
    """
    print(f"\nCreating ZIP file: {zip_name}")
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
                
    file_size = os.path.getsize(zip_name) / (1024 * 1024)  # Size in MB
    print(f"ZIP file created successfully: {zip_name}")
    print(f"File size: {file_size:.2f} MB")


def extract_frames(video_path, output_folder, frames_per_second=1, create_zip_file=True):
    """
    Extract frames from a video at specified rate and save as JPEG images
    
    Args:
        video_path: Path to the input MP4 video file
        output_folder: Path to folder where frames will be saved
        frames_per_second: Number of frames to extract per second (default: 1)
        create_zip_file: Whether to create a ZIP file after extraction (default: True)
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Extracting {frames_per_second} frame(s) per second")
    print(f"Output folder: {output_folder}")
    
    # Calculate frame interval
    frame_interval = int(fps / frames_per_second)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        # Read next frame
        success, frame = video.read()
        
        if not success:
            break
        
        # Save frame only at specified intervals
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
            print(f"Saved frame at {frame_count/fps:.2f}s -> {frame_filename}")
        
        frame_count += 1
    
    # Release video object
    video.release()
    
    print(f"\nCompleted! Extracted {saved_count} frames from {duration:.2f} seconds of video.")
    
    return None

# Example usage
if __name__ == "__main__":
    # Replace with your video path
    video_path = "Videos/Banana_timelapse15.mp4"
    
    # Replace with your desired output folder
    output_folder = "extracted_frames/15"
    
    extract_frames(video_path, output_folder)