import cv2
import os

video_file_path = '002/world.mp4'  # Example: 'video.mp4'
output_directory = 'frames/002'

# Create output directory if it does not exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Create a VideoCapture object
cap = cv2.VideoCapture(video_file_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if we are at the end of the video

    # Construct the output image file path
    output_file_path = os.path.join(output_directory, f'frame_{frame_count}.png')

    # Save the frame as an image file
    cv2.imwrite(output_file_path, frame)

    frame_count += 1

# Release the VideoCapture object
cap.release()

print(f"Finished! {frame_count} frames have been saved to {output_directory}.")
