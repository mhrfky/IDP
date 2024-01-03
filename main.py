import numpy as np
import cv2
from file_utils import get_blinks, get_eye_diameters, get_gaze_positions, get_head_poses, get_marker_positions
from opencv_visualization import  Visualizer
from utils import get_translated_view_rectangle
import argparse
from config import *
from utils import *


def add_blink_to_list(blinks: list, frame_stamp):
    blinks.append(frame_stamp)
    if len(blinks) == BLINK_LIST_LENGTH:
        return blinks.pop(0)



def main():
    video_path, headpose_tracker_path, marker_detections_path, pupil_positions_path, gaze_positions_path, blinks_path, world_timestamps_path= init_args()
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps = video.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)  # Real-time delay
    number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    blinks, eye_diameters, gaze_positions, head_poses, marker_positions = get_data_from_csvs(blinks_path,
                                                                                             gaze_positions_path,
                                                                                             headpose_tracker_path,
                                                                                             marker_detections_path,
                                                                                             pupil_positions_path,
                                                                                             world_timestamps_path)
    eye_diameter_iter, gaze_positions_iter, head_poses_iter, marker_positions_iter = get_iters_from_lists(eye_diameters,
                                                                                                          gaze_positions,
                                                                                                          head_poses,
                                                                                                          marker_positions)
    blink_list = [[0,0]]
    last_index = -1
    frame_index = 0
    visualizer = Visualizer(number_of_frames, (frame_width,frame_height), plot_heatmap=True)
    while True:
        ret, frame = video.read()

        gaze_position = next(gaze_positions_iter)
        pitch, yaw, roll = next(head_poses_iter)
        markers = next(marker_positions_iter)
        eye_diameter = next(eye_diameter_iter)

        fov_center, fov_rectangle = get_translated_view_rectangle(pitch, roll, yaw)
        adjusted_gaze_pos = estimate_gaze_pos(fov_center, gaze_position, roll)
        head_pose = {"yaw": yaw, "roll": roll, "pitch": pitch}
        diameter_mean = (eye_diameter[0] + eye_diameter[1]) / 2
        blink_rate, blink, last_index = estimate_blink_rate(blink_list, blinks, fps, frame_index, last_index)
        adjusted_normalized_markers = get_adjusted_normalized_markers(fov_center,markers,roll)


        visualizer.update(fov_rectangle, fov_center, adjusted_normalized_markers, head_pose, adjusted_gaze_pos, blink_rate, blink, diameter_mean,
                          frame_index)
        cv2.imshow("Image", frame)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("w"):
            key = cv2.waitKey()
            if key == ord("q"):
                break
        frame_index += 1
    cv2.destroyAllWindows()
    video.release()


def estimate_blink_rate(blink_list, blinks, fps, frame_index, last_index):
    blink = 0
    if blinks[0][0] == frame_index:
        blink = 1
        temp_value = add_blink_to_list(blink_list, blinks.pop(0))
        if temp_value != None:
            last_index = temp_value[0]
    blink_rate = (fps * len(blink_list)) / (frame_index - last_index)
    return blink_rate, blink, last_index


def get_iters_from_lists(eye_diameters, gaze_positions, head_poses, marker_positions):
    gaze_positions_iter = iter(gaze_positions)
    head_poses_iter = iter(head_poses)
    marker_positions_iter = iter(marker_positions)
    eye_diameter_iter = iter(eye_diameters)
    return eye_diameter_iter, gaze_positions_iter, head_poses_iter, marker_positions_iter


def get_data_from_csvs(blinks_path, gaze_positions_path, headpose_tracker_path, marker_detections_path,
                       pupil_positions_path, world_timestamps_path):
    blinks = get_blinks(blinks_path).values.tolist()
    gaze_positions = get_gaze_positions(gaze_positions_path).values.tolist()
    head_poses = get_head_poses(headpose_tracker_path,world_timestamps_path).values.tolist()
    marker_positions = get_marker_positions(marker_detections_path)
    eye_diameters = get_eye_diameters(pupil_positions_path).values.tolist()
    return blinks, eye_diameters, gaze_positions, head_poses, marker_positions


def init_args():
    parser = argparse.ArgumentParser(description="Process some integers.")

    # Define arguments
    parser.add_argument("-b", "--blinks", type=str, help="Path to the blink")
    parser.add_argument("-t", "--headpose_tracker", type=str, help="Path to head_pose_tracker_poses.csv CSV file")
    parser.add_argument("-m", "--marker_detections", type=str, help="Path to marker_detections.csv CSV file")
    parser.add_argument("-p", "--pupil_positions", type=str, help="Path to pupil_positions.csv CSV file")
    parser.add_argument("-g", "--gaze_positions", type=str, help="Path to gaze_positions.csv CSV file")
    parser.add_argument("-v", "--video", type=str, help="Path to the video of the world camera")
    parser.add_argument("-w", "--world_timestamps", type=str, help="Path to the video of the world_timestamps.csv file")

    # Parse arguments
    args = parser.parse_args()
    blinks_path = args.blinks
    video_path = args.video
    headpose_tracker_path = args.headpose_tracker
    marker_detections_path = args.marker_detections
    pupil_positions_path = args.pupil_positions
    gaze_positions_path = args.gaze_positions
    world_timestamps_path = args.world_timestamps

    return video_path, headpose_tracker_path, marker_detections_path, pupil_positions_path, gaze_positions_path, blinks_path, world_timestamps_path


if __name__ == "__main__":
    main()
