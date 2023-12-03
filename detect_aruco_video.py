import numpy as np
import cv2
from visualize_2d_new import  Visualizer
from utils import get_translated_view_rectangle
import pandas as pd
import argparse
from plot_data import initialize_plot, update_plot
from config import *
from utils import *


def get_gaze_positions(file_path):
    # Load the DataFrame (assuming you've already done this)
    # df = pd.read_csv("000/exports/000/gaze_positions.csv")
    df = pd.read_csv(file_path)

    # Group by 'world_index' and get the row with the highest 'confidence' for each group
    df_max_confidence = df.loc[df.groupby('world_index')['confidence'].idxmax()]

    df_max_confidence = df_max_confidence[['norm_pos_x', 'norm_pos_y']]

    return df_max_confidence.values.tolist()


def get_head_poses(head_poses, timestamps):
    # Load the DataFrame (assuming you've already done this)
    # df = pd.read_csv("000/exports/000/head_pose_tracker_poses.csv")
    head_pose_data = pd.read_csv(head_poses)
    frame_timestamps = pd.read_csv(timestamps)["# timestamps [seconds]"].tolist()
    # Group by 'world_index' and get the row with the highest 'confidence' for each group
    # Find nearest head pose for each frame
    nearest_head_poses = []
    for frame_ts in frame_timestamps:
        nearest_index = (head_pose_data['timestamp'] - frame_ts).abs().idxmin()
        nearest_head_poses.append(head_pose_data.iloc[nearest_index])

    nearest_head_poses_list_of_lists = [[pose['pitch'], pose['yaw'], pose['roll']] for pose in nearest_head_poses]

    return nearest_head_poses_list_of_lists


def get_marker_positions(file_path):
    # df = pd.read_csv("000/exports/000/surfaces/marker_detections.csv")
    df = pd.read_csv(file_path)
    max_world_index_value = df['world_index'].max()
    markers_per_frame = []

    for i in range(max_world_index_value + 1):  # Including the last index value
        df_grouped_by_world_index = df[df["world_index"] == i]
        markers_in_frame = {}

        # Iterate over the DataFrame rows
        for index, line in df_grouped_by_world_index.iterrows():
            marker_uid = line["marker_uid"][21:]
            markers_in_frame[marker_uid] = [
                [line["corner_0_x"], line["corner_0_y"]],
                [line["corner_1_x"], line["corner_1_y"]],
                [line["corner_2_x"], line["corner_2_y"]],
                [line["corner_3_x"], line["corner_3_y"]]
            ]

        markers_per_frame.append(markers_in_frame)

    return markers_per_frame


def get_eye_diameters(file_path):
    # df = pd.read_csv("000/exports/000/pupil_positions.csv")
    df = pd.read_csv(file_path)

    df_max_confidence = df.loc[df.groupby(['world_index', 'eye_id'])['confidence'].idxmax()]

    df_max_confidence = df_max_confidence[['world_index', 'eye_id', 'diameter']]
    pivot_df = df_max_confidence.pivot(index='world_index', columns='eye_id', values='diameter')

    return pivot_df.values.tolist()


def get_blinks(file_path):
    df = pd.read_csv(file_path)
    df = df['start_frame_index']
    return df.values.tolist()


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
    print(number_of_frames, len(head_poses), len(gaze_positions), len(marker_positions), len(eye_diameters))
    blink_list = [0]
    last_index = -1
    frame_index = 0
    visualizer = Visualizer(number_of_frames)
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
        blink_rate = estimate_blink_rate(blink_list, blinks, fps, frame_index, last_index)


        visualizer.update(fov_rectangle, fov_center, markers, head_pose, adjusted_gaze_pos, blink_rate, diameter_mean,
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
    if blinks[0] == frame_index:
        temp_value = add_blink_to_list(blink_list, blinks.pop(0))
        if temp_value != None:
            last_index = temp_value
    blink_rate = (fps * len(blink_list)) / (frame_index - last_index)
    return blink_rate


def get_iters_from_lists(eye_diameters, gaze_positions, head_poses, marker_positions):
    gaze_positions_iter = iter(gaze_positions)
    head_poses_iter = iter(head_poses)
    marker_positions_iter = iter(marker_positions)
    eye_diameter_iter = iter(eye_diameters)
    return eye_diameter_iter, gaze_positions_iter, head_poses_iter, marker_positions_iter


def get_data_from_csvs(blinks_path, gaze_positions_path, headpose_tracker_path, marker_detections_path,
                       pupil_positions_path, world_timestamps_path):
    blinks = get_blinks(blinks_path)
    gaze_positions = get_gaze_positions(gaze_positions_path)
    head_poses = get_head_poses(headpose_tracker_path,world_timestamps_path)
    marker_positions = get_marker_positions(marker_detections_path)
    eye_diameters = get_eye_diameters(pupil_positions_path)
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
