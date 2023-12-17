import pandas as pd

from config import cv2


def get_gaze_positions(file_path):
    # Load the DataFrame (assuming you've already done this)
    # df = pd.read_csv("000/exports/000/gaze_positions.csv")
    df = pd.read_csv(file_path)

    # Group by 'world_index' and get the row with the highest 'confidence' for each group
    df_max_confidence = df.loc[df.groupby('world_index')['confidence'].idxmax()]

    df_max_confidence = df_max_confidence[['norm_pos_x', 'norm_pos_y']]

    return df_max_confidence


def get_head_poses(head_poses, timestamps):
    # Load the DataFrame (assuming you've already done this)
    # df = pd.read_csv("000/exports/000/head_pose_tracker_poses.csv")
    head_pose_data = pd.read_csv(head_poses)
    frame_timestamps = pd.read_csv(timestamps)["# timestamps [seconds]"]
    # Group by 'world_index' and get the row with the highest 'confidence' for each group
    # Find nearest head pose for each frame
    nearest_head_poses = []
    for frame_ts in frame_timestamps:
        nearest_index = (head_pose_data['timestamp'] - frame_ts).abs().idxmin()
        nearest_head_poses.append(head_pose_data.iloc[nearest_index])

    nearest_head_poses_list_of_lists = [[pose['pitch'], pose['yaw'], pose['roll']] for pose in nearest_head_poses]
    df_head_poses = pd.DataFrame(nearest_head_poses_list_of_lists, columns= ['pitch', 'yaw', 'roll'])
    return df_head_poses


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
    
    pivot_df[0].interpolate(method='linear', inplace=True)
    pivot_df[1].interpolate(method='linear', inplace=True)
    return pivot_df


def get_blinks(file_path):
    df = pd.read_csv(file_path)
    df = df[['start_frame_index','end_frame_index']]
    return df


def get_frames(video_path):

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frames = []
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Append the frame to the list
            frames.append(frame)
        else:
            break


    # When everything done, release the video capture object
    cap.release()
    return frames, number_of_frames, frame_width, frame_height, fps