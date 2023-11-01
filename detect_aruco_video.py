'''
Sample Command:-
python detect_aruco_video.py --type DICT_5X5_100 --camera True
python detect_aruco_video.py --type DICT_5X5_100 --camera False --video test_video.mp4
'''

import numpy as np
from utils import ARUCO_DICT, aruco_display
import argparse
import time
import cv2
import sys
from visualize_in_2d import visualize_fov
from visualize_2d_new import visualize_head_pose_from_matrix

from scipy.spatial.transform import Rotation as R

import pandas as pd


def load_aruco_dictionary_from_yaml(filename):
    fs = cv2.FileStorage(filename, cv2.FileStorage_READ)
    if not fs.isOpened():
        print(f"Failed to open {filename}")
        return None

    aruco_dict = cv2.aruco.Dictionary()
    fn = fs.root()
    aruco_dict.readDictionary(fn)

    return aruco_dict


def setup_detector():
    global detector
    # load the ArUCo dictionary, grab the ArUCo parameters, and detect
    # the markers
    filename = "my_custom_dictionary.yml"
    arucoDict = load_aruco_dictionary_from_yaml(filename)
    arucoParams = cv2.aruco.DetectorParameters()
    arucoParams.adaptiveThreshConstant = 10
    arucoParams.adaptiveThreshWinSizeMax = 30
    arucoParams.adaptiveThreshWinSizeMin = 3
    arucoParams.adaptiveThreshWinSizeStep = 25

    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)


def draw_markers(image, corners):
    cv2.aruco.drawDetectedMarkers(image, corners)


def get_poses(ids, corners):
    d = []
    if ids is None:
        return
    for id, corner in zip(ids, corners):
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corner, 0.1341, camera_matrix, dist_matrix)
        rot_mtx, _ = cv2.Rodrigues(rvec)
        cameraPose = np.eye(4)
        cameraPose[:3, :3] = rot_mtx
        cameraPose[:3, 3] = tvec.squeeze()
        d.append((id, cameraPose))

    return {i[0][0]: i[1] for i in sorted(d, key=lambda x: x[0])}


def get_intrinsics():
    global camera_matrix, dist_matrix
    camera_matrix = np.array(
        [[794.3311439869655, 0.0, 633.0104437728625], [0.0, 793.5290139393004, 397.36927353414865], [0.0, 0.0, 1.0]])
    dist_matrix = np.array([[-0.3758628065070806, 0.1643326166951343, 0.00012182540692089567, 0.00013422608638039466,
                             0.03343691733865076, 0.08235235770849726, -0.08225804883227375, 0.14463365333602152]])


def display(image):
    # Get the dimensions of the image
    # height, width = image.shape[:2]
    #
    # # Define the scale factor
    # scale_factor = 2.0  # Example: 2.0 will double the display size of the image
    #
    # # Resize the image
    # image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))

    cv2.imshow("Image", image)


def decompose_homography(K, H):
    # Normalize the homography
    H /= H[2, 2]

    # Compute the rotation matrix
    K_inv = np.linalg.inv(K)
    R_ = np.dot(K_inv, H)

    # Orthonormalize the rotation matrix
    U, _, Vt = np.linalg.svd(R_[:, :2])

    R = np.zeros((3, 3))
    W = np.array([[1, 0], [0, 1], [0, 0]])  # We'll create a transformation matrix to fix the sizes
    R[:, :2] = np.dot(U, np.dot(W, Vt))
    R[:, 2] = np.cross(R[:, 0], R[:, 1])

    return R


def get_transformation_matrix(curr_poses):
    src_pts = []
    dst_pts = []

    for id, curr_pose in curr_poses.items():
        if id not in init_poses:
            continue

        src_pts.extend(init_poses[id])
        dst_pts.extend(curr_poses[id])
    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)

    H, _ = cv2.findHomography(src_pts, dst_pts)

    # Extract rotation matrix from the homography
    R = decompose_homography(camera_matrix, H)

    T = np.eye(4)
    T[:3, :3] = R

    return T


def kalman_filter_setup():
    # Initialize Kalman Filter
    kf = cv2.KalmanFilter(6, 6)  # State size: 6, Measurement size: 6

    kf.transitionMatrix = np.eye(6, dtype=np.float32)  # For example, Identity Matrix for a simple model
    kf.measurementMatrix = np.eye(6, dtype=np.float32)  # Adjust accordingly
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1  # Adjust the noise level accordingly
    kf.measurementNoiseCov = np.eye(6, dtype=np.float32) * 0.05  # Adjust the noise level accordingly
    kf.errorCovPost = np.eye(6, dtype=np.float32)  # Posteriori error estimate covariance matrix

    # Initialize statePost, if needed
    kf.statePost = np.zeros((6, 1), dtype=np.float32)  # Adjust initial state as needed

    return kf


def get_gaze_positions():

    # Load the DataFrame (assuming you've already done this)
    df = pd.read_csv("000/exports/000/gaze_positions.csv")

    # Group by 'world_index' and get the row with the highest 'confidence' for each group
    df_max_confidence = df.loc[df.groupby('world_index')['confidence'].idxmax()]

    df_max_confidence = df_max_confidence[['world_index', 'norm_pos_x', 'norm_pos_y']]

    return  df_max_confidence.values.tolist()


video = "000/world.mp4"
video = cv2.VideoCapture(video)

# Set the starting frame to 500
video.set(cv2.CAP_PROP_POS_FRAMES, 0)
get_intrinsics()
setup_detector()

init_poses = None
kf = kalman_filter_setup()

fps = video.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)  # Real-time delay
number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

weights = [0.5, 0.3, 0.2]  # Example weights for the last three transformations
measurements = []  # List to hold the last n measurements
n = 3  # Number of matrices to consider for moving average

# Initialize previous state for Kalman filter
previous_state = None

gaze_positions = get_gaze_positions()
gaze_positions_iter = iter(gaze_positions)

while True:
    ret, frame = video.read()
    gaze_position = next(gaze_positions_iter)
    shape = frame.shape
    if not ret:
        break

    corners, ids, rejected = detector.detectMarkers(frame)
    if ids is None:
        continue  # Skip the rest of the loop if no markers are detected
    ids = [id[0] for id in ids]
    cornerss = [marker[0] for marker in corners]

    if init_poses is None:
        init_poses = dict(zip(ids, cornerss))
    curr_poses = dict(zip(ids, cornerss))

    draw_markers(frame, corners)
    display(frame)

    avgCameraPose = get_transformation_matrix(curr_poses)

    if avgCameraPose is None:  # Handle the case if you don't get a transformation matrix
        continue

    # Convert transformation matrix to translation and Euler angles
    tvec = avgCameraPose[:3, 3]
    rotation = R.from_matrix(avgCameraPose[:3, :3])
    roll, pitch, yaw = rotation.as_euler('zyx', degrees=True)

    # Form the state vector [x, y, z, roll, pitch, yaw]
    state = np.array([tvec[0], tvec[1], tvec[2], roll, pitch, yaw], dtype=np.float32)
    state = state.reshape(-1, 1)  # Reshape to column vector

    if previous_state is None:
        kf.statePost = state
    else:
        # Use Kalman filter to predict the state
        predicted_state = kf.predict()

        # Update the Kalman filter with the current measurement
        kf.correct(state)

    # Store this state for the next iteration
    previous_state = state

    visualize_head_pose_from_matrix(avgCameraPose, gaze_position,curr_poses)

    key = cv2.waitKey(delay) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("w"):

        key =  cv2.waitKey()
        if key == ord("q"):
            break

cv2.destroyAllWindows()
video.release()
