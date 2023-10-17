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

def homography_to_transformation(H):
    T = np.zeros((4, 4))
    T[:3, :3] = H
    T[3, 3] = 1
    return T
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
        # Compute relative Camera Pose for current marker
    H, _ = cv2.findHomography(src_pts, dst_pts)


    return homography_to_transformation(H)


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


def visualize_marker_head_pose(rvec, tvec, camera_matrix, dist_matrix):
    rot_mtx, _ = cv2.Rodrigues(rvec)

    # Convert rotation matrix to Euler angles
    rotation = R.from_matrix(rot_mtx)
    roll, pitch, yaw = rotation.as_euler('zyx', degrees=True)

    # Prepare the transformation matrix for visualization
    cameraPose = np.eye(4)
    cameraPose[:3, :3] = rot_mtx
    cameraPose[:3, 3] = tvec.squeeze()

    visualize_head_pose_from_matrix(cameraPose)

video = "002/world.mp4"
video = cv2.VideoCapture(video)

# Set the starting frame to 500
video.set(cv2.CAP_PROP_POS_FRAMES, 400)
get_intrinsics()
setup_detector()

init_poses = None
kf = kalman_filter_setup()

fps = video.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)  # Real-time delay

weights = [0.5, 0.3, 0.2]  # Example weights for the last three transformations
measurements = []  # List to hold the last n measurements
n = 3  # Number of matrices to consider for moving average



while True:
    ret, frame = video.read()
    if not ret:
        break

    corners, ids, rejected = detector.detectMarkers(frame)
    ids = [id[0] for id in ids]
    cornerss = [marker[0] for marker in corners]
    if init_poses is None:
        init_poses = dict(zip(ids, cornerss))
    curr_poses = dict(zip(ids, cornerss))

    if curr_poses is None:
        continue

    draw_markers(frame, corners)
    display(frame)

    # For each detected marker, visualize its head pose
    for id, corner in zip(ids, corners):
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corner, 0.1341, camera_matrix, dist_matrix)
        visualize_marker_head_pose(rvec, tvec, camera_matrix, dist_matrix)

    avgCameraPose = get_transformation_matrix(curr_poses)

    if avgCameraPose is None:  # Handle the case if you don't get a transformation matrix
        continue

    visualize_head_pose_from_matrix(avgCameraPose)

    key = cv2.waitKey(delay) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
video.release()