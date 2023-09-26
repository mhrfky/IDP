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


def get_transformation_matrix(curr_poses):
    all_rotations = []
    all_translations = []

    for id, curr_pose in curr_poses.items():
        if id not in init_poses:
            continue

        init_pose = init_poses[id]

        # Compute relative Camera Pose for current marker
        relative_pose = np.dot(np.linalg.inv(init_pose), curr_pose)

        # Extract rotation and translation from the transformation matrix
        rotation = relative_pose[:3, :3]
        translation = relative_pose[:3, 3]

        all_rotations.append(rotation)
        all_translations.append(translation)

    # Convert rotation matrices to quaternions
    rotations_quaternion = [R.from_matrix(rot).as_quat() for rot in all_rotations]

    # Average the quaternions and translations
    avg_quaternion = np.mean(rotations_quaternion, axis=0)
    avg_translation = np.mean(all_translations, axis=0)

    # Convert averaged quaternion back to rotation matrix
    avg_rotation = R.from_quat(avg_quaternion).as_matrix()

    # Form the averaged transformation matrix
    avgCameraPose = np.eye(4)
    avgCameraPose[:3, :3] = avg_rotation
    avgCameraPose[:3, 3] = avg_translation
    return avgCameraPose


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


video = "002/world.mp4"
video = cv2.VideoCapture(video)
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

    if init_poses is None:
        init_poses = get_poses(ids, corners)

    curr_poses = get_poses(ids, corners)

    if curr_poses is None:
        continue

    draw_markers(frame, corners)
    display(frame)

    avgCameraPose = get_transformation_matrix(curr_poses)

    if avgCameraPose is None:  # Handle the case if you don't get a transformation matrix
        continue

    # Predict the next state
    prediction = kf.predict()

    # Extracting the Translation
    measurement_translation = avgCameraPose[:3, 3].reshape((3, 1))
    rot_matrix = avgCameraPose[:3, :3]
    rotation_vec, _ = cv2.Rodrigues(rot_matrix)
    measurement = np.vstack((measurement_translation, rotation_vec))

    measurements.append(measurement)
    if len(measurements) > n:
        measurements.pop(0)  # Remove the oldest measurement if we have more than n

    # Ensure the list of weights is the same length as measurements
    valid_weights = weights[-len(measurements):]

    # Normalize weights
    weights_sum = sum(valid_weights)
    normalized_weights = [w / weights_sum for w in valid_weights]

    avg_measurement = np.average(measurements, axis=0, weights=normalized_weights)
    avg_measurement = avg_measurement.astype(np.float32)

    # Predict the next state
    prediction = kf.predict()

    # Correct the predicted state
    corrected_state = kf.correct(avg_measurement)



    corrected_translation = corrected_state[:3]
    corrected_rotation_vec = corrected_state[3:]
    corrected_rot_matrix, _ = cv2.Rodrigues(corrected_rotation_vec)
    corrected_pose = np.eye(4)
    corrected_pose[:3, :3] = corrected_rot_matrix
    corrected_pose[:3, 3] = corrected_translation.flatten()



    visualize_fov(corrected_pose)



    # Use corrected_state for further processing as needed
    # ...

    key = cv2.waitKey(delay) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
video.release()
