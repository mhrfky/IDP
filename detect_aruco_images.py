'''
Sample Command:-
python detect_aruco_images.py --image Images/test_image_1.png --type DICT_5X5_100
'''
import numpy as np
from utils import ARUCO_DICT, aruco_display
import argparse
import cv2
import sys
import numpy as np

first_img_path = "frames/002/frame_500.png"
second_img_path = "frames/002/frame_886.png"

def load_aruco_dictionary_from_yaml(filename):
	fs = cv2.FileStorage(filename, cv2.FileStorage_READ)
	if not fs.isOpened():
		print(f"Failed to open {filename}")
		return None

	aruco_dict = cv2.aruco.Dictionary()
	fn = fs.root()
	aruco_dict.readDictionary(fn)


	return aruco_dict







def get_intrinsics():
	global camera_matrix, dist_matrix
	camera_matrix = np.array(
		[[794.3311439869655, 0.0, 633.0104437728625], [0.0, 793.5290139393004, 397.36927353414865], [0.0, 0.0, 1.0]])
	dist_matrix = np.array([[-0.3758628065070806, 0.1643326166951343, 0.00012182540692089567, 0.00013422608638039466,
							 0.03343691733865076, 0.08235235770849726, -0.08225804883227375, 0.14463365333602152]])

def read_image():
	global image
	image = cv2.imread(first_img_path)
	h, w, _ = image.shape
	width = 600
	height = int(width * (h / w))
	image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


def setup_detector():
	global detector
	# load the ArUCo dictionary, grab the ArUCo parameters, and detect
	# the markers
	filename = "my_custom_dictionary.yml"
	arucoDict = load_aruco_dictionary_from_yaml(filename)
	arucoParams = cv2.aruco.DetectorParameters()
	arucoParams.adaptiveThreshConstant = 7
	arucoParams.adaptiveThreshWinSizeMax = 30
	arucoParams.adaptiveThreshWinSizeMin = 3
	arucoParams.adaptiveThreshWinSizeStep = 25

	detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)



def draw_markers():
	cv2.aruco.drawDetectedMarkers(image, corners)

def get_poses():
	d = []
	for id,corner in zip(ids,corners):
		rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corner, 0.1341 , camera_matrix, dist_matrix)
		rot_mtx, _ = cv2.Rodrigues(rvec)
		cameraPose = np.eye(4)
		cameraPose[:3, :3] = rot_mtx
		cameraPose[:3, 3] = tvec.squeeze()
		d.append((id,cameraPose))

	return {i[0][0]:i[1] for i in sorted(d, key=lambda x: x[0])}

def display():
	# Get the dimensions of the image
	height, width = image.shape[:2]

	# Define the scale factor
	scale_factor = 2.0  # Example: 2.0 will double the display size of the image

	# Resize the image
	image_resized = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))

	cv2.imshow("Image", image_resized)
	cv2.waitKey(0)

def read_next_frame():
	global image
	image = cv2.imread(second_img_path)
	h, w, _ = image.shape
	width = 600
	height = int(width * (h / w))
	image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

def get_related_init(ids):
	global init_poses
	return [init_poses[id[0]] for id in ids ]


get_intrinsics()
read_image()

setup_detector()
corners, ids, rejected = detector.detectMarkers(image)
init_poses = get_poses()
draw_markers()
display()
read_next_frame()
corners, ids, rejected = detector.detectMarkers(image)
curr_poses = get_poses()

all_estimates = []



for id in curr_poses.keys():
	if id not in init_poses.keys():
		continue
	init_pose = init_poses[id]
	curr_pose = curr_poses[id]
	# Compute CameraPose for current marker
	cameraPoseEstimate = np.dot(np.linalg.inv(init_pose), curr_pose)
	all_estimates.append(cameraPoseEstimate)

# Compute the average pose
avgCameraPose = np.mean(all_estimates, axis=0)

print(avgCameraPose)

draw_markers()
display()