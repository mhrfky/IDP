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
import msgpack


def load_aruco_dictionary_from_yaml(filename):
	fs = cv2.FileStorage(filename, cv2.FileStorage_READ)
	if not fs.isOpened():
		print(f"Failed to open {filename}")
		return None

	aruco_dict = cv2.aruco.Dictionary()
	fn = fs.root()
	aruco_dict.readDictionary(fn)


	return aruco_dict


def arg_parse():
	global args
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True, help="path to input image containing ArUCo tag")
	ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tag to detect")
	args = vars(ap.parse_args())





def get_intrinsics():
	global camera_matrix, dist_matrix
	camera_matrix = np.array(
		[[794.3311439869655, 0.0, 633.0104437728625], [0.0, 793.5290139393004, 397.36927353414865], [0.0, 0.0, 1.0]])
	dist_matrix = np.array([[-0.3758628065070806, 0.1643326166951343, 0.00012182540692089567, 0.00013422608638039466,
							 0.03343691733865076, 0.08235235770849726, -0.08225804883227375, 0.14463365333602152]])

def read_image():
	global image
	image = cv2.imread(args["image"])
	h, w, _ = image.shape
	width = 600
	height = int(width * (h / w))
	image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

def get_tag():
	# verify that the supplied ArUCo tag exists and is supported by OpenCV
	if ARUCO_DICT.get(args["type"], None) is None:
		print(f"ArUCo tag type '{args['type']}' is not supported")
		sys.exit(0)
def setup_detector():
	global detector
	# load the ArUCo dictionary, grab the ArUCo parameters, and detect
	# the markers
	print("Detecting '{}' tags....".format(args["type"]))
	filename = "my_custom_dictionary.yml"
	arucoDict = load_aruco_dictionary_from_yaml(filename)
	arucoParams = cv2.aruco.DetectorParameters()
	detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)



def draw_markers():
	cv2.aruco.drawDetectedMarkers(image, corners)
def get_poses():
	d = []
	for id,corner in zip(ids,corners):
		rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corner, 0.18, camera_matrix, dist_matrix)
		rot_mtx, _ = cv2.Rodrigues(rvec)
		cameraPose = np.eye(4)
		cameraPose[:3, :3] = rot_mtx
		cameraPose[:3, 3] = tvec.squeeze()
		d.append((id,cameraPose))

	return [i[1] for i in sorted(d, key=lambda x: x[0])]

def display():
	cv2.imshow("Image", image)
	cv2.waitKey(0)

def read_next_frame():
	global image
	image = cv2.imread("frames/world112.jpg")
	h, w, _ = image.shape
	width = 600
	height = int(width * (h / w))
	image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

def get_related_init(ids):
	global init_poses
	return [init_poses[id[0]] for id in ids ]
arg_parse()
get_intrinsics()
read_image()
get_tag()
setup_detector()
corners, ids, rejected = detector.detectMarkers(image)
init_poses = get_poses()
draw_markers()
display()
read_next_frame()
corners, ids, rejected = detector.detectMarkers(image)
curr_poses = get_poses()

all_estimates = []

related_poses = get_related_init(ids)
for initPose, currPose in zip(related_poses, curr_poses):
    # Compute CameraPose for current marker
    cameraPoseEstimate = np.dot(np.linalg.inv(initPose), currPose)
    all_estimates.append(cameraPoseEstimate)

# Compute the average pose
avgCameraPose = np.mean(all_estimates, axis=0)

print(avgCameraPose)

draw_markers()
display()