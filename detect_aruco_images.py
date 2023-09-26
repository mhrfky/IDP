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

img_path = "frames/002/frame_0.png"
second_img_path = "frames/002/frame_885.png"

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
	image = cv2.imread(img_path)
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
	detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)



def draw_markers():
	cv2.aruco.drawDetectedMarkers(image, corners)
def get_poses():
	d = []
	tvecs = []
	for id,corner in zip(ids,corners):
		rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corner, 0.18, camera_matrix, dist_matrix)
		rot_mtx, _ = cv2.Rodrigues(rvec)
		cameraPose = np.eye(4)
		cameraPose[:3, :3] = rot_mtx
		cameraPose[:3, 3] = tvec.squeeze()
		tvecs.append((id,tvec))
		d.append((id,cameraPose))

	return {np.squeeze(i[0]) : np.squeeze(i[1]) for i in sorted(d, key=lambda x: x[0])}, {np.squeeze(i[0]) : np.squeeze(i[1]) for i in sorted(tvecs, key=lambda x: x[0])}

def display():
	cv2.imshow("Image", image)
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

def get_homography_and_warp(img,corners):
    h, w = 80, 80  # or whatever dimensions you choose

    reference_corners = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)
    contour_corners = np.squeeze(corners)
    print("Number of contour corners:", len(contour_corners))

    # Compute the homography
    H, _ = cv2.findHomography(contour_corners, reference_corners)

    # Warp the region enclosed by the contour
    warped = cv2.warpPerspective(img, H, (w, h))

    return warped

get_intrinsics()
read_image()
setup_detector()

corners, ids, rejected = detector.detectMarkers(image)
ids = [id[0] for id in ids]
for id, corner in zip(ids, corners):
	rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corner, 0.18, camera_matrix, dist_matrix)

init_corners_dict= {i[0] : np.squeeze(i[1]) for i in sorted(zip(ids,corners),key=lambda x: x[0])}
init_poses,tvecs = get_poses()
draw_markers()
display()

read_next_frame()
corners, ids, rejected = detector.detectMarkers(image)
ids = np.squeeze(ids)

marker_positions = []
for i,corner_set in enumerate(corners):
	pos = np.mean(np.squeeze(corner_set),0)
	marker_positions.append(pos)

points3d_to_compare = np.array([tvecs[i] for i in ids])
marker_positions = np.array(marker_positions)

#Number of points are not enough
a = cv2.solvePnP(points3d_to_compare,marker_positions,camera_matrix, dist_matrix)





draw_markers()
display()