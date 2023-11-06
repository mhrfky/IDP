import numpy as np
import cv2
from visualize_2d_new import visualize_head_pose_from_matrix
import pandas as pd


def setup_detector():
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

    detector_ = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    return detector_


def draw_markers(image, corners_for_drawing):
    cv2.aruco.drawDetectedMarkers(image, corners_for_drawing)


def get_intrinsics():
    global camera_matrix, dist_matrix
    camera_matrix = np.array(
        [[794.3311439869655, 0.0, 633.0104437728625], [0.0, 793.5290139393004, 397.36927353414865], [0.0, 0.0, 1.0]])
    dist_matrix = np.array([[-0.3758628065070806, 0.1643326166951343, 0.00012182540692089567, 0.00013422608638039466,
                             0.03343691733865076, 0.08235235770849726, -0.08225804883227375, 0.14463365333602152]])


def display(image):
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
    # print(src_pts, dst_pts)
    H, _ = cv2.findHomography(src_pts, dst_pts)

    # Extract rotation matrix from the homography
    R = decompose_homography(camera_matrix, H)

    T = np.eye(4)
    T[:3, :3] = R

    return T


def get_gaze_positions():
    # Load the DataFrame (assuming you've already done this)
    df = pd.read_csv("000/exports/000/gaze_positions.csv")

    # Group by 'world_index' and get the row with the highest 'confidence' for each group
    df_max_confidence = df.loc[df.groupby('world_index')['confidence'].idxmax()]

    df_max_confidence = df_max_confidence[['world_index', 'norm_pos_x', 'norm_pos_y']]

    return df_max_confidence.values.tolist()


def add_new_marker_using_homography(init_poses, new_marker_corners, H_current, camera_matrix, new_marker_id):

    # Compute the inverse of the current homography
    H_current_inv = np.linalg.inv(H_current)

    # Initialize an array to hold the estimated initial positions of the new marker corners
    estimated_init_corners = np.zeros_like(new_marker_corners)

    # Map each corner point of the new marker using the inverse homography
    for i, corner in enumerate(new_marker_corners):
        homogenous_corner = np.append(corner, 1)  # Convert to homogenous coordinates
        estimated_init_corner = np.dot(H_current_inv, homogenous_corner)
        # Normalize to get the estimated initial position in Cartesian coordinates
        estimated_init_corners[i] = estimated_init_corner[:2] / estimated_init_corner[2]

    # Add the estimated initial position of the new marker to init_poses
    init_poses[new_marker_id] = estimated_init_corners

    return init_poses


video = "000/world.mp4"
video = cv2.VideoCapture(video)
calibrated = False
# Set the starting frame to 500
video.set(cv2.CAP_PROP_POS_FRAMES, 1)
get_intrinsics()
detector = setup_detector()

init_poses = None

fps = video.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)  # Real-time delay
number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

gaze_positions = get_gaze_positions()
gaze_positions_iter = iter(gaze_positions)
avg_camera_pose = None

while True:
    ret, frame = video.read()
    gaze_position = next(gaze_positions_iter)

    if not ret:
        break
    corners, ids, rejected = detector.detectMarkers(frame)
    if ids is None:
        continue  # Skip the rest of the loop if no markers are detected
    ids = [id[0] for id in ids]
    cornerss = [marker[0] for marker in corners]

    curr_poses = dict(zip(ids, cornerss))
    if calibrated:

        try:
            avg_camera_pose = get_transformation_matrix(curr_poses)
        except Exception as e:
            print(f"No markers is not detected this frame, hence this exception has occured : {e}")

        if avg_camera_pose is  None:
            continue

        visualize_head_pose_from_matrix(avg_camera_pose, gaze_position, curr_poses)

    elif init_poses is None or (not calibrated and len(curr_poses.keys()) > len(init_poses)):
        init_poses = curr_poses

    draw_markers(frame, corners)
    display(frame)

    key = cv2.waitKey(delay) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r") and not calibrated:
        calibrated = True

    elif key == ord("w"):

        key = cv2.waitKey()
        if key == ord("q"):
            break

cv2.destroyAllWindows()
video.release()
