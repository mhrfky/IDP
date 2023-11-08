import numpy as np
import cv2

from scipy.spatial.transform import Rotation

N = 10  # number of past poses to store and visualize

past_poses = []  # a global variable to keep the previous poses
WIDTH = 240
HEIGHT = 160
VIDEO_WIDTH, VIDEO_HEIGHT = 1280, 720
IMG_SIZE = (600, 600)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
TEXT_COLOR = (0, 0, 0)  # Black color
TEXT_THICKNESS = 1
TEXT_LINE_SPACING = 20
FOV_THICKNESS = 2
FOV_COLOR = (255,0,0)
MARKER_THICKNESS=2
MARKER_VISIBLE_COLOR = (0,0,255)
MARKER_NOT_VISIBLE_COLOR = (0,255,255)
GAZE_CIRCLE_COLOR = (0,255,0)
GAZE_CIRCLE_RADIUS = 5
MARKER_ENLARGEMENT_RATE = 2

def visualize_head_pose_from_yaw_pitch_roll(yaw,pitch,roll,gaze_position,markers = None):
    img = np.ones((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8) * 255

    center, center_x, center_y, translated_rect = get_translated_view_rectangle(pitch, roll, yaw)

    draw_gaze_circle(center_x, center_y, gaze_position, img)
    draw_markers(center, markers, img)

    # Add the current pose to the buffer and remove the oldest if exceeded size
    visualize_fov(img, translated_rect)

    write_rotations(img, pitch, roll, yaw)
    cv2.imshow('Head Pose', img)


def visualize_head_pose_from_matrix(transformation_matrix, gaze_position, markers=None):
    global past_poses

    rotation_matrix = decompose_matrix(transformation_matrix)
    yaw, pitch, roll = rotation_matrix_to_euler_angles(rotation_matrix)

    img = np.ones((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8) * 255

    center, center_x, center_y, translated_rect = get_translated_view_rectangle(pitch, roll, yaw)

    draw_gaze_circle(center_x, center_y, gaze_position, img)
    draw_markers(center, markers, img)

    # Add the current pose to the buffer and remove the oldest if exceeded size
    visualize_fov(img, translated_rect)

    # write_rotations(img, , pitch, roll, yaw)
    cv2.imshow('Head Pose', img)


def decompose_matrix(matrix):
    rotation = matrix[:3, :3]
    return rotation


def rotation_matrix_to_euler_angles(rotation_matrix):
    rot = Rotation.from_matrix(rotation_matrix)  # Convert from rotation matrix to Rotation object
    euler_angles = rot.as_euler('zyx',
                                degrees=True)  # Convert to Euler angles ('zyx' is one of the possible conventions)
    return euler_angles


def get_translated_view_rectangle(pitch, roll, yaw):
    center_y = int((90 - yaw) / 180 * IMG_SIZE[1])
    center_x = int((90 + pitch) / 180 * IMG_SIZE[0])
    center = (center_x, center_y)
    rect = np.array([
        [-WIDTH / 2, -HEIGHT / 2],
        [WIDTH / 2, -HEIGHT / 2],
        [WIDTH / 2, HEIGHT / 2],
        [-WIDTH / 2, HEIGHT / 2]
    ])
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), roll, 1)
    rotated_rect = np.dot(rect, rotation_matrix[:, :2].T)
    translated_rect = rotated_rect + center
    return center, center_x, center_y, translated_rect


def visualize_fov(img, translated_rect):
    global past_poses
    past_poses.append(translated_rect)
    if len(past_poses) > N:
        past_poses.pop(0)
    # Draw past poses with decreasing opacity
    for i, previous_rect in enumerate(past_poses[::-1]):
        alpha = (N - i) / N  # fades out the further we go back in history
        for j in range(4):
            start_point = tuple(previous_rect[j].astype(int))
            end_point = tuple(previous_rect[(j + 1) % 4].astype(int))
            color = [c * alpha for c in FOV_COLOR]
            cv2.line(img, start_point, end_point, color,FOV_THICKNESS)


def write_rotations(img,  pitch, roll, yaw):
    # Add transformation matrix, roll, pitch, and yaw to the bottom righ

    y_pos = 3 * TEXT_LINE_SPACING
    cv2.putText(img, f"Yaw: {yaw:.2f}", (10, y_pos), FONT, FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
    cv2.putText(img, f"Pitch: {pitch:.2f}", (10, y_pos + TEXT_LINE_SPACING), FONT, FONT_SCALE, TEXT_COLOR,
                TEXT_THICKNESS)
    cv2.putText(img, f"Roll: {roll:.2f}", (10, y_pos + 2 * TEXT_LINE_SPACING), FONT, FONT_SCALE, TEXT_COLOR,
                TEXT_THICKNESS)



def draw_markers(center, markers, img, ):
    for marker_id, marker in markers.items():
        # Calculate the center of the marker
        marker_center = [sum(p[0] for p in marker) / len(marker),
                         sum(p[1] for p in marker) / len(marker)]

        # Scale the markers by the enlargement rate
        scaled_marker = [
            [marker_center[0] + (point[0] - marker_center[0]) * MARKER_ENLARGEMENT_RATE,
             marker_center[1] + (point[1] - marker_center[1]) * MARKER_ENLARGEMENT_RATE]
            for point in marker]

        # Normalize the marker points
        normalized_marker = [
            [(point[0] / VIDEO_WIDTH) * WIDTH + center[0] - WIDTH / 2,
             (point[1] / VIDEO_HEIGHT) * HEIGHT + center[1] - HEIGHT / 2]
            for point in scaled_marker]

        # Draw the markers inside the rectangle
        for j in range(4):
            start_point = (int(normalized_marker[j][0]), int(normalized_marker[j][1]))
            end_point = (int(normalized_marker[(j + 1) % 4][0]), int(normalized_marker[(j + 1) % 4][1]))
            cv2.line(img, start_point, end_point, MARKER_VISIBLE_COLOR, MARKER_THICKNESS)



def draw_gaze_circle(center_x, center_y, gaze_position, img):
    # Drawing gaze circle inside the rectangle
    _, norm_pos_x, norm_pos_y = gaze_position
    gaze_x = center_x - WIDTH / 2 + norm_pos_x * WIDTH
    gaze_y = center_y + HEIGHT / 2 - norm_pos_y * HEIGHT  # inverted Y axis as image coordinates work from top-left
    cv2.circle(img, (int(gaze_x), int(gaze_y)), GAZE_CIRCLE_RADIUS, GAZE_CIRCLE_COLOR, -1)  # Green circle with radius of 5
