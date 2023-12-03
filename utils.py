import numpy as np
from config import *
import math


def rotate_point(point, center, angle):
    """ Rotate a point around a center by a given angle. """
    angle_rad = np.deg2rad(angle)
    ox, oy = center
    px, py = point

    qx = ox + np.cos(angle_rad) * (px - ox) - np.sin(angle_rad) * (py - oy)
    qy = oy + np.sin(angle_rad) * (px - ox) + np.cos(angle_rad) * (py - oy)

    return qx, qy


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
    return center, translated_rect


def get_eye_dilation_radius(eye_diameter):
    # Calculate the average eye diameter for the circle size
    eye_diameter_mean = (eye_diameter[0] + eye_diameter[1]) / 2
    eye_diameter_mean_normalized = eye_diameter_mean / EYE_DIAMETER_NORMALIZING_FACTOR
    eye_dilation_radius = math.floor(eye_diameter_mean_normalized * GAZE_CIRCLE_RADIUS)
    return eye_dilation_radius


def estimate_gaze_pos(fov_center_pos, gaze_position, roll):
    norm_pos_x, norm_pos_y = gaze_position
    gaze_x = fov_center_pos[0] - WIDTH / 2 + norm_pos_x * WIDTH
    gaze_y = fov_center_pos[
                 1] + HEIGHT / 2 - norm_pos_y * HEIGHT  # inverted Y axis as image coordinates work from top-left
    rotated_gaze_x, rotated_gaze_y = rotate_point((gaze_x, gaze_y), (fov_center_pos[0], fov_center_pos[1]), roll)
    return (rotated_gaze_x, rotated_gaze_y)
