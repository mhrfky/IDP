import numpy as np
from config import *
import math


def gaussian_kernel(size, sigma=1):
    """Generates a 2D Gaussian kernel."""
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()
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
    print("fuck me oh")

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
    gaze_y = fov_center_pos[1] + HEIGHT / 2 - norm_pos_y * HEIGHT  # inverted Y axis as image coordinates work from top-left
    rotated_gaze_x, rotated_gaze_y = rotate_point((gaze_x, gaze_y), (fov_center_pos[0], fov_center_pos[1]), roll)
    return (rotated_gaze_x, rotated_gaze_y)

def get_center_of_marker_from_corners(marker):
    marker_center = [sum(p[0] for p in marker) / len(marker),
                     sum(p[1] for p in marker) / len(marker)]
    return marker_center
def normalize_marker_positions(markers, img_size, bins):
    normalized_positions = []
    for i, marker in markers.items():
        marker_center = get_center_of_marker_from_corners(marker)
        normalized_x = marker_center[0] / img_size[0]
        normalized_y = marker_center[1] / img_size[1]

        # Scale to heatmap bin range
        heatmap_x = int(normalized_x * (bins - 1))
        heatmap_y = int(normalized_y * (bins - 1))

        normalized_positions.append((heatmap_x, heatmap_y))

    return normalized_positions

def get_adjusted_normalized_markers(center,markers,roll):
    normalized_markers = []
    for marker_id, marker in markers.items():
        marker_center = get_center_of_marker_from_corners(marker)
        scaled_marker = [[marker_center[0] + (point[0] - marker_center[0]) * MARKER_ENLARGEMENT_RATE,
                          marker_center[1] + (point[1] - marker_center[1]) * MARKER_ENLARGEMENT_RATE]
                         for point in marker]

        # Rotate the markers
        rotated_marker = [rotate_point(point, marker_center, -roll) for point in scaled_marker]

        # Normalize the marker points
        normalized_marker = [[(point[0] / VIDEO_WIDTH) * WIDTH + center[0] - WIDTH / 2,
                              (point[1] / VIDEO_HEIGHT) * HEIGHT + center[1] - HEIGHT / 2]
                             for point in rotated_marker]
        marker_center = get_center_of_marker_from_corners(normalized_marker)

        normalized_markers.append((marker_id,marker_center,normalized_marker))
    return normalized_markers