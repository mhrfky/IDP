import numpy as np
import cv2

from scipy.spatial.transform import Rotation


def decompose_matrix(matrix):
    rotation = matrix[:3, :3]
    return rotation


def rotation_matrix_to_euler_angles(R):
    # Yaw
    psi = np.arctan2(R[1, 0], R[0, 0])

    # Pitch
    theta = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))

    # Roll
    phi = np.arctan2(R[2, 1], R[2, 2])

    rot = Rotation.from_matrix(R)  # Convert from rotation matrix to Rotation object
    euler_angles = rot.as_euler('zyx',
                                degrees=True)  # Convert to Euler angles ('zyx' is one of the possible conventions)
    print(euler_angles)
    return euler_angles


def visualize_head_pose_from_matrix(matrix):
    R = decompose_matrix(matrix)
    yaw, pitch, roll = rotation_matrix_to_euler_angles(R)

    # Increase the image size for better presentation
    img_size = (600, 600)
    img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255

    center_y = int((90 - yaw) / 180 * img_size[1])
    center_x = int((90 + pitch) / 180 * img_size[0])
    center = (center_x, center_y)

    width = 60  # Increased for better visibility
    height = 30  # Increased for better visibility

    rect = np.array([
        [-width / 2, -height / 2],
        [width / 2, -height / 2],
        [width / 2, height / 2],
        [-width / 2, height / 2]
    ])

    rotation_matrix = cv2.getRotationMatrix2D((0, 0), roll, 1)
    rotated_rect = np.dot(rect, rotation_matrix[:, :2].T)
    translated_rect = rotated_rect + center

    for i in range(4):
        start_point = tuple(translated_rect[i].astype(int))
        end_point = tuple(translated_rect[(i + 1) % 4].astype(int))
        cv2.line(img, start_point, end_point, (255, 0, 0), 2)

    # Add transformation matrix, roll, pitch, and yaw to the bottom right
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 0)  # Black color
    thickness = 1
    line_spacing = 20
    y0 = img_size[1] - (matrix.shape[0] * line_spacing + 3 * line_spacing)

    for i, row in enumerate(matrix):
        text = " ".join(["{:.2f}".format(val) for val in row])
        position = (10, y0 + i * line_spacing)
        cv2.putText(img, text, position, font, font_scale, color, thickness)

    y_pos = y0 + matrix.shape[0] * line_spacing
    cv2.putText(img, f"Yaw: {yaw:.2f}", (10, y_pos), font, font_scale, color, thickness)
    cv2.putText(img, f"Pitch: {pitch:.2f}", (10, y_pos + line_spacing), font, font_scale, color, thickness)
    cv2.putText(img, f"Roll: {roll:.2f}", (10, y_pos + 2 * line_spacing), font, font_scale, color, thickness)

    cv2.imshow('Head Pose', img)


# Example usage:
matrix = np.array([
    [0.8660254, -0.5, 0, 0],
    [0.5, 0.8660254, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
visualize_head_pose_from_matrix(matrix)
