import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2


def visualize_fov(transformation_matrix):
    r = 7.5
    yaw_head = np.arctan2(transformation_matrix[0, 2], transformation_matrix[0, 0]) * (180 / np.pi)

    sensitivity_factor = 0.5  # Lower value, less sensitivity. You can adjust this value.
    yaw_head *= sensitivity_factor  # Reduce the sensitivity of the yaw movement

    fov = 45  # Also consider reducing the field of view for less jumpiness
    apparent_height = 150
    y_position = transformation_matrix[1, 3] * 50

    img = np.ones((300, 600, 3), dtype=np.uint8) * 255

    center_x = int(300 + yaw_head * 300 / 50)
    center_y = int(150 - y_position)

    start_point = (center_x - int(fov), center_y - int(apparent_height / 2))
    end_point = (center_x + int(fov), center_y + int(apparent_height / 2))
    color = (0, 0, 255)
    thickness = 2

    img = cv2.rectangle(img, start_point, end_point, color, thickness)

    cv2.imshow('Field of View', img)


if __name__ == "__main__":
    # Example 4x4 transformation matrix
    T = np.array([[0.98, -0.17, 0.08, 0.5],
                  [0.17, 0.98, 0.03, 0.1],
                  [-0.08, -0.03, 0.99, 0.2],
                  [0, 0, 0, 1]])

    visualize_fov(T)
