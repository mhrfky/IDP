import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2



def rotate_point(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

def visualize_fov(transformation_matrix):
    r = 7.5
    yaw_head = np.arctan2(transformation_matrix[0, 2], transformation_matrix[0, 0]) * (180 / np.pi)

    r20 = transformation_matrix[2, 0]
    r21 = transformation_matrix[2, 1]
    r22 = transformation_matrix[2, 2]
    pitch = np.arctan2(-r20, np.sqrt(r21**2 + r22**2))

    r10 = transformation_matrix[1, 0]
    r00 = transformation_matrix[0, 0]
    roll = np.arctan2(r10, r00)

    sensitivity_factor = 0.5  # Lower value, less sensitivity. You can adjust this value.
    yaw_head *= sensitivity_factor  # Reduce the sensitivity of the yaw movement

    fov = 180  # Also consider reducing the field of view for less jumpiness
    apparent_height = 150
    y_position = transformation_matrix[1, 3] * 50  # Reverse the y_position

    img = np.ones((300, 600, 3), dtype=np.uint8) * 255

    center_x = int(300 + yaw_head * 300 / 50)
    center_y = int(150 + y_position)  # Adjusted the addition here

    # Compute the four vertices of the rectangle
    top_left = (center_x - int(fov), center_y - int(apparent_height / 2))
    top_right = (center_x + int(fov), center_y - int(apparent_height / 2))
    bottom_right = (center_x + int(fov), center_y + int(apparent_height / 2))
    bottom_left = (center_x - int(fov), center_y + int(apparent_height / 2))


    # Draw the rotated rectangle
    color = (0, 0, 255)
    thickness = 2
    pts = np.array([top_left, top_right, bottom_right, bottom_left], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

    cv2.imshow('Field of View', img)

if __name__ == "__main__":
    # Example 4x4 transformation matrix
    T = np.array([[0.98, -0.17, 0.08, 0.5],
                  [0.17, 0.98, 0.03, 0.1],
                  [-0.08, -0.03, 0.99, 0.2],
                  [0, 0, 0, 1]])

    visualize_fov(T)
