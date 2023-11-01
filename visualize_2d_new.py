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


N = 10  # number of past poses to store and visualize

past_poses = []  # a global variable to keep the previous poses

def visualize_head_pose_from_matrix(matrix, gaze_position, curr_poses= None):

    global past_poses
    R = decompose_matrix(matrix)
    yaw, pitch, roll = rotation_matrix_to_euler_angles(R)

    width = 120
    height = 80
    video_width, video_height = 1280, 720
    rect_width, rect_height = width, height

    img_size = (600, 600)
    img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255

    center_y = int((90 - yaw) / 180 * img_size[1])
    center_x = int((90 + pitch) / 180 * img_size[0])
    center = (center_x, center_y)


    rect = np.array([
        [-width / 2, -height / 2],
        [width / 2, -height / 2],
        [width / 2, height / 2],
        [-width / 2, height / 2]
    ])

    rotation_matrix = cv2.getRotationMatrix2D((0, 0), roll, 1)
    rotated_rect = np.dot(rect, rotation_matrix[:, :2].T)
    translated_rect = rotated_rect + center

    # Drawing gaze circle inside the rectangle
    _, norm_pos_x, norm_pos_y = gaze_position
    gaze_x = center_x - width/2 + norm_pos_x * (width)
    gaze_y = center_y + height/2 - norm_pos_y * (height)  # inverted Y axis as image coordinates work from top-left
    cv2.circle(img, (int(gaze_x), int(gaze_y)), 5, (0, 255, 0), -1)  # Green circle with radius of 5
    for id, marker in curr_poses.items():
        normalized_marker = [
            [(point[0] / video_width) * rect_width + center[0] - rect_width / 2,
             (point[1] / video_height) * rect_height + center[1] - rect_height / 2]
            for point in marker]

        # Draw the markers inside the rectangle
        for j in range(4):
            start_point = (int(normalized_marker[j][0]), int(normalized_marker[j][1]))
            end_point = (int(normalized_marker[(j + 1) % 4][0]), int(normalized_marker[(j + 1) % 4][1]))
            cv2.line(img, start_point, end_point, (0, 255, 0), 2)  # Green color for markers

    # Add the current pose to the buffer and remove the oldest if exceeded size
    past_poses.append(translated_rect)
    if len(past_poses) > N:
        past_poses.pop(0)

    # Draw past poses with decreasing opacity
    for i, previous_rect in enumerate(past_poses[::-1]):
        alpha = (N - i) / N  # fades out the further we go back in history
        for j in range(4):
            start_point = tuple(previous_rect[j].astype(int))
            end_point = tuple(previous_rect[(j + 1) % 4].astype(int))
            cv2.line(img, start_point, end_point, (255 * alpha, 0, 0), 2)

    # ... [rest of the function remains unchanged]

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

