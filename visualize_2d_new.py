import math

import numpy as np
import cv2
from config import *
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from utils import rotate_point, get_translated_view_rectangle, get_eye_dilation_radius, estimate_gaze_pos

N = 10  # number of past poses to store and visualize


class Plotter:
    def __init__(self, x_lim=(0, 5000), y_lim=(0, 1), title="", x_label="", y_label=""):
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], '-')  # Initialize an empty line plot
        self.ax.set_xlim(*x_lim)  # Set x-axis limits
        self.ax.set_ylim(*y_lim)  # Set y-axis limits
        self.ax.set_title(title)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)

    def update(self, frame_id, value):
        x_data, y_data = self.line.get_data()
        x_data = list(x_data)
        y_data = list(y_data)
        x_data.append(frame_id)
        y_data.append(value)
        # Update line data
        self.line.set_data(x_data, y_data)

        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class Visualizer:
    def __init__(self, number_of_frames, plot_blink=True, plot_dilation=True):
        if plot_blink:
            self.blink_plotter = Plotter((0, number_of_frames), (0, 1),"Blink Rate over time", "Time", "Blink rate")
        else:
            self.blink_plotter = None
        if plot_dilation:
            self.dilation_plotter = Plotter((0, number_of_frames), (0, 100), "Eye Dilation over time", "Time", "Eye Dilation")
        else:
            self.dilation_plotter = None

        self.past_poses = []
        self.gaze_positions = []

    def update(self, translated_rect, fov_center, markers, head_pose, gaze_position, blink_rate, dilation_radius,
               frame_id):
        img = np.ones((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8) * 255
        fields_dict = head_pose
        if self.blink_plotter is not None:
            self.blink_plotter.update(frame_id, blink_rate)
            fields_dict["Blink rate"] = blink_rate
        if self.dilation_plotter is not None:
            self.dilation_plotter.update(frame_id, dilation_radius)
            fields_dict["Dilation Radius"] = dilation_radius


        self.visualize_fov(img, translated_rect)
        self.draw_gaze_and_dilation_circle(img, gaze_position, dilation_radius)
        self.draw_markers(img, fov_center, markers, head_pose['roll'])

        self.write_features(img, fields_dict)

        cv2.imshow("View", img)

    def visualize_fov(self, img, translated_rect):
        self.past_poses.append(translated_rect)
        if len(self.past_poses) > N:
            self.past_poses.pop(0)
        # Draw past poses with decreasing opacity
        for i, previous_rect in enumerate(self.past_poses[::-1]):
            alpha = (N - i) / N  # fades out the further we go back in history
            for j in range(4):
                start_point = tuple(previous_rect[j].astype(int))
                end_point = tuple(previous_rect[(j + 1) % 4].astype(int))
                color = [c * alpha for c in FOV_COLOR]
                cv2.line(img, start_point, end_point, color, FOV_THICKNESS)

    def write_features(self, img, fields: dict):
        y_pos = 3 * TEXT_LINE_SPACING
        i = 0
        for field_name, field_value in fields.items():
            cv2.putText(img,
                        f"{field_name}: {field_value:.2f}",
                        (10, y_pos + TEXT_LINE_SPACING * i),
                        FONT,
                        FONT_SCALE,
                        TEXT_COLOR,
                        TEXT_THICKNESS)
            i += 1

    def draw_markers(self, img, center, markers, roll):
        for marker_id, marker in markers.items():
            marker_center = [sum(p[0] for p in marker) / len(marker),
                             sum(p[1] for p in marker) / len(marker)]

            scaled_marker = [[marker_center[0] + (point[0] - marker_center[0]) * MARKER_ENLARGEMENT_RATE,
                              marker_center[1] + (point[1] - marker_center[1]) * MARKER_ENLARGEMENT_RATE]
                             for point in marker]

            # Rotate the markers
            rotated_marker = [rotate_point(point, marker_center, roll) for point in scaled_marker]

            # Normalize the marker points
            normalized_marker = [[(point[0] / VIDEO_WIDTH) * WIDTH + center[0] - WIDTH / 2,
                                  (point[1] / VIDEO_HEIGHT) * HEIGHT + center[1] - HEIGHT / 2]
                                 for point in rotated_marker]

            # Draw the markers
            for j in range(4):
                start_point = (int(normalized_marker[j][0]), int(normalized_marker[j][1]))
                end_point = (int(normalized_marker[(j + 1) % 4][0]), int(normalized_marker[(j + 1) % 4][1]))
                cv2.line(img, start_point, end_point, MARKER_VISIBLE_COLOR, MARKER_THICKNESS)

    def draw_gaze_and_dilation_circle(self, img, gaze_position, eye_diameter):
        eye_dilation_radius = eye_diameter / EYE_DIAMETER_NORMALIZING_FACTOR
        eye_dilation_radius = math.floor(eye_dilation_radius * GAZE_CIRCLE_RADIUS)

        self.gaze_positions.append((gaze_position[0], gaze_position[1]))

        # Draw the rotated gaze circle
        cv2.circle(img, (int(gaze_position[0]), int(gaze_position[1])), GAZE_CIRCLE_RADIUS, GAZE_CIRCLE_COLOR, -1)
        cv2.circle(img, (int(gaze_position[0]), int(gaze_position[1])), eye_dilation_radius, EYE_DILATION_COLOR,
                   EYE_DIAMETER_CIRCLE_THICKNESS)

def generate_heatmap(gaze_positions, img_size, bins=40):
    # Convert gaze positions to pixel coordinates
    pixel_x_data = [pos[0] for pos in gaze_positions]
    pixel_y_data = [pos[1] for pos in gaze_positions]

    # Create a 2D histogram from the gaze data
    heatmap, xedges, yedges = np.histogram2d(pixel_x_data, pixel_y_data, bins=bins)

    # Display the heatmap
    plt.imshow(heatmap.T, origin='lower', cmap='hot', extent=[0, img_size[0], 0, img_size[1]])
    plt.colorbar()
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Gaze Heatmap')
    plt.show()


# Global list to store gaze positions
gaze_positions = []
