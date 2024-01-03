import math

import numpy as np
import cv2
from config import *
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from utils import gaussian_kernel

import matplotlib.pyplot as plt
import numpy as np


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Heatmappper:
    def __init__(self, img_size, bins=40):
        self.bins = bins
        self.img_size = img_size
        self.heatmap_data = np.zeros((bins, bins))  # Initialize the heatmap data

        # Create the heatmap figure and axes only once
        self.fig, self.ax = plt.subplots()
        self.heatmap_plot = self.ax.imshow(self.heatmap_data, cmap='hot', aspect='auto')
        self.colorbar = self.fig.colorbar(self.heatmap_plot)
        self.ax.set_title('Cumulative Gaze Heatmap')
        self.marker_plots = []  # List to store marker plot objects
        self.kernel = gaussian_kernel(GAZE_KERNEL_FOR_HEATMAP)
        plt.show(block=False)

    def add_markers(self, marker_positions):
        # Remove previous markers
        for plot in self.marker_plots:
            plot.remove()
        self.marker_plots.clear()

        # Plot new markers
        for pos in marker_positions:
            pixel_x = int(pos[0] )
            pixel_y = int(pos[1] )
            plot = self.ax.plot(pixel_x, pixel_y, 'o', color='blue')
            self.marker_plots.extend(plot)

    def update(self, gaze_position, marker_positions=None):
        # Normalize and convert the gaze position to pixel coordinates
        x_bin, y_bin = gaze_position

        # Check bounds to avoid index errors
        x_bin = max(0, min(x_bin, self.bins - 1))
        y_bin = max(0, min(y_bin, self.bins - 1))
        # Accumulate the gaze data
        for i in range(-GAZE_KERNEL_FOR_HEATMAP // 2, GAZE_KERNEL_FOR_HEATMAP // 2 + 1):
            for j in range(-GAZE_KERNEL_FOR_HEATMAP // 2, GAZE_KERNEL_FOR_HEATMAP // 2 + 1):
                if 0 <= x_bin + i < self.bins and 0 <= y_bin + j < self.bins:
                    self.heatmap_data[y_bin + j, x_bin + i] += self.kernel[j + GAZE_KERNEL_FOR_HEATMAP // 2, i + GAZE_KERNEL_FOR_HEATMAP // 2]
        # print(x_bin,y_bin)
        # Update the heatmap plot with new data
        self.heatmap_plot.set_data(self.heatmap_data)
        self.heatmap_plot.set_clim(vmin=0, vmax=np.max(self.heatmap_data))

        # # Add markers to the heatmap if provided
        if marker_positions:
            self.add_markers(marker_positions)

        # Redraw the plot
        self.fig.canvas.draw_idle()
        plt.pause(0.001)


class BlinkPlotter:
    def __init__(self, x_lim=(0, 10), y_lim=(0, 1), title="", x_label="", y_label=""):
        plt.ion()  # Interactive mode on
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(*x_lim)  # Set x-axis limits (last 10 seconds)
        self.ax.set_ylim(*y_lim)  # Set y-axis limits
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.ax.set_title(title)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.blink_events = []  # Store the timestamps of blinks

    def add_blink(self, timestamp):
        # Add the timestamp of the blink to the list
        self.blink_events.append(timestamp)

    def update(self, current_time):
        # Remove blinks older than 10 seconds from the current time
        self.blink_events = [t for t in self.blink_events if current_time - t <= 10]

        # Clear the plot
        self.ax.cla()
        self.ax.set_title(self.title )
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.set_xlim(current_time - 10, current_time)  # Update x-axis to the last 10 seconds
        self.ax.set_ylim(0, 1)

        # Plot each blink as a vertical line
        for t in self.blink_events:
            self.ax.axvline(x=t, color='red', linestyle='-', linewidth=2)

        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


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
    def __init__(self, number_of_frames, plot_blink=True, plot_dilation=True, plot_heatmap=True):
        if plot_blink:
            self.blink_rate_plotter = Plotter((0, number_of_frames/FPS), (0, 2),"Blink Rate over time", "Time", "Blink rate")
            self.blink_plotter = BlinkPlotter(
                x_lim=(0, 10),
                y_lim=(0, 1),
                title="Blinks over time",
                x_label="Time",
                y_label="Blink"
            )
        else:
            self.blink_rate_plotter = None
        if plot_dilation:
            self.dilation_plotter = Plotter((0, number_of_frames/FPS), (0, 100), "Eye Dilation over time", "Time", "Eye Dilation")
        else:
            self.dilation_plotter = None
        if plot_heatmap:
            self.heatmappper = Heatmappper(IMG_SIZE, bins=80)
        else:
            self.heatmappper = None


        self.past_poses = []
        self.gaze_positions = []
        self.bins = 80

    def update(self, translated_rect, fov_center, markers, head_pose, gaze_position, blink_rate, blink, dilation_radius,
               frame_id):
        img = np.ones((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8) * 255
        fields_dict = head_pose
        if self.blink_rate_plotter is not None:
            self.blink_rate_plotter.update(frame_id/FPS, blink_rate)
            if blink:
                self.blink_plotter.add_blink(frame_id/FPS)
            self.blink_plotter.update(frame_id/FPS)

            fields_dict["Blink rate"] = blink_rate
        if self.dilation_plotter is not None:
            self.dilation_plotter.update(frame_id/FPS, dilation_radius)
            fields_dict["Dilation Radius"] = dilation_radius
        if self.heatmappper is not None:


            # Normalize and scale gaze position to heatmap bins
            normalized_gaze_x = int((gaze_position[0] / IMG_SIZE[0]) * (self.bins - 1))
            normalized_gaze_y = int((gaze_position[1]/ IMG_SIZE[1]) * (self.bins - 1))

            # print(f"Normalized Gaze for Heatmap: {normalized_gaze_x}, {normalized_gaze_y}")

            # Normalize marker positions
            heatmap_marker_positions = [(int(marker_center[0] * (self.bins - 1)/IMG_SIZE[0]),int(marker_center[1] * (self.bins - 1)/IMG_SIZE[1])) for id, marker_center, corners in markers]

            # Update heatmap with normalized and scaled gaze position
            self.heatmappper.update([normalized_gaze_x, normalized_gaze_y], heatmap_marker_positions)

        self.visualize_fov(img, translated_rect)
        self.draw_gaze_and_dilation_circle(img, gaze_position, dilation_radius)
        self.draw_markers(img, markers)

        self.write_features(img, fields_dict)
        cv2.imshow("View", img)

    def visualize_fov(self, img, translated_rect):
        self.past_poses.append(translated_rect)
        if len(self.past_poses) > NUMBER_OF_PAST_POSES_TO_VISUALIZE:
            self.past_poses.pop(0)
        # Draw past poses with decreasing opacity
        for i, previous_rect in enumerate(self.past_poses[::-1]):
            alpha = (NUMBER_OF_PAST_POSES_TO_VISUALIZE - i) / NUMBER_OF_PAST_POSES_TO_VISUALIZE  # fades out the further we go back in history
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

    def draw_markers(self, img, markers):
        for marker_id, marker_center, marker in markers:
            # Draw the markers
            for j in range(4):
                start_point = (int(marker[j][0]), int(marker[j][1]))
                end_point = (int(marker[(j + 1) % 4][0]), int(marker[(j + 1) % 4][1]))
                cv2.line(img, start_point, end_point, MARKER_VISIBLE_COLOR, MARKER_THICKNESS)
    def draw_gaze_and_dilation_circle(self, img, gaze_position, eye_diameter):
        eye_dilation_radius = eye_diameter / EYE_DIAMETER_NORMALIZING_FACTOR
        eye_dilation_radius = math.floor(eye_dilation_radius * GAZE_CIRCLE_RADIUS)

        self.gaze_positions.append((gaze_position[0], gaze_position[1]))

        # Draw the rotated gaze circle
        cv2.circle(img, (int(gaze_position[0]), int(gaze_position[1])), GAZE_CIRCLE_RADIUS, GAZE_CIRCLE_COLOR, -1)
        cv2.circle(img, (int(gaze_position[0]), int(gaze_position[1])), eye_dilation_radius, EYE_DILATION_COLOR,
                   EYE_DIAMETER_CIRCLE_THICKNESS)

    def generate_heatmap(self):
        # Convert gaze positions to pixel coordinates
        pixel_x_data = [pos[0] for pos in self.gaze_positions]
        pixel_y_data = [pos[1] for pos in self.gaze_positions]

        # Create a 2D histogram from the gaze data
        heatmap, xedges, yedges = np.histogram2d(pixel_x_data, pixel_y_data, bins=self.bins)

        # Display the heatmap
        plt.imshow(heatmap.T, origin='lower', cmap='hot', extent=[0, IMG_SIZE[0], 0, IMG_SIZE[1]])
        plt.colorbar()
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title('Gaze Heatmap')
        plt.show()


