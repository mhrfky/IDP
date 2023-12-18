import matplotlib.pyplot as plt
import random

def initialize_plot(x_lim=(0, 5000)):
    """Initializes and returns the figure and line objects for the plot."""
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    line, = ax.plot([], [], '-')  # Initialize an empty line plot
    ax.set_xlim(*x_lim)  # Set x-axis limits
    ax.set_ylim(0, 1)  # Assuming blink rate is normalized between 0 and 1
    return fig, ax, line

def update_plot(fig, ax, line, frame, blink_rate):
    """Updates the plot with new data."""
    # Append new data to the line's data
    x_data, y_data = line.get_data()
    x_data = list(x_data)
    y_data = list(y_data)
    x_data.append(frame)
    y_data.append(blink_rate)

    # Update line data
    line.set_data(x_data, y_data)

    # Redraw the plot
    fig.canvas.draw()
    fig.canvas.flush_events()

# # Example usage
# fig, ax, line = initialize_plot()
#
# for frame in range(5000):
#     # Simulate getting a new blink rate value
#     new_blink_rate = random.random()  # Replace with actual blink rate
#
#     # Update the plot with the new blink rate
#     update_plot(fig, ax, line, frame, new_blink_rate)
