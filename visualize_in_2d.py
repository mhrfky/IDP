import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def visualize_fov(transformation_matrix):
    """
    Visualize the Field of View on a cylindrical screen based on the transformation matrix.

    :param transformation_matrix: 4x4 numpy array representing the transformation matrix.
    """

    # radius of the cylindrical screen (distance to the user)
    r = 7.5  # Modify as needed

    # Extract yaw (rotation around y-axis in this case) from the transformation matrix
    yaw_head = np.arctan2(transformation_matrix[0, 2], transformation_matrix[0, 0]) * (
                180 / np.pi)  # Convert to degrees

    # Create a new figure and set axis limits
    fig, ax = plt.subplots()
    ax.set_xlim(-180, 180)  # Cylinder unfolded in x-axis from -180 to 180 degrees
    ax.set_ylim(0, 2)  # Cylinder height (in arbitrary units)

    # Draw a rectangle to represent the field of view
    rect = patches.Rectangle((yaw_head - 5, 1 - 0.1), 10, 0.2,
                             color='r')  # 10 degrees FOV, placed at height 1, 0.2 height
    ax.add_patch(rect)

    # Display the plot
    plt.xlabel('Yaw (degrees)')
    plt.ylabel('Height')
    plt.title('Field of View on Cylindrical Screen')
    plt.show()


def __main__():
    # Example 4x4 transformation matrix
    T = np.array([[0.98, -0.17, 0.08, 0.5],
                  [0.17, 0.98, 0.03, 0.1],
                  [-0.08, -0.03, 0.99, 0.2],
                  [0, 0, 0, 1]])

    visualize_fov(T)
