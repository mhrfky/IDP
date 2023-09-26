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

    # Compute the apparent height and position of the rectangle on the viewing plane

    fov = 10  # Assuming a field of view of 10 degrees for simplicity
    distance = np.linalg.norm(transformation_matrix[:3, 3]) * 8
    apparent_height = np.tan(np.radians(fov / 2)) * distance * 2
    y_position = transformation_matrix[1, 3]  # y translation of the transformation matrix

    # Create a new figure and set axis limits
    fig, ax = plt.subplots()
    ax.set_xlim(-50, 50)  # Cylinder unfolded in x-axis from -180 to 180 degrees
    ax.set_ylim(-3, 3)  # Cylinder height in the range [-r, r]

    # Draw a rectangle to represent the field of view
    rect = patches.Rectangle((yaw_head - fov / 2, -y_position - apparent_height / 2), fov, apparent_height,
                             color='r')
    ax.add_patch(rect)

    # Display the plot
    plt.xlabel('Yaw (degrees)')
    plt.ylabel('Height')
    plt.title('Field of View on Cylindrical Screen')
    plt.show()


if __name__ == "__main__":
    # Example 4x4 transformation matrix
    T = np.array([[0.98, -0.17, 0.08, 0.5],
                  [0.17, 0.98, 0.03, 0.1],
                  [-0.08, -0.03, 0.99, 0.2],
                  [0, 0, 0, 1]])

    visualize_fov(T)
