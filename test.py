import numpy as np
from scipy.spatial.transform import Rotation as R

matrix = np.array([
    [0.71733174, -0.16687222, 0.08569628, 0.44797837],
    [-0.1671141, 0.89867953, 0.10462531, 0.43330681],
    [-0.08786958, -0.10218067, 0.61646624, 1.5546751],
    [0., 0., 0., 1.]
])

# Extract rotation matrix and translation vector
rotation_matrix = matrix[:3, :3]
translation_vector = matrix[:3, 3]

# Convert rotation matrix to Euler angles
r = R.from_matrix(rotation_matrix)
euler_angles = r.as_euler('xyz', degrees=True)

print("Rotation around X:", euler_angles[0])
print("Rotation around Y:", euler_angles[1])
print("Rotation around Z:", euler_angles[2])
print("Translation vector:", translation_vector)