import cv2
import numpy as np

# Parameters
marker_bits = 6  # for a 6x6 marker; adjust as needed
marker_count = 100  # number of markers you want in the custom dictionary

# Generate new dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# At this point, 'aruco_dict' contains randomly generated markers.
custom_marker_id = 0  # ID for your custom marker within the dictionary
custom_marker_bits = np.array([[1, 1, 1, 1, 1, 0],
                               [1, 0, 1, 0, 0, 1],
                               [0, 1, 0, 1, 1, 0],
                               [1, 0, 0, 0, 1, 0],
                               [1, 1, 1, 1, 0, 1],
                               [1, 1, 1, 0, 1, 0]], dtype=np.uint8)
print(aruco_dict.bytesList[custom_marker_id])
print(cv2.aruco.Dictionary_getByteListFromBits(custom_marker_bits))

aruco_dict.bytesList[custom_marker_id] = cv2.aruco.Dictionary_getByteListFromBits(custom_marker_bits)
print(aruco_dict.bytesList[custom_marker_id])
# At this point, 'aruco_dict' contains randomly generated markers.
custom_marker_id = 1  # ID for your custom marker within the dictionary
custom_marker_bits = np.array([[1, 1, 1, 0, 1, 1],
                               [1, 1, 1, 0, 0, 1],
                               [1, 0, 1, 0, 1, 1],
                               [0, 1, 0, 0, 0, 1],
                               [1, 1, 1, 0, 0, 0],
                               [1, 0, 1, 0, 1, 1]], dtype=np.uint8)

aruco_dict.bytesList[custom_marker_id] = cv2.aruco.Dictionary_getByteListFromBits(custom_marker_bits)

# At this point, 'aruco_dict' contains randomly generated markers.
custom_marker_id = 2  # ID for your custom marker within the dictionary
custom_marker_bits = np.array([[1, 1, 0, 1, 1, 1],
                               [0, 1, 0, 1, 0, 0],
                               [1, 1, 1, 0, 0, 0],
                               [1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 1],
                               [0, 1, 0, 0, 1, 0]], dtype=np.uint8)

aruco_dict.bytesList[custom_marker_id] = cv2.aruco.Dictionary_getByteListFromBits(custom_marker_bits)

# At this point, 'aruco_dict' contains randomly generated markers.
custom_marker_id = 3  # ID for your custom marker within the dictionary
custom_marker_bits = np.array([[1, 1, 0, 1, 1, 0],
                               [0, 1, 1, 0, 1, 0],
                               [0, 1, 0, 1, 0, 0],
                               [1, 0, 1, 1, 1, 1],
                               [1, 0, 1, 1, 1, 0],
                               [0, 0, 1, 1, 0, 1]], dtype=np.uint8)

aruco_dict.bytesList[custom_marker_id] = cv2.aruco.Dictionary_getByteListFromBits(custom_marker_bits)

# Save dictionary to a file for later use
# ... your code to create the aruco_dict ...

filename = "my_custom_dictionarya.yml"
fs = cv2.FileStorage(filename, cv2.FileStorage_WRITE)

if fs.isOpened():
    aruco_dict.writeDictionary(fs)
    fs.release()
else:
    print(f"Error: Could not open {filename} for writing!")