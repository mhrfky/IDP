
import numpy as np
import cv2
def load_aruco_dictionary_from_yaml(filename):
	fs = cv2.FileStorage(filename, cv2.FileStorage_READ)
	if not fs.isOpened():
		print(f"Failed to open {filename}")
		return None

	aruco_dict = cv2.aruco.Dictionary()

	# Use the correct key capitalization from the YAML
	markerSize = int(fs.getNode("markersize").real())
	maxCorrectionBits = int(fs.getNode("maxCorrectionBits").real())
	nmarkers = int(fs.getNode("nmarkers").real())

	# Construct the bytesList from individual marker strings
	byte_arrays = []
	for i in range(nmarkers):
		marker_key = f"marker_{i}"
		byte_string = fs.getNode(marker_key).string()

		# Convert the binary string to uint8 array and add to byte_arrays
		byte_array = np.array(list(map(int, list(byte_string))), dtype=np.uint8)
		byte_arrays.append(byte_array)

	# Convert byte_arrays to a numpy matrix and assign to aruco_dict.bytesList
	aruco_dict.bytesList = np.vstack(byte_arrays)
	aruco_dict.markerSize = markerSize
	aruco_dict.maxCorrectionBits = maxCorrectionBits

	return aruco_dict

filename = "my_custom_dictionary.yml"

arucoDict = load_aruco_dictionary_from_yaml(filename)

tag = np.zeros((200, 200, 1), dtype="uint8")

cv2.aruco.generateImageMarker(arucoDict, 0, 6, tag, 1)

# Save the tag generated

cv2.imshow("ArUCo Tag", tag)
cv2.waitKey(0)
cv2.destroyAllWindows()