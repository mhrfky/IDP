import cv2
import numpy as np


def detect_aruco_and_show_warped(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load the predefined dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters_create()

    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    # If some markers are found
    if markerIds is not None:
        # Draw detected markers on the original image
        frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), markerCorners, markerIds)

        # Warp each marker's content
        for corner in markerCorners:
            # Define the points to warp from and to
            pts_src = np.array(corner[0], dtype=float)
            pts_dst = np.array([[0, 0], [299, 0], [299, 299], [0, 299]], dtype=float)

            # Compute the homography matrix
            matrix, _ = cv2.findHomography(pts_src, pts_dst)

            # Warp the region using the homography matrix
            warped = cv2.warpPerspective(frame, matrix, (300, 300))

            # Show the warped content in a new window
            cv2.imshow("Warped", warped)

        return frame_markers

    return frame


# Capture from the default camera
video_path = 'MarkerMovie.MP4'

# Initialize video capture with video file
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect ArUco markers and show the warped content
    displayed_frame = detect_aruco_and_show_warped(frame)
    cv2.imshow('Detected ArUco', displayed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()