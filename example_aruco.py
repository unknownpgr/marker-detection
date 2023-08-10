import cv2

# Open the device at the ID 0
cap = cv2.VideoCapture(0)

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, parameters)

# Check whether user selected camera is opened successfully.
if not (cap.isOpened()):
    print("Could not open video device")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    corners, ids, _ = detector.detectMarkers(frame)

    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # Display the resulting frame
    cv2.imshow('preview', frame)

    key = cv2.waitKey(1)

    # Waits for a user input to quit the application
    if key & 0xFF == ord('q'):
        break

    # If s is pressed capture and save image
    if key & 0xFF == ord('s'):
        cv2.imwrite('example_aruco.png', frame)
