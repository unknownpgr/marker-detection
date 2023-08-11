import cv2
import numpy as np

N = 5
K = 6

ALVAR_DICT = {
    "ID_0": '1101111011101010111001110',
    "ID_1": '1101111011101011110100110'  # Be careful when the image is flipped horizontally. Asymmetric markers will not be recognized when image flipped horizontally.
}

_ALVAR_DICT = {}
ROTATIONS = ['A', 'B', 'C', 'D']
for key in ALVAR_DICT:
    matrix = np.array(list(map(int, ALVAR_DICT[key]))).reshape(N, N)
    for i in range(4):
        matrix = np.rot90(matrix)
        rotated_key = key + "_" + ROTATIONS[i]
        _ALVAR_DICT[''.join(map(str, matrix.flatten()))] = rotated_key
ALVAR_DICT = _ALVAR_DICT


def parse_marker(marker):

    thresh = cv2.threshold(
        marker,  # src
        0,  # thresh
        255,  # maxval
        cv2.THRESH_BINARY | cv2.THRESH_OTSU  # type
    )[1]
    cv2.imshow('marker', thresh)

    bits = []
    for i in range(2, N+2):
        for j in range(2, N+2):
            x = j * K
            y = i * K
            bit_value = np.mean(thresh[y:y + K, x:x + K]) > 127
            bit_value = 1 if bit_value else 0
            bits.append(bit_value)

    bits = ''.join(map(str, bits))

    # Check if marker is correct
    if bits in ALVAR_DICT:
        return ALVAR_DICT[bits]
    return None


# Open the device at the ID 0
cap = cv2.VideoCapture(0)

# Decrease frame size for faster processing
cap.set(3, 640)
cap.set(4, 480)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray,  # src
        255,  # maxValue
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # adaptiveMethod
        cv2.THRESH_BINARY,  # thresholdType
        31,  # blockSize
        10  # C
    )

    # Find squares
    contours, hierarchy = cv2.findContours(
        thresh,  # image
        cv2.RETR_TREE,  # mode
        cv2.CHAIN_APPROX_SIMPLE  # method
    )

    # Filter contours by area
    contours = [c for c in contours if cv2.contourArea(c) > 100]

    # Approximate contours to polygons
    contours = [cv2.approxPolyDP(c, 3, True) for c in contours]

    # Filter only squares
    contours = [c for c in contours if len(c) == 4]

    markers = []

    for c in contours:
        # Convert contour to N*K matrix
        marker = np.array([
            [0, 0],
            [0, (N+4)*K],
            [(N+4)*K, (N+4)*K],
            [(N+4)*K, 0]
        ], dtype=np.float32)

        transform = cv2.getPerspectiveTransform(
            np.float32(c),  # src
            marker  # dst
        )

        marker = cv2.warpPerspective(
            gray,  # src
            transform,  # M
            ((N+4)*K, (N+4)*K)  # dsize
        )

        marker = parse_marker(marker)

        markers.append(marker)

    # Draw contours
    for i in range(len(contours)):
        if markers[i] is None:
            continue
        cv2.drawContours(frame, contours, i, (0, 255, 0), 3)
        text_position = tuple(contours[i][0][0])
        cv2.putText(frame, str(markers[i]), text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('preview', frame)

    key = cv2.waitKey(1)

    # Waits for a user input to quit the application
    if key & 0xFF == ord('q'):
        break

    # If s is pressed capture and save image
    if key & 0xFF == ord('s'):
        cv2.imwrite('example_manual.png', frame)
