import numpy as np
import math
import cv2 as cv


def region_of_interest(img, vertices):
    # Define a black matrix that matсhes the image height and width
    mask = np.zeros_like(img)

    # Create a color of mask
    mask_color = 255

    # Fill the polygon
    cv.fillPoly(mask, vertices, mask_color)

    # Returning the masked image
    masked_image = cv.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[0, 0, 255], thickness=3):
    if lines is None:
        return

    img = np.copy(img)
    
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        np.uint8
    )

    # Draw each line on the blank image
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(line_img, (x1, y1), (x2, y2), color, thickness)

    # Merge drawn lines with the image
    img = cv.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img

# Pipeline = конвейер
def pipeline(img):
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image_blur = cv.GaussianBlur(gray_image, (9, 9), 0)

    height, width = image_blur.shape[0], image_blur.shape[1]

    # Vertices' coordinates of triange
    roi_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]

    # We use the Canny Edge
    edges = cv.Canny(gray_image, 100, 200)

    # Keep roi only
    masked = region_of_interest(edges, np.array([roi_vertices], np.int32))

    # Detect lines using Probabilistic Hough Transform
    lines = cv.HoughLinesP(
        masked,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )

    # Split lines in two categories: left and right
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            # Only consider extreme slope
            if math.fabs(slope) < 0.5:
                continue

            if slope <= 0:
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

    min_y = img.shape[0] * 3 // 5     # Below the horizon
    max_y = img.shape[0]

    # Generate linear function
    poly_left = np.poly1d(np.polyfit(
        left_line_y,
        left_line_x,
        deg=1
    ))

    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))

    poly_right = np.poly1d(np.polyfit(
        right_line_y,
        right_line_x,
        deg=1
    ))

    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    ##########################################

    # Draw lines on the image
    img_with_lines = draw_lines(
        img,
        [[
            [left_x_start, max_y, left_x_end, min_y],
            [right_x_start, max_y, right_x_end, min_y]
        ]],
        thickness=6
    )

    return img_with_lines

# image = cv.imread('test_images/solidYellowCurve.jpg')

cap = cv.VideoCapture('test_videos/challenge.mp4')


while cap.isOpened():
    ret, frame = cap.read()

    processed_frame = pipeline(frame)

    cv.imshow('Video', processed_frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()