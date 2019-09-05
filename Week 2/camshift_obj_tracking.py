import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)

r, c, w, h =  130, 230, 550, 360
track_wdw = (c, r, w, h)

rect = cv.imread('v3.jpg')
rect = cv.GaussianBlur(rect, (5, 5), 0)

roi = rect[r:r+h, c:c+w]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((70., 60., 32.)), np.array((110., 255., 255.)))
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)


# analyzing hsv palette to extract the most frequecy values

# unique, count = np.unique(hsv_roi, return_counts=True)
# for i in range(len(unique)):
#     print(f'{unique[i]}: {count[i]}, ') 

# setting mouse callback on the image
def click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(f'x: {x}, y: {y}')

# cv.namedWindow('rect')
# cv.setMouseCallback('rect', click)


while(True):
    # capture.read() function returns a bool variable, i.e. is frame read correctly
    _, frame = capture.read()
    
    frame = cv.GaussianBlur(frame, (5, 5), 0)
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    proj = cv.calcBackProject([frame_hsv], [0], roi_hist, [0, 180], scale=1)

    ret, track_window = cv.CamShift(proj, (c, r, w, h), term_crit)

    pts = cv.boxPoints(ret)
    pts = np.int0(pts)
    cv.polylines(frame, [pts], True, (255, 0, 0), 2)

    cv.imshow('Cam', frame)
    cv.imshow('Mask', proj)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

capture.release()
cv.destroyAllWindows()