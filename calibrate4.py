import cv2
import numpy as np
import func_timeout
import os
import time

cap = cv2.VideoCapture(2)

termCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30000, 0.000001)
W = 6
L = 8
objp = np.zeros((W*L, 3), np.float32)
objp[:, :2] = np.mgrid[0:L, 0:W].T.reshape(-1, 2) * (14/16)

last = False

_, img_zero = cap.read()
img_zero = cv2.cvtColor(img_zero, cv2.COLOR_BGR2GRAY)
corners = []
objectPoints = []
while True:
    ret, img_org = cap.read()

    def foo(img):
        global img_org
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret2, corners = cv2.findChessboardCorners(gray, (L, W), None)
        refined_corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11), (-1, -1),
            termCriteria
        )
        cv2.drawChessboardCorners(img_org, (L, W), refined_corners, ret2)
        return refined_corners

    output = None
    try:
        output = func_timeout.func_timeout(0.1, foo, (img_org,))
    except func_timeout.FunctionTimedOut:
        pass

    cv2.imshow("img", img_org)
    key = cv2.waitKey(5) & 0xFF
    if key in (ord("q"), ord("s")):
        if key == ord("q"):
            break
        else:
            if not last and output is not None:
                objectPoints.append(objp)
                corners.append(output)
                last = True
    else:
        last = False

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, corners, img_zero.shape[::-1], None, None)

h, w = img_zero.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
dst = cv2.undistort(img_zero, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = img_zero[y:y+h, x:x+w]
while True:
    if dst is not None:
        cv2.imshow('calib', dst)
        cv2.imshow('uncalib', img_zero)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
