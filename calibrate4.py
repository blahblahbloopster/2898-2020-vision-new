import cv2
import numpy as np
import func_timeout
import pickle as pkl

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

termCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30000, 0.000001)
W = 7
L = 7
objp = np.zeros((W*L, 3), np.float32)
objp[:, :2] = np.mgrid[0:L, 0:W].T.reshape(-1, 2) * 1

last = False

_, img_zero = cap.read()
img_zero = cv2.cvtColor(img_zero, cv2.COLOR_BGR2GRAY)
corners = []
objectPoints = []
while True:
    ret, img_org = cap.read()

    cv2.imshow("aaa", img_org)

    gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    ret2, corners2 = cv2.findChessboardCorners(gray, (L, W), None)
    refined_corners = corners2
    if ret2:
        refined_corners = cv2.cornerSubPix(
            gray,
            corners2,
            (11, 11), (-1, -1),
            termCriteria
        )
    cv2.drawChessboardCorners(img_org, (L, W), refined_corners, ret2)

    cv2.imshow("img", img_org)
    key = cv2.waitKey(5) & 0xFF
    if key in (ord("q"), ord("s")):
        if key == ord("q"):
            break
        else:
            if not last and refined_corners is not None:
                objectPoints.append(objp)
                corners.append(refined_corners)
                last = True
    else:
        last = False

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, corners, img_zero.shape[::-1], None, None)

h, w = img_zero.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
dst = cv2.undistort(img_zero, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
while True:
    if dst is not None:
        cv2.imshow('calib', dst)
        cv2.imshow('uncalib', img_zero)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

with open('calibration/elp_camera.pkl', 'wb') as f:
    pkl.dump([ret, mtx, dist, rvecs, tvecs], f)
