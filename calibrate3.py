import cv2
import numpy as np
import os
import pickle as pkl


dir = "images/ps3-camera-calib-0/"
# images = np.asarray([img for img in [
#     cv2.cvtColor(
#         cv2.imread(dir+img),
#         cv2.COLOR_BGR2GRAY
#     )
#     for img in os.listdir(dir)
# ]])

images = []
for img in os.listdir(dir):
    images.append(cv2.imread(dir + img, 0))

names = os.listdir(dir)

termCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30000, 0.000001)
W = 7
L = 7
objp = np.zeros((W*L, 3), np.float32)
objp[:, :2] = np.mgrid[0:L, 0:W].T.reshape(-1, 2)

objpoints = []
imgpoints = []

for index, img in enumerate(images):
    ret, corners = cv2.findChessboardCorners(img, (L, W), None)
    if ret:
        objpoints.append(objp)
        refinedCorners = cv2.cornerSubPix(
            img,
            corners,
            (11, 11), (-1, -1),
            termCriteria
        )
        imgpoints.append(refinedCorners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (W, L), refinedCorners, ret)
    print(names[index])

    # while True:
    #    cv2.imshow('img', img)
    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break
    cv2.destroyAllWindows()
# print(imgpoints)
# exit()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, images[0].shape[::-1], None, None)

with open('calibration/ps3_cam3.pkl', 'wb') as f:
    pkl.dump([ret, mtx, dist, rvecs, tvecs], f)

print("mtx")
print(mtx)
print("dist")
print(dist)

h, w = images[0].shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
dst = cv2.undistort(images[0], mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
while True:
    if dst is not None:
        cv2.imshow('calib', dst)
        cv2.imshow('uncalib', images[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
