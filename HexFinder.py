import time

import cv2
import numpy as np
from utils import find_extreme_points, rotate_contours, compute_output_values
from EasyContour import EasyContour
import pickle as pkl

XCenterOffset = 19.125
YCenterOffset = 8.5
HEX_DIMENSIONS = [[0 - XCenterOffset, 17 - YCenterOffset], [39.25 - XCenterOffset, 17 - YCenterOffset], [29.437 - XCenterOffset, 0 - YCenterOffset], [9.812 - XCenterOffset, 0 - YCenterOffset]]
HEX_DIMENSIONS = EasyContour(HEX_DIMENSIONS)
HEX_DIMENSIONS = HEX_DIMENSIONS.format([["x", "y", 0], ["x", "y", 0]], np.float32)

with open('imaginary_cam.pkl', 'rb') as f:
    ret, mtx, dist, rotation_vectors, translation_vectors = pkl.load(f)


class HexFinder:

    def __init__(self, camera):
        self.camera = camera
        self.img_org = None

    def update(self):
        contours = self.contours(self.capture_video())
        corners = self.corners(contours)
        self.subpixel(corners)
        translation = self.solvepnp(corners)
        return translation

    def capture_video(self):
        ret, self.img_org = self.camera.read()
        if not ret:
            return None
        hsv = cv2.cvtColor(self.img_org, cv2.COLOR_RGB2HSV)
        thresh = cv2.inRange(hsv, (0, 150, 0), (200, 255, 255))
        return thresh

    def subpixel(self, corners):
        subpixels = []
        for corner in corners:
            formatted = np.float32(np.reshape(corner, (4, 1, 2)))
            subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            gray = cv2.cvtColor(self.img_org, cv2.COLOR_RGB2GRAY)
            subpixels.append(cv2.cornerSubPix(gray, formatted, (3, 3), (-1, -1), subpix_criteria))
        return subpixels

    def contours(self, img):
        return cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    def outside_corners(self, contour):
        return find_extreme_points(contour)

    def corners(self, contours):
        points = []
        for contour in contours:
            corners = self.outside_corners(contour)
            if not corners:
                return None
            point1, point2, _, _ = corners
            rotated = rotate_contours(45, contour)
            point3 = find_extreme_points(rotated)[1]
            point3 = tuple(rotate_contours(-45, np.array([point3]))[0][0])

            rotated = rotate_contours(-45, contour)
            point4 = find_extreme_points(rotated)[0]
            point4 = tuple(rotate_contours(45, np.array([point4]))[0][0])

            points.append((point1, point2, point3, point4))

        return points

    def solvepnp(self, points):
        angles = []
        for point in points:
            points2 = np.array(point, dtype=np.float32)

            ret, rotation, translation = cv2.solvePnP(HEX_DIMENSIONS,
                                                      points2, mtx, dist)
            # cv2.aruco.drawAxis(self.img_org, mtx, dist, rotation, translation, 20)
            # cv2.imshow("axis!", self.img_org)
            angles.append(compute_output_values(rotation, translation))
        return angles


start = time.time()
reps = 0
finder = HexFinder(cv2.VideoCapture("output.avi"))
while True:
    # print(finder.update())
    if reps >= 200:
        time_per_frame = (time.time() - start) / reps
        fps = 1 / time_per_frame
        print("Avg time per frame: %f5, avg fps: %f2  (avg over %d samples)" %
              (time_per_frame, fps, reps))
        reps = 0
        start = time.time()
    reps += 1
    finder.update()
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break
