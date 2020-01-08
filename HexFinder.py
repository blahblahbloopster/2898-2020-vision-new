import time

import cv2
import numpy as np
from utils import find_extreme_points, rotate_contours, compute_output_values
from EasyContour import EasyContour
import pickle as pkl
import multiprocessing

XCenterOffset = 19.125
YCenterOffset = 8.5
HEX_DIMENSIONS = [[0 - XCenterOffset, 17 - YCenterOffset], [39.25 - XCenterOffset, 17 - YCenterOffset], [29.437 - XCenterOffset, 0 - YCenterOffset], [9.812 - XCenterOffset, 0 - YCenterOffset]]
HEX_DIMENSIONS = EasyContour(HEX_DIMENSIONS)
HEX_DIMENSIONS = HEX_DIMENSIONS.format([["x", "y", 0], ["x", "y", 0]], np.float32)

STOP = "stop"

with open('imaginary_cam.pkl', 'rb') as f:
    ret, mtx, dist, rotation_vectors, translation_vectors = pkl.load(f)


class HexFinder:

    def __init__(self, camera):
        self.camera = camera
        self.img_org = None
        self.queues = [
            multiprocessing.Queue(),  # Input
            multiprocessing.Queue(),  # Thresh -> contours
            multiprocessing.Queue(),  # Contours -> corners
            multiprocessing.Queue(),  # Corners -> subpixel
            multiprocessing.Queue(),  # Subpixel -> solvepnp
            multiprocessing.Queue()   # Solvepnp -> out
        ]
        for i in range(10):
            self.queues[0].put(None)
        self.processes = [
            multiprocessing.Process(target=self.work_function(self.capture_video, self.queues[0], self.queues[1], camera)),
            multiprocessing.Process(target=self.work_function(self.contours, self.queues[1], self.queues[2])),
            multiprocessing.Process(target=self.work_function(self.corners, self.queues[2], self.queues[3])),
            multiprocessing.Process(target=self.work_function(self.subpixel, self.queues[3], self.queues[4])),
            multiprocessing.Process(target=self.work_function(self.solvepnp, self.queues[4], self.queues[5]))
        ]
        for p in self.processes:
            p.start()

    def update(self):
        self.queues[0].put(None)
        return self.queues[-1].get()

    def work_function(self, target, inp_q, out_q, camera=False):
        print("Initalizing")
        if camera:
            self.camera = cv2.VideoCapture(camera)
        gotten = None
        while gotten is not STOP:
            print("getting")
            gotten = inp_q.get()
            print("gotten")
            if gotten is None:
                out_q.put(None)
            output = target(gotten)
            out_q.put(output)
        if gotten is STOP:
            out_q.put(STOP)
            print("Process exiting %s" % target)
            exit()

    def capture_video(self, inp):
        ret, self.img_org = self.camera.read()
        if not ret:
            return STOP
        hsv = cv2.cvtColor(self.img_org, cv2.COLOR_RGB2HSV)
        thresh = cv2.inRange(hsv, (0, 150, 0), (200, 255, 255))
        return thresh

    def subpixel(self, corners):
        corners, img = corners
        subpixels = []
        for corner in corners:
            formatted = np.float32(np.reshape(corner, (4, 1, 2)))
            subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            subpixels.append(cv2.cornerSubPix(gray, formatted, (3, 3), (-1, -1), subpix_criteria))
        return subpixels

    def contours(self, img):
        return cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], img

    def outside_corners(self, contour):
        return find_extreme_points(contour)

    def corners(self, contours):
        img = contours[1]
        contours = contours[0]
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

        return points, img

    def solvepnp(self, points):
        points, img = points
        angles = []
        for point in points:
            points2 = np.array(point, dtype=np.float32)

            ret, rotation, translation = cv2.solvePnP(HEX_DIMENSIONS,
                                                      points2, mtx, dist)
            # cv2.aruco.drawAxis(self.img_org, mtx, dist, rotation, translation, 20)
            # cv2.imshow("axis!", self.img_org)
            angles.append(compute_output_values(rotation, translation))
        return angles, img

    def kill(self):
        for p in self.processes:
            p.kill()

    def stop(self):
        self.queues[0].put(STOP)
        self.queues[0].put(STOP)
        self.queues[0].put(STOP)


finder = HexFinder("output.avi")
while True:
    # print(finder.update())
    print(finder.update())
