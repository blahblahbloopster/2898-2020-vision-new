import time
from math import sqrt

import cv2
import numpy as np
from utils import find_extreme_points, rotate_contours, compute_output_values, VirtualCamera, simplify, getCoords
from EasyContour import EasyContour
import pickle as pkl
from pprint import pprint
import os

TRACING = True  # Enables/disables time tracking
PIPELINE = True  # Enables/disables multiprocessing
USE_FIXED_IMG = True  # Enables/disables using a fixed, rendered image instead of the video
DISPLAY = False  # Enables/disables displaying debug windows

if PIPELINE:
    import multiprocessing
else:
    import multiprocessing.dummy as multiprocessing

HEX_DIMENSIONS = [[0, 17],
                  [39.25, 17],
                  [29.437, 0],
                  [9.812, 0]]
# HEX_DIMENSIONS.reverse()
XCenterOffset = max(HEX_DIMENSIONS, key=lambda x: x[0])[0] / 2
YCenterOffset = max(HEX_DIMENSIONS, key=lambda x: x[1])[1]

for index, d in enumerate(HEX_DIMENSIONS):
    HEX_DIMENSIONS[index] = d[0] - XCenterOffset, d[1]
HEX_DIMENSIONS = EasyContour(HEX_DIMENSIONS)
HEX_DIMENSIONS = HEX_DIMENSIONS.format([["x", "y", 0], ["x", "y", 0]], np.float32)

num = 0 if int(cv2.__version__[0]) >= 4 else 1

STOP = "stop"

if USE_FIXED_IMG:
    with open('imaginary_cam_for_real.pkl', 'rb') as f:
        ret, mtx, dist, rotation_vectors, translation_vectors = pkl.load(f)
else:
    with open('elp_camera.pkl', 'rb') as f:
        ret, mtx, dist, rotation_vectors, translation_vectors = pkl.load(f)

times_dict = {}
times_record = {}
frame_count = 0
colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255))


def time_it(name, starting=True):
    # This is a debugging/tracing function.  You use it by putting it around some code you
    # want to time, like this:
    """
    time_it("solvepnp")
    foo = cv2.solvepnp(arguments blah blah)
    time_it("solvepnp", False)
    """
    # The function will measure the amount of time between the calls and record it.  Processes
    # will send it back to the main process which will report it.
    if not TRACING:
        return
    if starting:
        times_dict[name] = time.monotonic()
    else:
        if name in times_record:
            times_record[name]["total"] += time.monotonic() - times_dict[name]
        else:
            times_record[name] = {"total": time.monotonic() - times_dict[name],
                                  "calls": 1}


class HexFinder:

    def __init__(self, camera):
        self.camera = camera
        self.img_org = None
        # self.tasks = [self.capture_video, self.contours, self.filter, self.corners, self.solvepnp]

    def update(self):
        got_output, img_org = self.camera.read()
        img_org = img_org.copy()
        if not got_output:
            return STOP
        hsv = cv2.cvtColor(img_org, cv2.COLOR_RGB2HSV)
        thresh = cv2.inRange(hsv, (0, 0, 10), (255, 255, 255))
        # thresh = cv2.inRange(self.img_org, (0, 60, 0), (175, 255, 200))
        # size = 12
        # thresh = cv2.dilate(thresh, (size, size), iterations=1)
        # thresh = cv2.erode(thresh, (size, size), iterations=1)
        if DISPLAY:
            undistort = cv2.undistort(img_org, mtx, dist)
            cv2.imshow("undistort", undistort)
            cv2.imshow("thresh", thresh)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if DISPLAY:
            cv2.drawContours(img_org, contours[num], -1, (255, 0, 0))
            cv2.imshow("contours", img_org)
        filtered = []
        for cnt in contours:
            # Checks number of points (rough estimation of size)
            if len(cnt) < 100:
                continue

            # Checks area
            if cv2.contourArea(cnt) < 100:
                print("a")
                continue

            # # Checks perimeter/area ratio
            # if not 0.25 < cv2.arcLength(cnt, True) / cv2.contourArea(cnt) < 0.3:
            #     continue

            # Checks convex hull perimeter / perimeter ratio
            if not 0.6 < cv2.arcLength(cv2.convexHull(cnt), True) / cv2.arcLength(cnt, True) < 0.9:
                print("b")
                continue

            # Checks if it is too close to the edge
            extreme_points = find_extreme_points(cnt)
            if extreme_points[0][0] < 10 or extreme_points[1][0] > 1070:
                continue

            filtered.append(cnt)
        points = []
        for contour in filtered:
            contour = cv2.approxPolyDP(cv2.convexHull(contour), 5, True)
            center = getCoords(contour)
            unordered = list(map(lambda x: tuple(x[0]), sorted(contour, key=lambda x: sqrt(((x[0][0] - center[0]) ** 2) + ((x[0][1] - center[1]) ** 2)), reverse=True)))
            sort = sorted(unordered, key=lambda x: x[1])
            bottom_points = sort[2:]
            top_points = sort[:2]
            top_horizontal = sorted(top_points, key=lambda x: x[0])
            point1 = top_horizontal[-1]
            point2 = top_horizontal[0]
            bottom_horizontal = sorted(bottom_points, key=lambda x: x[0])
            point3 = bottom_horizontal[0]
            point4 = bottom_horizontal[1]
            """
               1_______2
               4\_____/3
            """
            points.append((point1, point2, point3, point4))
        for p in points:
            for index, point in enumerate(p):
                cv2.drawMarker(img_org, point, colors[index])
        subpixels = []
        for corner in points:
            formatted = np.float32(np.reshape(corner, (4, 1, 2)))
            subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            gray = cv2.cvtColor(img_org, cv2.COLOR_RGB2GRAY)
            subpixels.append(cv2.cornerSubPix(gray, formatted, (3, 3), (-1, -1), subpix_criteria))

        angles = []
        for point in subpixels:
            points2 = np.array(point, dtype=np.float32)

            got_output, rotation, translation = cv2.solvePnP(HEX_DIMENSIONS,
                                                             points2, mtx, dist)
            angles.append(compute_output_values(rotation, translation))
            if DISPLAY:
                for index, p in enumerate(point):
                    cv2.drawMarker(img_org, tuple(p[0]), colors[index])
                cv2.aruco.drawAxis(img_org, mtx, dist, rotation, translation, 20)
                for index, q in enumerate(np.squeeze(HEX_DIMENSIONS)):
                    p = q.copy()
                    p = p[0:2]
                    p[0] += 20
                    p[1] += 20
                    cv2.drawMarker(img_org, tuple(p), colors[index])
                # new_points, _ = cv2.projectPoints(HEX_DIMENSIONS, rotation, translation, mtx, dist)
                # for p in new_points:
                #     p = (int(np.ndarray.tolist(p)[0][0]), int(np.ndarray.tolist(p)[0][1]))
                #     cv2.drawMarker(img, p, (0, 0, 255))
        if DISPLAY:
            cv2.imshow("aaa", img_org)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                return STOP
        return angles

    def outside_corners(self, contour):
        return find_extreme_points(contour)

    def process_output(self, vectors):
        processed = []
        for rotation, translation in vectors:
            processed.append(compute_output_values(rotation, translation))
        return processed


finder = HexFinder(VirtualCamera(img=cv2.imread("rendered_images/400in_straight2.png")))
# finder.start()
start = time.time()
reps = 0
starting = time.time()
try:
    while time.time() - starting < 10 or not USE_FIXED_IMG:
        if reps >= 200:
            time_per_frame = (time.time() - start) / reps
            fps = 1 / time_per_frame
            print("Avg time per frame: %f5, avg fps: %f2  (avg over %d samples)" %
                  (time_per_frame, fps, reps))
            reps = 0
            start = time.time()
        reps += 1
        gotten = finder.update()
        print(gotten)
        # if len(gotten) > 0:
        #     print("%f in away, %f, %f" % gotten[0])
finally:
    cv2.destroyAllWindows()
