import time

import cv2
import numpy as np
from utils import find_extreme_points, rotate_contours, compute_output_values
from EasyContour import EasyContour
import pickle as pkl
import multiprocessing
from pprint import pprint
import os

XCenterOffset = 19.125
YCenterOffset = 8.5
HEX_DIMENSIONS = [[0 - XCenterOffset, 17 - YCenterOffset],
                  [39.25 - XCenterOffset, 17 - YCenterOffset],
                  [29.437 - XCenterOffset, 0 - YCenterOffset],
                  [9.812 - XCenterOffset, 0 - YCenterOffset]]
HEX_DIMENSIONS = EasyContour(HEX_DIMENSIONS)
HEX_DIMENSIONS = HEX_DIMENSIONS.format([["x", "y", 0], ["x", "y", 0]], np.float32)

STOP = "stop"

with open('imaginary_cam.pkl', 'rb') as f:
    ret, mtx, dist, rotation_vectors, translation_vectors = pkl.load(f)


times_dict = {}
times_record = {}
frame_count = 0


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
    # if not TRACING:
    #     return
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
        self.camera = None
        self.img_org = None
        self.tasks = [self.capture_video, self.contours, self.corners, self.solvepnp]
        self.times_q = multiprocessing.Queue()
        self.queues = []
        self.queues.append(multiprocessing.Queue())
        for task in self.tasks:
            self.queues.append(multiprocessing.Queue())
        for i in range(10):
            self.queues[0].put(0)
        self.processes = []
        for index, task in enumerate(self.tasks):
            camera = camera if index == 0 else False
            self.processes.append(multiprocessing.Process(
                target=self.work_function, args=(task, self.queues[index],
                                                 self.queues[index + 1], self.times_q,
                                                 camera)
            ))

    def start(self):
        for p in self.processes:
            p.start()

    def update(self):
        self.queues[0].put(0)
        return self.queues[-1].get()

    def work_function(self, target, inp_q, out_q, times_q, camera=False):
        # print("Initalizing")
        # print("{}, {}".format(str(target), os.getpid()))
        if camera:
            self.camera = cv2.VideoCapture(camera)
        gotten = None
        output = None
        name = str(target) if not camera else "camera"
        while type(gotten) is not str or type(output) is str:
            gotten = inp_q.get()
            if gotten is None:
                out_q.put(None)
                continue
            if type(gotten) is str:
                break
            time_it(name)
            output = target(gotten)
            time_it(name, False)
            out_q.put(output)
        if type(gotten) is str or type(output) is str:
            out_q.put(STOP)
            times_q.put(times_record)
            print("Process exiting %s" % target)
            exit()

    def capture_video(self, inp):
        ret, self.img_org = self.camera.read()
        if not ret:
            return STOP
        # hsv = cv2.cvtColor(self.img_org, cv2.COLOR_RGB2HSV)
        # thresh = cv2.inRange(hsv, (0, 0, 0), (255, 255, 255))
        thresh = cv2.inRange(self.img_org, (0, 150, 0), (100, 255, 100))
        # print(thresh)
        return thresh

    def subpixel(self, corners):
        # Not being used at the moment
        corners, img = corners
        subpixels = []
        for corner in corners:
            formatted = np.float32(np.reshape(corner, (4, 1, 2)))
            subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            subpixels.append(cv2.cornerSubPix(gray, formatted, (3, 3), (-1, -1), subpix_criteria))
        return subpixels

    def contours(self, img):
        num = 0 if int(cv2.__version__[0]) >= 4 else 1
        return cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[num]

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
            point3 = tuple(rotate_contours(-45, np.array([[point3]]))[0][0])

            rotated = rotate_contours(-45, contour)
            point4 = find_extreme_points(rotated)[0]
            point4 = tuple(rotate_contours(45, np.array([[point4]]))[0][0])

            points.append((point1, point2, point3, point4))

        return points

    def solvepnp(self, points):
        angles = []
        for point in points:
            points2 = np.array(point, dtype=np.float32)

            got_output, rotation, translation = cv2.solvePnP(HEX_DIMENSIONS,
                                                      points2, mtx, dist)
            angles.append(compute_output_values(rotation, translation))
        return angles

    def kill(self):
        for p in self.processes:
            p.kill()
        times = {}
        for t in range(self.times_q.qsize()):
            times.update(self.times_q.get())
        pprint(times)

    def stop(self):
        self.queues[0].put(STOP)
        self.queues[0].put(STOP)
        self.queues[0].put(STOP)


finder = HexFinder("output.avi")
finder.start()
start = time.time()
reps = 0
print("Main pid: %s" % os.getpid())
while True:
    if reps >= 200:
        time_per_frame = (time.time() - start) / reps
        fps = 1 / time_per_frame
        print("Avg time per frame: %f5, avg fps: %f2  (avg over %d samples)" %
              (time_per_frame, fps, reps))
        reps = 0
        start = time.time()
    reps += 1
    gotten = finder.update()
    if gotten == STOP:
        break
time.sleep(2)
finder.kill()
