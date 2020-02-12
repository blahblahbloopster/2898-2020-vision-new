import time

import cv2
import numpy as np
from utils import find_extreme_points, rotate_contours, compute_output_values, VirtualCamera
from EasyContour import EasyContour
import pickle as pkl
from pprint import pprint
import os

TRACING = True  # Enables/disables time tracking
PIPELINE = True  # Enables/disables multiprocessing
USE_FIXED_IMG = True  # Enables/disables using a fixed, rendered image instead of the video
DISPLAY = True  # Enables/disables displaying debug windows

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
print(HEX_DIMENSIONS)
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
        self.camera = None
        self.img_org = None
        self.tasks = [self.capture_video, self.contours, self.filter, self.corners, self.solvepnp]
        self.times_q = multiprocessing.Queue()
        self.queues = []
        self.queues.append(multiprocessing.Queue(maxsize=20))
        for task in self.tasks:
            self.queues.append(multiprocessing.Queue(maxsize=5))
        for i in range(5):
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
        if camera:
            if USE_FIXED_IMG:
                # img = cv2.imread("test_img4.png")
                # print(img.shape)
                self.camera = VirtualCamera(img=cv2.imread("rendered_images/400in10d10dup.png"))
            else:
                self.camera = cv2.VideoCapture(camera)
        output = None
        name = str(target) if not camera else "camera"
        while True:
            gotten_from_queue = inp_q.get()
            if gotten_from_queue is None:
                out_q.put(None)
                continue
            if type(gotten_from_queue) is str:
                break
            time_it(name)
            output = target(gotten_from_queue)
            time_it(name, False)
            out_q.put(output)
            if type(output) is str:
                break
        if type(gotten_from_queue) is str or type(output) is str:
            out_q.put(STOP)
            times_q.put(times_record)
            print("Process exiting %s" % name)
            exit()

    def capture_video(self, inp):
        got_output, self.img_org = self.camera.read()
        if not got_output:
            return STOP
        hsv = cv2.cvtColor(self.img_org, cv2.COLOR_RGB2HSV)
        thresh = cv2.inRange(hsv, (0, 0, 10), (255, 255, 255))
        # thresh = cv2.inRange(self.img_org, (0, 60, 0), (175, 255, 200))
        # size = 12
        # thresh = cv2.dilate(thresh, (size, size), iterations=1)
        # thresh = cv2.erode(thresh, (size, size), iterations=1)
        if DISPLAY:
            undistort = cv2.undistort(self.img_org, mtx, dist)
            cv2.imshow("undistort", undistort)
            cv2.imshow("thresh", thresh)
            undistort = cv2.undistort(self.img_org, mtx, dist)
            cv2.imshow("undistort", undistort)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                return STOP
        # undistort = cv2.undistort(thresh, mtx, dist)

        return thresh

    def filter(self, contours):
        filtered = []
        for cnt in contours:
            # Checks number of points (rough estimation of size)
            if len(cnt) < 100:
                continue

            # Checks area
            if cv2.contourArea(cnt) < 100:
                continue

            # # Checks perimeter/area ratio
            # if not 0.25 < cv2.arcLength(cnt, True) / cv2.contourArea(cnt) < 0.3:
            #     continue

            # Checks convex hull perimeter / perimeter ratio
            if not 0.6 < cv2.arcLength(cv2.convexHull(cnt), True) / cv2.arcLength(cnt, True) < 0.9:
                continue

            # Checks if it is too close to the edge
            extreme_points = find_extreme_points(cnt)
            if extreme_points[0][0] < 10 or extreme_points[1][0] > 1070:
                continue

            filtered.append(cnt)
        # if len(filtered) == 0:
        #     print("All contours filtered out")
        return filtered

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
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if DISPLAY:
            img = np.zeros((720, 1080, 3))
            cv2.drawContours(img, contours[num], -1, (255, 0, 0))
            cv2.imshow("contours", img)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                return STOP

        return contours[num]

    def outside_corners(self, contour):
        return find_extreme_points(contour)

    def corners(self, contours):
        points = []
        for contour in contours:
            corners = self.outside_corners(contour)
            if not corners:
                return None
            point1, point2, _, _ = corners

            reverse = -1.05

            rotated = rotate_contours(45 * reverse, contour)
            point3 = find_extreme_points(rotated)[0]
            point3 = tuple(rotate_contours(-45 * reverse, np.array([[point3]]))[0][0])

            rotated = rotate_contours(-45 * reverse, contour)
            point4 = find_extreme_points(rotated)[1]
            point4 = tuple(rotate_contours(45 * reverse, np.array([[point4]]))[0][0])

            points.append((point1, point2, point4, point3))
        img = np.zeros((720, 1080, 3))
        for p in points:
            for index, point in enumerate(p):
                cv2.drawMarker(img, point, colors[index])
        cv2.drawContours(img, contours, -1, (0, 0, 255))
        cv2.imshow("corners", img)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            return STOP

        return points

    def solvepnp(self, points):
        angles = []
        for point in points:
            points2 = np.array(point, dtype=np.float32)

            got_output, rotation, translation = cv2.solvePnP(HEX_DIMENSIONS,
                                                             points2, mtx, dist)
            angles.append(compute_output_values(rotation, translation))
            if DISPLAY:
                img = np.zeros((720, 1080, 3))
                for index, p in enumerate(point):
                    cv2.drawMarker(img, p, colors[index])
                cv2.aruco.drawAxis(img, mtx, dist, rotation, translation, 20)
                for index, q in enumerate(np.squeeze(HEX_DIMENSIONS)):
                    p = q.copy()
                    p = p[0:2]
                    p[0] += 20
                    p[1] += 20
                    cv2.drawMarker(img, tuple(p), colors[index])
                # new_points, _ = cv2.projectPoints(HEX_DIMENSIONS, rotation, translation, mtx, dist)
                # for p in new_points:
                #     p = (int(np.ndarray.tolist(p)[0][0]), int(np.ndarray.tolist(p)[0][1]))
                #     cv2.drawMarker(img, p, (0, 0, 255))
                cv2.imshow("axis", img)
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    return STOP
        return angles

    def process_output(self, vectors):
        processed = []
        for rotation, translation in vectors:
            processed.append(compute_output_values(rotation, translation))
        return processed

    def kill(self):
        for p in self.processes:
            try:
                p.kill()
            except AttributeError:
                p.terminate()
        times = {}
        for t in range(self.times_q.qsize()):
            times.update(self.times_q.get())
        pprint(times)

    def stop(self):
        self.queues[0].put(STOP)
        self.queues[0].put(STOP)
        self.queues[0].put(STOP)


finder = HexFinder(3)
finder.start()
start = time.time()
reps = 0
starting = time.time()
print("Main pid: %s" % os.getpid())
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
        if gotten == STOP:
            break
finally:
    finder.stop()
    cv2.destroyAllWindows()
    time.sleep(0.25)
    finder.kill()
