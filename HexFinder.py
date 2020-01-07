import cv2
import numpy as np
from utils import find_extreme_points, rotate_contours
from EasyContour import EasyContour
import pickle as pkl

HEX_DIMENSIONS = [[0, 17], [39.25, 17], [29.437, 0], [9.812, 0]]
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
        corners = []
        for cnt in contours:
            corners.append(self.corners(cnt))
        translation = []
        for points in corners:
            translation.append(self.solvepnp(points))

    def capture_video(self):
        ret, self.img_org = self.camera.read()
        if not ret:
            return None
        hsv = cv2.cvtColor(self.img_org, cv2.COLOR_RGB2HSV)
        thresh = cv2.inRange(hsv, (0, 150, 0), (200, 255, 255))
        cv2.imshow("thresh", thresh)
        return thresh

    def contours(self, img):
        return cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    def outside_corners(self, contour):
        return find_extreme_points(contour)

    def corners(self, contour):
        # corner1, corner2 = self.outside_corners(contour)
        # print(corner1)
        corners = self.outside_corners(contour)
        if not corners:
            return None
        point1, point2, _, _ = corners
        # cv2.drawMarker(self.img_org, point1, (255, 0, 0))
        # cv2.drawMarker(self.img_org, point2, (0, 255, 0))
        # contour2 = contour.copy()
        # cv2.drawContours(self.img_org, rotate_contours(45, contour), -1, (255, 0, 0))
        rotated = rotate_contours(45, contour)
        point3 = find_extreme_points(rotated)[1]
        point3 = tuple(rotate_contours(-45, np.array([point3]))[0][0])

        rotated = rotate_contours(-45, contour)
        point4 = find_extreme_points(rotated)[0]
        point4 = tuple(rotate_contours(45, np.array([point4]))[0][0])

        # cv2.drawMarker(self.img_org, point3, (0, 0, 255))
        # cv2.drawMarker(self.img_org, point4, (255, 0, 255))

        cv2.imshow("corners", self.img_org)

        return point1, point2, point3, point4

    def solvepnp(self, points):
        points2 = np.array(points, dtype=np.float32)

        ret, translation, rotation = cv2.solvePnP(HEX_DIMENSIONS,
                                                  points2, mtx, dist)
        cv2.aruco.drawAxis(self.img_org, mtx, dist, rotation, translation, 20)
        # cv2.drawMarker(self.img_org, points[3], (255, 0, 0))
        # undistorted = cv2.undistort(self.img_org, mtx, dist)
        cv2.imshow("axis!", self.img_org)
        # cv2.imshow("undistort", undistorted)


finder = HexFinder(cv2.VideoCapture("output.avi"))
while True:
    # print(finder.update())
    finder.update()
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break
