import cv2
import numpy as np
from math import sin, cos, radians

cached = [(45,
           [[cos(radians(45)), -sin(radians(45)), 0],
            [sin(radians(45)), cos(radians(45)), 0],
            [0, 0, 1]]),
          (-45,
           [[cos(radians(-45)), -sin(radians(-45)), 0],
            [sin(radians(-45)), cos(radians(-45)), 0],
            [0, 0, 1]],
           )]


def find_extreme_points(cnt):
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    return leftmost, rightmost, topmost, bottommost


def rotate_contours(degrees, list_of_points):
    list_of_points = np.array(list_of_points, dtype=np.float16)
    # Convert to radians
    rads = radians(degrees)
    # Create affine transform matrix
    rotation_matrix = None
    for c in cached:
        if c[0] == degrees:
            rotation_matrix = c[1]
    if not rotation_matrix:
        rotation_matrix = [[cos(rads), -sin(rads), 0],
                           [sin(rads), cos(rads), 0],
                           [0, 0, 1]]
    rotation_matrix = np.array(rotation_matrix)
    rotated_points = []
    for point in list_of_points:
        if type(point) != np.int32:
            # Convert point to list
            if type(point) != list:
                point = np.ndarray.tolist(point)
            # Append 1 to point for matrix multiplication
            if len(point) == 1:
                point = [point[0][0], point[0][1], 1]
            else:
                point = [point[0], point[1], 1]
            # Apply transform
            multiplied = np.matmul(point, rotation_matrix)
            rotated_points.append([[multiplied[0], multiplied[1]]])
    rotated_points = np.array(rotated_points, dtype=np.int32)

    return rotated_points


class VirtualCamera:

    def __init__(self, camera):
        self.camera = camera
        self.ret, self.img = self.camera.read()

    def update(self):
        self.ret, self.img = self.camera.read()

    def read(self):
        return self.ret, self.img
