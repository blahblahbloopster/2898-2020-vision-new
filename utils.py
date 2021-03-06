import cv2
import numpy as np
from math import sin, cos, radians
import math

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
    # try:
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    return leftmost, rightmost, topmost, bottommost
    # except:
    #     print(cnt)


def compute_output_values(rvec, tvec):
    # Compute the necessary output distance and angles

    # The tilt angle only affects the distance and angle1 calcs

    x = tvec[0][0]
    angle = 0
    z = math.sin(math.radians(angle)) * tvec[1][0] + math.cos(math.radians(angle)) * tvec[2][0]

    # distance in the horizontal plane between camera and target
    distance = math.sqrt(x ** 2 + z ** 2)

    # horizontal angle between camera center line and target
    angle1 = math.atan2(x, z)

    rot, _ = cv2.Rodrigues(rvec)
    rot_inv = rot.transpose()
    # This should be pzero_world = numpy.matmul(rot_inv, -tvec)
    B = np.mat(rot_inv)
    C = np.mat(-tvec)

    A = B * C
    pzero_world = A

    angle2 = math.atan2(pzero_world[0][0], pzero_world[2][0])

    return distance, math.degrees(angle1), math.degrees(angle2)


# def compute_output_values(rotation_vec, translation_vec):
#     # Stolen from ligerbots 2019 vision code
#
#     # Compute the necessary output distance and angles
#     x = translation_vec[0][0] + 0
#     z = 0 * translation_vec[1][0] + 1 * translation_vec[2][0]
#
#     # distance in the horizontal plane between robot center and target
#     robot_distance = math.sqrt(x**2 + z**2)
#
#     # horizontal angle between robot center line and target
#     robot_to_target_angle = math.atan2(x, z)
#
#     rot, _ = cv2.Rodrigues(rotation_vec)
#     rot_inv = rot.transpose()
#
#     # version if there is not offset for the camera (VERY slightly faster)
#     # pzero_world = numpy.matmul(rot_inv, -tvec)
#
#     # version if camera is offset
#     pzero_world = np.matmul(rot_inv, 0 - translation_vec)
#
#     other_angle = math.atan2(pzero_world[0][0], pzero_world[2][0])
#
#     return robot_distance, robot_to_target_angle, other_angle


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

    result = np.zeros((len(list_of_points), 1, 3))

    result[:list_of_points.shape[0], :list_of_points.shape[1], :list_of_points.shape[2]] = list_of_points
    rotated_points = np.matmul(result, rotation_matrix)
    rotated_points = np.delete(rotated_points, 2, 2)
    # print(rotated_points)
    # for point in list_of_points:
    #     if type(point) != np.int32:
    #         # Convert point to list
    #         if type(point) != list:
    #             point = np.ndarray.tolist(point)
    #         # Append 1 to point for matrix multiplication
    #         if len(point) == 1:
    #             point = [point[0][0], point[0][1], 1]
    #         else:
    #             point = [point[0], point[1], 1]
    #         # Apply transform
    #         multiplied = np.matmul(point, rotation_matrix)
    #         rotated_points.append([[multiplied[0], multiplied[1]]])

    rotated_points = np.array(rotated_points, dtype=np.int32)

    return rotated_points


def simplify(contour, numPoints, tick=None):
    """Simplifies contour to numPoints total points"""
    if tick is None:
        tick = cv2.arcLength(contour, True) * 0.01
    epsilon = cv2.arcLength(contour, True) * 0.01
    while len(cv2.approxPolyDP(contour, epsilon, True)) > numPoints:
        epsilon += tick
    return cv2.approxPolyDP(contour, epsilon, True)


def getCoords(contour):
    """Returns contour XY coords"""
    M = cv2.moments(contour)
    x = int(M['m10']/M['m00'])
    y = int(M['m01']/M['m00'])
    return (x, y)

class VirtualCamera:

    def __init__(self, camera=None, img=None):
        self.ret, self.img = True, np.zeros((480, 360, 3))
        self.camera = None

        if camera:
            self.camera = camera
            self.ret, self.img = self.camera.read()
        elif img is not None:
            self.ret, self.img = True, img

    def update(self):
        if self.camera:
            self.ret, self.img = self.camera.read()

    def read(self):
        return self.ret, self.img
