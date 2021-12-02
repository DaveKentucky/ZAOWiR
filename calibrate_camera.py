import numpy as np
import cv2 as cv
from split_images import split_images
from read_images import read_images
import json
import os


def extract_calibration_parameters(file, show):
    """
    Extracts camera calibration parameters
    :param file: file with info about images to read (created with split_images function)
    :type file: str
    :param show: if the progress should be displayed
    :type show: bool
    :return: json calibration and distortion matrices
    :rtype: tuple
    """
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    obj_p = np.zeros((6*8, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.

    images = read_images(file)

    for name in images[:5]:
        img = cv.imread(name)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (8, 6), None)
        # If found, add object points, image points (after refining them)
        if ret is True:
            obj_points.append(obj_p)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners)
            if show:
                # Draw and display the corners
                cv.drawChessboardCorners(img, (8, 6), corners2, ret)
                cv.imshow('img', img)
                cv.waitKey(0)

    # get calibration parameters
    ret, mtx, dist, r_vectors, t_vectors = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    print(mtx)
    print(mtx.shape)

    mtx_json, dist_json = write_params(mtx, dist)
    if show:
        print(f'camera matrix: {mtx_json}')
        print(f'distortion matrix: {dist_json}')

    cv.destroyAllWindows()
    return mtx_json, dist_json


def write_params(mtx, dist):
    mtx_json = mtx.tolist()
    dist_json = dist.tolist()
    dictionary = {
        'mtx': mtx_json,
        'dist': dist_json
    }
    file = open('calib_params.txt', 'w')
    file.write(json.dumps(dictionary))

    return mtx_json, dist_json


extract_calibration_parameters(os.path.join('indices', 'indices_both_s1.txt'), False)
