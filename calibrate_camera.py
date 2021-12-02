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

    for name in images:
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


def calibrate_stereo_camera_system(file, show):
    """
    Calibrates stereo camera system
    :param file: file with info about images to read (created with split_images function)
    :type file: str
    :param show: if the progress should be displayed
    :type show: bool
    :return:
    :rtype:
    """
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    obj_p = np.zeros((6*8, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space
    img_points_l = []  # 2d points in left camera image plane.
    img_points_r = []  # 2d points in right camera image plane.
    img_shape = None

    images = read_images(file)

    for i, name in enumerate(images[::2]):
        img_l = cv.imread(images[2 * i])
        img_r = cv.imread(images[2 * i + 1])
        gray_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
        gray_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)

        ret_l, corners_l = cv.findChessboardCorners(gray_l, (8, 6), None)
        ret_r, corners_r = cv.findChessboardCorners(gray_r, (8, 6), None)
        obj_points.append(obj_p)

        if ret_l is True:
            rt = cv.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            img_points_l.append(rt)

            if show:
                # Draw and display the corners
                ret_l = cv.drawChessboardCorners(img_l, (8, 6), rt, ret_l)
                cv.imshow(images[2 * i], img_l)
                cv.waitKey(0)

        if ret_r is True:
            rt = cv.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            img_points_r.append(rt)

            if show:
                # Draw and display the corners
                ret_r = cv.drawChessboardCorners(img_r, (8, 6), rt, ret_r)
                cv.imshow(images[2 * i + 1], img_r)
                cv.waitKey(0)

        img_shape = gray_l.shape[::-1]

    rt, mtx_l, dst_l, r_l, t_l = cv.calibrateCamera(
        obj_points, img_points_l, img_shape, None, None
    )
    rt, mtx_r, dst_r, r_r, t_r = cv.calibrateCamera(
        obj_points, img_points_r, img_shape, None, None
    )

    stereocalib_criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 1e-5)
    ret, mtx_1, dst_1, mtx_2, dst_2, R, T, E, F = cv.stereoCalibrate(
        obj_points,
        img_points_l,
        img_points_r,
        mtx_l,
        dst_l,
        mtx_r,
        dst_r,
        img_shape,
        criteria=stereocalib_criteria
    )
    print(ret)
    print('Intrinsic_mtx_1', mtx_l)
    print('dist_1', dst_l)
    print('Intrinsic_mtx_2', mtx_r)
    print('dist_2', dst_r)
    print('R', R)
    print('T', T)
    print('E', E)
    print('F', F)
    cv.destroyAllWindows()

    return {
        'mtx_l': mtx_l,
        'dts_l': dst_l,
        'mtx_r': mtx_r,
        'dst_r': dst_r,
        'R': R,
        'T': T,
        'E': E,
        'F': F
    }


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


# extract_calibration_parameters(os.path.join('indices', 'indices_both_s1.txt'), True)
calibrate_stereo_camera_system(os.path.join('indices', 'indices_both_s1.txt'), False)
