import numpy as np
import cv2 as cv
import json


def remove_distortion_from_image(image, params, show):
    mtx, dist = read_params(params)
    img = cv.imread(image)
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv.undistort(img, mtx, dist, None, new_camera_matrix)
    x1, y1, w1, h1 = roi
    dst = dst[y1:y1 + h1, x1:x1 + w1]
    dst = cv.resize(dst, (w, h))

    if show:
        scale = 0.7
        img_resized = cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        dst_resized = cv.resize(dst, (int(dst.shape[1] * scale), int(dst.shape[0] * scale)))
        images = np.hstack((img_resized, dst_resized))
        cv.imshow('distortion removed', images)
        cv.waitKey(0)
    return dst


def read_params(params_file):
    with open(params_file) as file:
        data = json.load(file)
        mtx = np.array(data['mtx'])
        dist = np.array(data['dist'])
    return mtx, dist


remove_distortion_from_image('s1\\left_146.png', 'calib_params.txt', True)
