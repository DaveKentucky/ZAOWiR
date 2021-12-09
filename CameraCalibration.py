import numpy as np
import cv2 as cv
from file_utils import write_indices_to_file as write_indices, \
    write_calibration_params_to_file as write_params, \
    read_images
import os
import glob


class CameraCalibration:
    def __init__(self, chessboard_size, field_size):
        self._indices_folder = 'indices'
        self.chessboard_size = chessboard_size
        w = chessboard_size[0]
        h = chessboard_size[1]
        # termination criteria
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self._obj_p = np.zeros((w * h, 3), np.float32)
        self._obj_p[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        self._obj_p *= field_size
        # Arrays to store object points and image points from all the images.
        self.obj_points = []  # 3d point in real world space
        self.img_points = []  # 2d points in image plane.

    def split_images(self, folder, show=False):
        """
        Splits PNG images from given folder based on criteria
        if calibration board visible on left, right or both cameras
        :param folder: path to folder where the calibration images are stored
        :type folder: str
        :param show: if the progress should be displayed
        :type show: bool
            """
        images = glob.glob(f'{folder}/*.png')
        # dictionary storing indices of images with pattern visible in left, right or both cameras
        indices = {
            "left": [],
            "right": [],
            "left right": []
        }
        # save indices of images where the pattern is visible in left or right camera in proper lists
        for name in images:
            try:
                print(f'searching chessboard in \'{name}\'...')
                img = cv.imread(name)
                ret, corners = cv.findChessboardCorners(img, self.chessboard_size, None)
                if ret is True:
                    ind = name.find('_') + 1
                    img_index = int(name[ind:-4])
                    if name.find('left') >= 0:
                        indices["left"].append(img_index)
                    elif name.find("right") >= 0:
                        indices["right"].append(img_index)
            except cv.error:
                print(f'OpenCV error occurred while searching chessboard - skipping image {name}')

        # extract images where the pattern is visible in both cameras to separate list
        for left_index in indices["left"]:
            try:
                indices["right"].index(left_index)
            except ValueError:
                continue
            indices["left right"].append(left_index)
            indices["left"].remove(left_index)
            indices["right"].remove(left_index)

        # save indices to files
        if not os.path.exists(self._indices_folder):
            os.makedirs(self._indices_folder)     # create dir if it does not exist yet
        for key in indices:
            print(f'writing indices for {key} to file...')
            write_indices(folder, self._indices_folder, key, indices[key])

        if show:
            print(f'both cameras: {indices["left right"]}')
            print(f'just left camera: {indices["left"]}')
            print(f'just right camera: {indices["right"]}')
        return

    def calibrate_single_camera(self, indices_file, show=False):
        """
        Extracts camera calibration parameters
        :param indices_file: file with info about images to read (created with split_images function)
        :type indices_file: str
        :param show: if the progress should be displayed
        :type show: bool
        :return: json calibration and distortion matrices
        :rtype: tuple
        """
        print('reading images from file...')
        images = read_images(self._indices_folder, indices_file)
        for name in images:
            try:
                print(f'searching chessboard in \'{name}\'...')
                img = cv.imread(name)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                # Find the chess board corners
                ret, corners = cv.findChessboardCorners(gray, self.chessboard_size, None)
                # If found, add object points, image points (after refining them)
                if ret is True:
                    self.obj_points.append(self._obj_p)
                    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                    self.img_points.append(corners)

                    if show:
                        # Draw and display the corners
                        cv.drawChessboardCorners(img, (8, 6), corners2, ret)
                        cv.imshow('img', img)
                        cv.waitKey(0)
            except cv.error:
                print(f'OpenCV error occurred while searching chessboard - skipping image {name}')

        # get calibration parameters
        print('calibrating camera...')
        try:
            ret, mtx, dist, r_vectors, t_vectors = cv.calibrateCamera(
                self.obj_points, self.img_points, gray.shape[::-1], None, None
            )
            mtx_json, dist_json = write_params(mtx, dist)
            if show:
                print(f'camera matrix: {mtx_json}')
                print(f'distortion matrix: {dist_json}')

            cv.destroyAllWindows()
            return mtx_json, dist_json
        except NameError:
            print('no images found to calibrate the camera')
            return


def calibrate_stereo_camera_system(file, show=False):
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
    obj_p *= 28.67
    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space
    img_points_l = []  # 2d points in left camera image plane.
    img_points_r = []  # 2d points in right camera image plane.
    img_shape = None

    images = read_images('indices', file)

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