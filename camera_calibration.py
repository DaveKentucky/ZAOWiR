import numpy as np
import cv2 as cv
from file_utils import write_indices_to_file as write_indices, \
    write_calibration_params_to_file as write_params, \
    read_images, \
    read_calibration_params_from_file as read_params
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
        self.img_points_l = []  # 2d points in image plane for left camera image
        self.img_points_r = []  # 2d points in image plane for right camera image
        self.single_camera_params = {}
        self.stereo_camera_params = {}

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
        :return: dictionary with calibration and distortion matrices in JSON format
        :rtype: dict
        """
        # make sure class containers are empty
        self.obj_points = []
        self.img_points = []

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
            self.single_camera_params = {
                "mtx": mtx,
                "dist": dist
            }
            params_json = write_params(self.single_camera_params, 'calibration_params.json')

            print(f'camera matrix: {params_json["mtx"]}')
            print(f'distortion matrix: {params_json["dist"]}')

            cv.destroyAllWindows()
            return params_json
        except NameError:
            print('no images found to calibrate the camera')
            return {}

    def calibrate_stereo_camera_system(self, file, show=False):
        """
        Calibrates stereo camera system
        :param file: file with info about images to read (created with split_images function)
        :type file: str
        :param show: if the progress should be displayed
        :type show: bool
        :return: dictionary with calibration params in JSON format
        :rtype: dict
        """
        self.obj_points = []    # make sure obj_points list is empty
        img_shape = None

        images = read_images(self._indices_folder, file)

        for i, name in enumerate(images[::2]):
            try:
                print(f'searching chessboard in images with index {i}...')
                img_l = cv.imread(images[2 * i])
                img_r = cv.imread(images[2 * i + 1])
                gray_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
                gray_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)

                ret_l, corners_l = cv.findChessboardCorners(gray_l, self.chessboard_size, None)
                ret_r, corners_r = cv.findChessboardCorners(gray_r, self.chessboard_size, None)
                self.obj_points.append(self._obj_p)

                if ret_l is True:
                    rt = cv.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), self.criteria)
                    self.img_points_l.append(rt)

                    if show:
                        # Draw and display the corners
                        cv.drawChessboardCorners(img_l, self.chessboard_size, rt, ret_l)
                        cv.imshow(images[2 * i], img_l)
                        cv.waitKey(0)

                if ret_r is True:
                    rt = cv.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), self.criteria)
                    self.img_points_r.append(rt)

                    if show:
                        # Draw and display the corners
                        cv.drawChessboardCorners(img_r, self.chessboard_size, rt, ret_r)
                        cv.imshow(images[2 * i + 1], img_r)
                        cv.waitKey(0)

                img_shape = gray_l.shape[::-1]
            except cv.error:
                print(f'OpenCV error occurred while searching chessboard - skipping image {name}')

        print('calibrating left camera...')
        rt, mtx_l, dst_l, r_l, t_l = cv.calibrateCamera(
            self.obj_points, self.img_points_l, img_shape, None, None
        )
        print('calibrating right camera...')
        rt, mtx_r, dst_r, r_r, t_r = cv.calibrateCamera(
            self.obj_points, self.img_points_r, img_shape, None, None
        )

        print('calibrating stereo camera system...')
        stereo_calibration_criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, mtx_1, dst_1, mtx_2, dst_2, R, T, E, F = cv.stereoCalibrate(
            self.obj_points,
            self.img_points_l,
            self.img_points_r,
            mtx_l,
            dst_l,
            mtx_r,
            dst_r,
            img_shape,
            criteria=stereo_calibration_criteria
        )
        self.stereo_camera_params = {
            'mtx_l': mtx_l,
            'dts_l': dst_l,
            'mtx_r': mtx_r,
            'dst_r': dst_r,
            'R': R,
            'T': T,
            'E': E,
            'F': F
        }
        params_json = write_params(self.stereo_camera_params, 'calibration_params_stereo.json')

        print('Calibration error', ret)
        print('Intrinsic_mtx_1', params_json["mtx_l"])
        print('dist_1', params_json["dst_l"])
        print('Intrinsic_mtx_2', params_json["mtx_r"])
        print('dist_2', params_json["dst_r"])
        print('R', params_json["R"])
        print('T', params_json["T"])
        print('E', params_json["E"])
        print('F', params_json["F"])

        cv.destroyAllWindows()
        return params_json

    def remove_distortion_from_image(self, image, params_file=None, show=False):
        """
        Removes distortion from image with given parameters
        :param image: path to image to remove distortion from
        :type image: str
        :param params_file: path to file with camera matrix and distortion matrix or None if saved params should be used
        :type params_file: str
        :param show: if the progress should be displayed
        :type show: bool
        :return: image with distortion removed
        :rtype: np.ndarray
        """
        params = read_params(params_file) if params_file else self.single_camera_params     # read camera params
        img = cv.imread(image)  # read image
        # refine the camera matrix with given calibration params
        h, w = img.shape[:2]
        camera_matrix, roi = cv.getOptimalNewCameraMatrix(params["mtx"], params["dist"], (w, h), 1, (w, h))
        # remove distortion from image
        dst = cv.undistort(img, params["mtx"], params["dist"], None, camera_matrix)
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
