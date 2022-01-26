from camera_calibration import CameraCalibration
import os
import cv2 as cv


cc = CameraCalibration((8, 6), 35, 's2')

# cc.remove_distortion_from_images("indices_left_k0.txt", "k0_result", "calibration_params.json")
# cc.remove_distortion_from_image(os.path.join("s1", "left_45.png"), "calibration_params.json")
# cc.calibrate_stereo_camera_system('indices_left_right_k0.txt')

# cc.split_images('s2')
# cc.calibrate_stereo_camera_system('indices_left_right_s2.txt')
# cc.rectify_stereo_camera_system('indices_left_right_s2.txt', 'calibration_params_stereo.json')
# cc.rectify_images('indices_left_right_s2.txt')


img_left = cv.imread('cones/im2.png', 0)
img_right = cv.imread('cones/im6.png', 0)
cc.get_depth_map(img_left, img_right, 0.3)
