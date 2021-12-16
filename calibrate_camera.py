from camera_calibration import CameraCalibration
import os


cc = CameraCalibration((8, 6), 35)

# cc.remove_distortion_from_images("indices_left_k0.txt", "k0_result", "calibration_params.json")
# cc.remove_distortion_from_image(os.path.join("s1", "left_45.png"), "calibration_params.json")
cc.calibrate_stereo_camera_system('indices_left_right_k0.txt')
