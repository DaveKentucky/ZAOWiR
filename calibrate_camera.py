from CameraCalibration import CameraCalibration


cc = CameraCalibration((8, 6), 28.67)
cc.calibrate_stereo_camera_system('indices_left_right_s1.txt')
