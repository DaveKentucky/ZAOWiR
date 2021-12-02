import cv2 as cv
import glob
import os


def split_images(folder):
    """
    Splits PNG images from given folder based on criteria
    if calibration board visible on left, right or both cameras
    :param folder: path to folder where the calibration images are stored
    :type folder: str
    """
    images = glob.glob(f'{folder}/*.png')
    images_left = []
    images_right = []
    indices_left = []
    indices_right = []
    indices_both = []

    for name in images:
        img = cv.imread(name)
        ret, corners = cv.findChessboardCorners(img, (8, 6), None)
        if ret is True:
            ind = name.find('_') + 1
            img_index = int(name[ind:-4])
            if name.find('left') >= 0:
                images_left.append(name)
                indices_left.append(img_index)
            elif name.find('right') >= 0:
                images_right.append(name)
                indices_right.append(img_index)

    for left_index in indices_left:
        try:
            right_index = indices_right.index(left_index)
        except ValueError:
            continue
        indices_both.append(left_index)
        indices_left.remove(left_index)
        indices_right.remove(left_index)

    folder_path = 'indices'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if len(indices_left) > 0:
        file = open(os.path.join(folder_path, f'indices_left_only_{folder}.txt'), 'w')
        file.write(f'{folder}\n')
        file.write('left\n')
        for index in indices_left:
            file.write(str(index) + '\n')
        file.close()
    if len(indices_right) > 0:
        file = open(os.path.join(folder_path, f'indices_right_only_{folder}.txt'), 'w')
        file.write(f'{folder}\n')
        file.write('left\n')
        for index in indices_right:
            file.write(str(index) + '\n')
        file.close()
    if len(indices_both) > 0:
        file = open(os.path.join(folder_path, f'indices_both_{folder}.txt'), 'w')
        file.write(f'{folder}\n')
        file.write('left right\n')
        for index in indices_both:
            file.write(str(index) + '\n')
        file.close()

    print(f'both cameras: {indices_both}')
    print(f'just left camera: {indices_left}')
    print(f'just right camera: {indices_right}')
    return


# split_images('s1')
