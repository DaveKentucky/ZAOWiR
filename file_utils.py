import os
import json


def write_indices_to_file(source_folder, target_folder, cameras, indices):
    """
    Writes indices from a list to a text file
    :param source_folder: name of the source dir of the images
    :type source_folder: str
    :param target_folder: name of the target dir for the output file
    :type target_folder: str
    :param cameras: names of cameras where the pattern was visible, also identifiers of the images
    :type cameras: str
    :param indices: list of indices
    :type indices: list
    :return: if write operation succeeded
    """
    if len(indices) > 0:
        file = open(os.path.join(target_folder, f'indices_{cameras.replace(" ", "_")}_{source_folder}.txt'), 'w')
        file.write(f'{source_folder}\n{cameras}\n')
        for index in indices:
            file.write(str(index) + '\n')
        file.close()
        return True
    else:
        return False


def read_images(source_folder, filename):
    """
    Reads images from file
    :param source_folder: name of the source dir of the indices files
    :type source_folder: str
    :param filename: file with info about images to read (created with split_images function)
    :type filename: str
    :return: list of image files to calibrate
    :rtype: list
    """
    # read file contents
    file = open(os.path.join(source_folder, filename), 'r')
    content = file.read()
    file.close()

    # extract important params from first 2 lines
    lines = content.split('\n')
    source_folder = lines[0]
    cameras = lines[1].split(' ')

    # create list with names of all image files
    lines = lines[2:]
    images_filenames = []
    for index in lines[:-1]:
        for camera in cameras:
            images_filenames.append(os.path.join(source_folder, camera + '_' + index + '.png'))
    return images_filenames


def write_calibration_params_to_file(mtx, dist):
    """
    Write calibration camera parameters to JSON file
    :param mtx: camera matrix
    :type mtx: numpy.ndarray
    :param dist: distortion matrix
    :type dist: numpy.ndarray
    :return: camera matrix and distortion matrix in JSON format
    :rtype: tuple
    """
    mtx_json = mtx.tolist()
    dist_json = dist.tolist()
    dictionary = {
        "mtx": mtx_json,
        "dist": dist_json
    }
    file = open('calibration_params.json', 'w')
    file.write(json.dumps(dictionary))

    return mtx_json, dist_json
