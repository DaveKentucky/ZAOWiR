import os
import json
import numpy as np


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


def write_calibration_params_to_file(matrices, output_file):
    """
    Writes camera calibration parameters to JSON file
    :param matrices: dictionary with parameters stored in numpy arrays
    :type matrices: dict
    :param output_file: name of the output file
    :type output_file: str
    :return: dictionary with parameters matrices converted into JSON format
    :rtype: dict
    """
    for key in matrices:
        matrices[key] = matrices[key].tolist()
    with open(output_file, 'w') as file:
        file.write(json.dumps(matrices))
    return matrices


def read_calibration_params_from_file(params_file):
    """
    Reads camera calibration parameters from JSON file
    :param params_file: name of the input file with parameters
    :type params_file: str
    :return: dictionary with parameters matrices converted into numpy arrays
    :rtype: dict
    """
    with open(params_file) as file:
        data = json.load(file)
        for key in data:
            data[key] = np.array(data[key])
    return data
