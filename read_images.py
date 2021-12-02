def read_images(filename):
    """
    Read images from file
    :param filename: file with info about images to read (created with split_images function)
    :type filename: str
    :return: list of image files to calibrate
    :rtype: list
    """
    file = open(filename, 'r')
    content = file.read()
    file.close()
    images_indices = content.split('\n')
    folder_name = images_indices[0]
    side_names = images_indices[1].split(' ')
    images_indices = images_indices[2:]
    images_filenames = []
    for index in images_indices[:-1]:
        for side in side_names:
            images_filenames.append(f'{folder_name}\\{side}_{index}.png')

    return images_filenames

# print(read_images('indices_left_only_s1.txt'))
