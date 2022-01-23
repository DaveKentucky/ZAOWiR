import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def disparity_stereoBM(img1, img2, show=False):
    stereo = cv.StereoBM_create(numDisparities=128, blockSize=5)
    disparity = stereo.compute(img1, img2)
    cv.filterSpeckles(disparity, 0, 30, 128)
    # _, disparity = cv.threshold(
    #     disparity, 0, max_disparity * 16, cv.THRESH_TOZERO)
    # disparity_scaled = (disparity / 16.).astype(np.uint8)
    if show:
        plt.imshow(disparity / 4, 'gray')
        plt.show()
    return disparity


def disparity_stereoSGBM(img1, img2, show=False):
    stereo = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=5,
        speckleWindowSize=40,
        speckleRange=25,
        preFilterCap=20
    )
    disparity = stereo.compute(img1, img2)
    # cv.filterSpeckles(disparity, 0, 30, 128)
    # _, disparity = cv.threshold(
    #     disparity, 0, max_disparity * 16, cv.THRESH_TOZERO)
    # disparity_scaled = (disparity / 16.).astype(np.uint8)
    if show:
        plt.imshow(disparity / 4, 'gray')
        plt.show()
    return disparity


def preprocess_images(img1, img2):
    img1 = cv.equalizeHist(img1)
    img2 = cv.equalizeHist(img2)
    blurred1 = cv.GaussianBlur(img1, (5, 5), 0)
    blurred2 = cv.GaussianBlur(img2, (5, 5), 0)
    return blurred1, blurred2


def calculate_disparity_matrix(img1, img2):
    left_image, right_image = preprocess_images(img1, img2)
    rows, cols = right_image.shape

    kernel = np.ones([block_size, block_size]) / block_size

    disparity_maps = np.zeros(
        [left_image.shape[0], left_image.shape[1], num_disparities])
    for d in range(0, num_disparities):
        # shift image
        translation_matrix = np.float32([[1, 0, d], [0, 1, 0]])
        shifted_image = cv.warpAffine(
            right_image, translation_matrix,
            (cols, rows))
        # calculate squared differences
        SAD = abs(np.float32(left_image) - np.float32(shifted_image))
        # convolve with kernel and find SAD at each point
        filtered_image = cv.filter2D(SAD, -1, kernel)
        disparity_maps[:, :, d] = filtered_image

    disparity = np.argmin(disparity_maps, axis=2)
    disparity = np.uint8(disparity * 255 / num_disparities)
    disparity = cv.equalizeHist(disparity)
    plt.imshow(disparity, cmap='gray', vmin=0, vmax=255)
    plt.show()


img_left = cv.imread('cones/im2.png', 0)
img_right = cv.imread('cones/im6.png', 0)
num_disparities = 128
block_size = 7

# disparity_stereoBM(img_left, img_right, True)
# disparity_stereoSGBM(img_left, img_right, True)
calculate_disparity_matrix(img_left, img_right)
