import numpy as np
import cv2 as cv
import random
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()

cap = cv.VideoCapture(args.image if not args.image == "" else 0)
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
execution_times = []

while True:
    start_time = time.time()
    ret, frame2 = cap.read()
    nxt = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # create binary mask out of the magnitude array
    _, mask = cv.threshold(mag, 127, 255, cv.THRESH_TRUNC)
    mask8bit = cv.normalize(mask, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    th, thresh_mask = cv.threshold(mask8bit, 50, 192, cv.THRESH_OTSU)

    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    blurred = cv.GaussianBlur(bgr, (5, 5), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    masked = cv.bitwise_and(gray, thresh_mask)

    sift = cv.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(prvs, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(masked, None)

    # Define parameters for our Flann Matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
    search_params = dict(checks=100)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Obtain matches using K-Nearest Neighbor Method
    # the result 'matches' is the number of similar matches found in both images
    if descriptors_1 is None or descriptors_2 is None:
        continue
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Store good matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    drawing = cv.cvtColor(nxt, cv.COLOR_GRAY2RGB)

    if len(good_matches) > 10:
        list_kp1 = [keypoints_1[mat.queryIdx].pt for mat in good_matches]
        list_kp2 = [keypoints_2[mat.trainIdx].pt for mat in good_matches]

        # for x, y in list_kp2:
        #     cv.circle(drawing, (int(x), int(y)), 3, (0, 255, 0))
        # cv.rectangle(frame, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), (0,255,0), 3)
        # cv2.putText(frame,'Object Found',(50,50), cv2.FONT_HERSHEY_COMPLEX, 2 ,(0,255,0), 2)

    # detect keypoints on the image filtered with motion areas mask
    ret, thresh = cv.threshold(masked, 150, 255, cv.THRESH_BINARY)
    canny = cv.Canny(masked, 100, 200)
    contours, _ = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1)
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])

    for i in range(len(contours)):
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
                     (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)

    execution_times.append(time.time() - start_time)

    cv.imshow('Objects', drawing)

    cv.imshow('frame2', bgr)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)
    prvs = nxt

print(f'Mean execution time: {sum(execution_times) / len(execution_times)}')
