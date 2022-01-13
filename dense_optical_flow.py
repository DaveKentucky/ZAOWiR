import numpy as np
import cv2 as cv
import random


cap = cv.VideoCapture('vtest.avi')
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # create binary mask out of the magnitude array
    _, mask = cv.threshold(mag, 127, 255, cv.THRESH_TRUNC)
    mask8bit = cv.normalize(mask, None, 0, 255, cv.NORM_MINMAX).astype('uint8')

    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # detect keypoints on the image filtered with motion areas mask
    masked = cv.bitwise_and(next, mask8bit)
    canny = cv.Canny(masked, 100, 200)
    contours, _ = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])

    drawing = cv.cvtColor(next, cv.COLOR_GRAY2RGB)

    for i in range(len(contours)):
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
                     (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)

    cv.imshow('Contours', drawing)

    # # Find contours
    # sift = cv.SIFT_create()
    # features, desc = sift.detectAndCompute(next, mask=mask8bit)
    # img = cv.drawKeypoints(next, features, frame2)

    cv.imshow('frame2', bgr)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)
    prvs = next
