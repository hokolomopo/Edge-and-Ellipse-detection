import cv2
import numpy as np
import math


def hough_determinist(edge_img, img_print, rho=1, theta=np.pi / 180,
                      threshold=300):

    lines = cv2.HoughLines(edge_img, rho, theta, threshold)

    for line in lines:
        for rho, theta in line:
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(img_print, pt1, pt2, (255, 0, 0), 1)

    return line


def hough_probabilist(edge_img, img_print, rho=1, theta=np.pi / 180,
                      threshold=50, minLineLength=30, maxLineGap=10):

    lines = cv2.HoughLinesP(edge_img, rho, theta, threshold, minLineLength,
                            maxLineGap)

    for (x1, y1, x2, y2) in lines[:, 0]:
        cv2.line(img_print, (x1, y1), (x2, y2), (255, 0, 0), 1)

    return lines[:, 0]


def contours(img, img_print, thresh=240, maxval=255):

    _, thresh = cv2.threshold(img, thresh, maxval, 0)
    contours, _ = cv2.findContours(thresh,
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(img_print, contours, -1, (255, 0, 0), 1)

    return contours
