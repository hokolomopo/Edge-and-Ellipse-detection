import cv2
import numpy as np
import math


def hough_determinist(edge_img, img_print, rho=1, theta=np.pi / 180,
                      threshold=300):
    """
    Returns the contours and draw them on the second argument

    Parameters
    ----------
    - edge_img :        The image detecting the on which to detect the contour
    - img_print :       Image on which to draw the contour
    - rho :             Distance resolution of the accumulator in pixels.
    - theta :           Angle resolution of the accumulator in radians.
    - threshold :       Accumulator threshold parameter. Only those lines are
                        returned that get enough votes

    Return
    ------
    The contour of the image
    """
    lines = cv2.HoughLines(edge_img, rho, theta, threshold)

    # https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
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
    """
    Returns the contours and draw them on the second argument

    Parameters
    ----------
    - edge_img :        The image detecting the on which to detect the contour
    - img_print :       Image on which to draw the contour
    - rho :             Distance resolution of the accumulator in pixels.
    - theta :           Angle resolution of the accumulator in radians.
    - threshold :       Accumulator threshold parameter. Only those lines are
                        returned that get enough votes.
    - minLineLength :   Minimum line length. Line segments shorter than that are
                        rejected.
    - maxLineGap :      Maximum allowed gap between points on the same line to
                        link them.

    Return
    ------
    The contour of the image
    """
    lines = cv2.HoughLinesP(edge_img, rho, theta, threshold, minLineLength,
                            maxLineGap)

    for (x1, y1, x2, y2) in lines[:, 0]:
        cv2.line(img_print, (x1, y1), (x2, y2), (255, 0, 0), 1)

    return lines[:, 0]


def contours(img, img_print, thresh=240, maxval=255):
    """
    Returns the contours and draw them on the second argument

    Parameters
    ----------

    - img :       The image on which to detect the contour
    - img_print : Image on which to draw the contour
    - thresh :    Threshold of intensity in the image
    - maxval :    Maximal value in the img

    Return
    ------
    The contour of the image
    """
    _, thresh = cv2.threshold(img, thresh, maxval, 0)
    contours, _ = cv2.findContours(thresh,
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(img_print, contours, -1, (255, 0, 0), 1)

    return contours
