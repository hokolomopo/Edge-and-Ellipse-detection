import cv2
import numpy as np
import math
import tools
import matplotlib.cm as cm


def hough_determinist(edge_img, img_print, rho=1, theta=np.pi / 180,
                      threshold=300):
    """
    Returns the contours and draw them on the second argument

    Parameters
    ----------
    - edge_img :        The image gradient on which to detect the contour
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
    if lines is not None:
        for line in lines:
            for rho, theta in line:
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv2.line(img_print, pt1, pt2, (255, 0, 0), 1)

    return lines


def hough_determinist_print(edge_img, img_print, rho=1, theta=np.pi / 180,
                            threshold=300):
    """
    Print the edges next to the original image

    Parameters
    ----------
    - edge_img :        The image on which to detect the contour
    - img_print :       Image on which to draw the contour
    - rho :             Distance resolution of the accumulator in pixels.
    - theta :           Angle resolution of the accumulator in radians.
    - threshold :       Accumulator threshold parameter. Only those lines are
                        returned that get enough votes
    """
    img_print = cv2.cvtColor(img_print, cv2.COLOR_GRAY2BGR)
    hough_determinist(edge_img, img_print, rho, theta, threshold)

    tools.multiPlot(1, 2,
        (edge_img, img_print),
        ('Original image', 'Lines detected'),
        cmap_tuple=(cm.gray, cm.gray))


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


def hough_probabilist_print(edge_img, img_print, rho=1, theta=np.pi / 180,
                            threshold=50, minLineLength=30, maxLineGap=10):
    """
    Print the edges next to the original image

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
    """
    img_print = cv2.cvtColor(img_print, cv2.COLOR_GRAY2BGR)
    hough_probabilist(  edge_img, img_print, rho, theta, threshold,
                        minLineLength, maxLineGap)
    tools.multiPlot(1, 2,
        (edge_img, img_print),
        ('Original image', 'Lines detected'),
        cmap_tuple=(cm.gray, cm.gray))


def get_edges_on_lines(edges, img_w_lines, kernel_size=3):
    """
    Get the edges that belong to a line in an image

    Parameters
    ----------

    - edges :           The edges of the image
    - img_w_lines :     Image with all the lines drawed
    - kernel_size :     Size of the kernel to determine if a point of an edge is
                        on a line

    Return
    ------
    The image containing only the edges points on a line
    """
    final = np.zeros(edges.shape)
    kernel = np.ones((kernel_size, kernel_size))
    r = cv2.filter2D(img_w_lines, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if(edges[i][j] == 0):
                continue
            if(r[i][j] > 0):
                final[i][j] = 255
            else:
                final[i][j] = 0

    return final
