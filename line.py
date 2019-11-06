#############################################################################
#       This file contains all the functions associated with                #
#       lines extraction of an image                                        #
#############################################################################


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
                pt1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * (a)))
                pt2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * (a)))
                cv2.line(img_print, pt1, pt2, (255, 0, 0), 2)

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
    lines = cv2.HoughLinesP(edge_img, rho, theta, threshold, None,
                            minLineLength, maxLineGap)

    for (x1, y1, x2, y2) in lines[:, 0]:
        cv2.line(img_print, (x1, y1), (x2, y2), (255, 0, 0), 2)

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
    The image containing only the edges points on a line and the convoluted image
    """

    #Create image that will contain result
    final = np.zeros(edges.shape)

    #Widen the lines
    r = widen_lines(img_w_lines, kernel_size=kernel_size)

    #Apply classifier to edges
    t = np.indices((final.shape[0],final.shape[1])).transpose((1,2,0))
    f = lambda c : classify(c[0], c[1], r, edges, final)
    np.apply_along_axis(f, axis=2, arr=t)

    return (final, r)

def classify(x, y, lines, edges, final):
    """
    Function that classifiy an edge point as being on a line or not and put the result in a given array

    Parameters
    ----------

    - x :               The x cocordinate of the point
    - y :               The y cocordinate of the point
    - lines :           The image containing the lines
    - edges :           The image containing the edges
    - final :           The image on wuich to print the result
    """

    if(edges[x][y] == 0):
        return
    if(lines[x][y] > 0):
        final[x][y] = 255
    else:
        final[x][y] = 0

def widen_lines(img, kernel_size=3):
    """
    Widen the lines of the image

    Parameters
    ----------
    - img :             The image
    - kernel_size :     The size of the kernel used to widen the lines

    Return
    ------
    The image with widened lines
    """
    kernel = np.ones((kernel_size, kernel_size))
    r = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    r = np.where(r > 0, 255, 0).astype(np.uint8)

    return r


def get_optimal_lines(imageName, edges, print_img, lines_method):
    """
    Get the lines in the image with the tuned parameters for Hough

    Parameters
    ----------
    - img :         The image on which to get the lines
    - edges :       Extracted edges of the image
    - print_img :   Image on which to print the lines
    - lines_method :Method to extract the lines (Hough or HoughProba)


    Return
    ------
    The lines of the images
    """
    if lines_method == "Hough":
        if imageName == "sudoku":
            lines = hough_determinist(edges, print_img, 1, 0.015, 240)
        else:
            lines = hough_determinist(edges, print_img, 1, 0.015, 360)
    elif lines_method == "HoughProba":
        if imageName == "soccer":
            lines = hough_probabilist(edges, print_img, 1, 0.03, 80, 110, 5)
        elif imageName == "sudoku":
            lines = hough_probabilist(edges, print_img, 1, 0.015, 50, 100, 5)
        else:
            lines = hough_probabilist(edges, print_img, 1, 0.015, 50, 30, 5)

    return lines
