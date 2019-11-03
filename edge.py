import cv2
import numpy as np
import tools
from image import *

def saturate_cast_uint8(img):
    """
    Taken from practical sessions

    Saturate an image (make the pixels between 0 and 255)

    Parameters
    ----------
    - img: The image to saturate

    Returns
    -------
    The saturated image
    """
    return np.where(img > 255.0, 255.0,
                    np.where( img < 0.0, 0.0, img)).astype(np.uint8)

def clean_grad(grad, saturate, threshold):
    """
    Clean the gradients by either saturating those (make them be between 0 and
    255) or thresholding those or both.

    Parameters
    ----------
    - saturate:  A boolean indicating whether to saturate the gradients.
    - threshold: A boolean indicating whether to threshold the gradients.

    Returns
    -------
    The cleaned gradients
    """
    if saturate:
        grad = saturate_cast_uint8(grad)

    if threshold is not None:
        grad = np.where(grad > threshold, 255., 0.).astype(np.uint8)

    return grad

def low_pass_filter(img, kernel_size, filter_type):
    """
    Apply a low pass filter on an image.

    Parameters
    ----------
    - kernel_size: The kernel size of the filter
    - filter_type: The type of filter to apply may be either "uniform",
                   "median" or "gaussian".

    Returns
    -------
    The low-pass filtered image
    """
    if filter_type == "uniform":
        ka = np.ones((kernel_size, kernel_size), dtype=float) / kernel_size**2
        lowPass = cv2.filter2D(img, -1, ka, borderType=cv2.BORDER_CONSTANT)
    if filter_type == "median":
        lowPass = cv2.medianBlur(img, kernel_size)
    elif filter_type == "gaussian":
        lowPass = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    return lowPass

def high_pass_filter(img, kernel_size, strength, filter_type):
    """
    Taken from practical sessions

    Apply a high pass filter on an image

    Parameters
    ----------
    - img:         The image on which to apply a high pass filter
    - kernel_size: The kernel size of the filter
    - strength:    The strength of the filter
    - filter_type: The type of the filter

    Returns
    -------
    The high-pass filtered image
    """
    lowPass = low_pass_filter(img, kernel_size, filter_type)
    return saturate_cast_uint8(strength * img - ( strength - 1.0) * lowPass)

def filtering(img, low_filtering=False, low_filter_type=None,
              low_filtering_kernel_size=None, high_filtering=False,
              high_filter_type=None, high_filtering_kernel_size=None,
              high_filtering_strength=None):

    """
    Filter an image by apply optionnally a low pass filter followed by a high
    pass filter

    Parameters
    ----------
    - img:                     The image on which to apply the filter
    - low_filtering:           A boolean indicating whether to apply a low pass
                               filter
    - low_filter_type:         The type of low pass filtering to apply, can be
                               either "uniform", "median" or "gaussian".
    - low_filter_kernel_size:  The kernel size of the low pass filter
    - high_filtering:          A boolean indicating whether to apply a high pass
                               filter
    - high_filter_type:        The type of high pass filtering to apply, can be
                               either "uniform", "median" or "gaussian".
    - ligh_filter_kernel_size: The kernel size of the high pass filter
    - high_filtering_strength: The strength of the high pass filter

    Returns
    -------
    The filtered image
    """
    if low_filtering:
        img = low_pass_filter(img, low_filtering_kernel_size, low_filter_type)

    if high_filtering:
        img = high_pass_filter(img, high_filtering_kernel_size, high_filtering_strength, high_filter_type)

    return img

def sobel_edge(img, thresholding=True, threshold=15, kernel_size=1):
    """
    https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/
    sobel_derivatives/sobel_derivatives.html

    Compute the edges of an image using the sobel method

    Parameters
    ----------
    - img:          The image on which to apply the filter
    - thresholding: Boolean for wether or not a thresholing is applied to the
                    edges
    - threshold:    The threshold at which to consider a pixel as an edge
    - kernel_size:  The size of the sobel kernel.

    Returns
    -------
    A opencv image with the pixel corresponding to edges set to 255 and the
    others set to 0
    """
    gradientX = cv2.Sobel(img, -1, 1, 0, kernel_size)
    gradientY = cv2.Sobel(img, -1, 0, 1, kernel_size)

    absGradientX = cv2.convertScaleAbs(gradientX)
    absGradientY = cv2.convertScaleAbs(gradientY)

    grad = cv2.addWeighted(absGradientX, 0.5, absGradientY, 0.5, 0)

    if thresholding:
        grad = clean_grad(grad, False, threshold)

    return grad

def naive_gradient(img, thresholding=True, threshold=16):
    """
    Compute the edges of an image computing gradients of the image.

    Parameters
    ----------
    - img:          The image on which to apply the filter
    - thresholding: Boolean for wether or not a thresholing is applied to the
                    edges
    - threshold:    The threshold at which to consider a pixel as an edge

    Returns
    -------
    A opencv image with the pixel corresponding to edges set to 255 and the
    others set to 0
    """
    xFilter = np.array([[-1, 0, 1]])
    yFilter = np.array([[-1], [0], [1]])

    gradientX = cv2.filter2D(img, cv2.CV_64F, xFilter)
    gradientY = cv2.filter2D(img, cv2.CV_64F, yFilter)

    absGradientX = cv2.convertScaleAbs(gradientX)
    absGradientY = cv2.convertScaleAbs(gradientY)

    grad = cv2.addWeighted(absGradientX, 0.5, absGradientY, 0.5, 0)

    if thresholding:
        grad = clean_grad(grad, False, threshold)

    return grad

def scharr_edge(img, thresholding = True, threshold=64):
    """
    Compute the edges of an image using the Scharr method.

    Parameters
    ----------
    - img:          The image on which to apply the filter
    - thresholding: Boolean for wether or not a thresholing is applied to the
                    edges
    - threshold:    The threshold at which to consider a pixel as an edge

    Returns
    -------
    A opencv image with the pixel corresponding to edges set to 255 and the
    others set to 0
    """
    gradientX = cv2.Scharr(img, cv2.CV_32F, 1, 0)/4.0
    gradientY = cv2.Scharr(img, cv2.CV_32F, 0, 1)/4.0

    absGradientX = cv2.convertScaleAbs(gradientX)
    absGradientY = cv2.convertScaleAbs(gradientY)

    grad = cv2.addWeighted(absGradientX, 0.5, absGradientY, 0.5, 0)

    if thresholding:
        grad = clean_grad(grad, False, threshold)

    return grad

def canny_edge(img, low_threshold=50, high_threshold=255, aperture_size=3):
    """
    Compute the edges of an image using the Canny method.

    Parameters
    ----------
    - img:           The image on which to apply the filter
    - aperture_size: The aperture size of the sobel operator

    Returns
    -------
    A opencv image with the pixel corresponding to edges set to 255 and the
    others set to 0
    """
    return cv2.Canny(img, low_threshold, high_threshold, None, aperture_size)

def stacking(grads, thresholding = False, threshold = 128):
    """
    Compute the edges of an image stacking all the existing methods. A pixel
    is classified as an edge if enough methods have classified it as an edge

    Parameters
    ----------
    - thresholding: A boolean indicating whether to perform thresholding or not.
    - threshold:    The threshold to apply if thresholding is set to True.

    Returns
    -------
    A opencv image with the pixel corresponding to edges set to 255 and the
    others set to 0
    """

    weight = 1/len(grads)
    if weight == 1:
        return grads[0]

    tot = cv2.addWeighted(grads[0], weight, grads[1], weight, 0)
    for grad in grads[2:]:
        tot = cv2.addWeighted(tot, 1., grad, weight, 0)

    if thresholding:
        tot = clean_grad(tot, False, threshold)

    return tot

def beucher_edge(img, thresholding=True, threshold=60, kernel_size = 3):
    """
    Compute the edges of an image using the Beucher Gradient method.

    Parameters
    ----------
    - img:           The image on which to apply the filter
    - thresholding:  Boolean for wether or not a thresholing is applied to the
                     edges
    - threshold:    The threshold to apply if thresholding is set to True.

    Returns
    -------
    A opencv image with the edges of the original image
    """

    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    if thresholding:
        grad = clean_grad(grad, False, threshold)

    return grad


def get_edges_on_lines(edges, img_w_lines, kernel_size=3):
    """
    Get the edges that belong to a line in an image

    Parameters
    ----------

    - edges :           The edges of the image
    - img_w_lines :     Image with all the lines drawed
    - kernel_size :     Size of the kernel to determine if a point of an edge is on a line

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


def get_optimal_grads(image, method):
    img = load_gray_img("img/{}.png".format(image))

    if image == "building":
        if method == "Sobel":
            filtered = filtering(img, low_filtering=True,
                                 low_filter_type="uniform",
                                 low_filtering_kernel_size=7,
                                 high_filtering=True,
                                 high_filter_type="gaussian",
                                 high_filtering_kernel_size=3,
                                 high_filtering_strength=1.5)

            grad = sobel_edge(filtered, thresholding=True, threshold=30,
                              kernel_size=5)

        elif method == "Naive Gradient":
            filtered = filtering(img, low_filtering=True,
                                 low_filter_type="uniform",
                                 low_filtering_kernel_size=9,
                                 high_filtering=True,
                                 high_filter_type="gaussian",
                                 high_filtering_kernel_size=5,
                                 high_filtering_strength=2.)

            grad = naive_gradient(filtered, thresholding=True, threshold=10)

        elif method == "Scharr":
            filtered = filtering(img, low_filtering=True,
                                 low_filter_type="uniform",
                                 low_filtering_kernel_size=7,
                                 high_filtering=True,
                                 high_filter_type="gaussian",
                                 high_filtering_kernel_size=3,
                                 high_filtering_strength=2.)

            grad = scharr_edge(filtered, thresholding=True, threshold=50)

        elif method == "Beucher":
            filtered = filtering(img, low_filtering=True,
                                 low_filter_type="uniform",
                                 low_filtering_kernel_size=7,
                                 high_filtering=True,
                                 high_filter_type="gaussian",
                                 high_filtering_kernel_size=3,
                                 high_filtering_strength=2.)

            grad = beucher_edge(filtered, thresholding=True, threshold=23,
                                kernel_size=3)

        elif method == "Canny":
            filtered = filtering(img, low_filtering=True,
                                 low_filter_type="uniform",
                                 low_filtering_kernel_size=7,
                                 high_filtering=True,
                                 high_filter_type="gaussian",
                                 high_filtering_kernel_size=3,
                                 high_filtering_strength=2.)

            grad = canny_edge(filtered, low_threshold=230,
                               high_threshold=255, aperture_size=3)

        elif method == "Stacking":
            grads = [get_optimal_grads(image, "Sobel"),
                     get_optimal_grads(image, "Naive Gradient"),
                     get_optimal_grads(image, "Scharr"),
                     get_optimal_grads(image, "Beucher")]

            grad = stacking(grads, thresholding=True, threshold=129)

    elif image == "sudoku":
        if method == "Sobel":
            filtered = filtering(img, low_filtering=True,
                                 low_filter_type="uniform",
                                 low_filtering_kernel_size=3,
                                 high_filtering=True,
                                 high_filter_type="gaussian",
                                 high_filtering_kernel_size=3,
                                 high_filtering_strength=2.)

            grad = sobel_edge(filtered, thresholding=True, threshold=20,
                              kernel_size=5)

        elif method == "Naive Gradient":
            filtered = filtering(img, low_filtering=True,
                                 low_filter_type="uniform",
                                 low_filtering_kernel_size=3,
                                 high_filtering=True,
                                 high_filter_type="gaussian",
                                 high_filtering_kernel_size=7,
                                 high_filtering_strength=3.)

            filtered = cv2.adaptiveThreshold(filtered, 255,
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY,
                                             27, 15)

            grad = naive_gradient(filtered, thresholding=True, threshold=13)

        elif method == "Scharr":
            filtered = filtering(img, low_filtering=True,
                                 low_filter_type="uniform",
                                 low_filtering_kernel_size=3,
                                 high_filtering=True,
                                 high_filter_type="gaussian",
                                 high_filtering_kernel_size=5,
                                 high_filtering_strength=5.5)

            filtered = cv2.adaptiveThreshold(filtered, 255,
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY,
                                             27, 15)

            grad = scharr_edge(filtered, thresholding=True, threshold=55)

        elif method == "Beucher":
            filtered = filtering(img, low_filtering=True,
                                 low_filter_type="uniform",
                                 low_filtering_kernel_size=7,
                                 high_filtering=True,
                                 high_filter_type="gaussian",
                                 high_filtering_kernel_size=3,
                                 high_filtering_strength=1.)

            filtered = cv2.adaptiveThreshold(filtered, 255,
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY,
                                             27, 13)

            grad = beucher_edge(filtered, thresholding=True, threshold=97,
                                kernel_size=3)

        elif method == "Canny":
            filtered = filtering(img, low_filtering=True,
                                 low_filter_type="uniform",
                                 low_filtering_kernel_size=3,
                                 high_filtering=True,
                                 high_filter_type="gaussian",
                                 high_filtering_kernel_size=7,
                                 high_filtering_strength=2.)

            grad = canny_edge(filtered, low_threshold=44,
                               high_threshold=54, aperture_size=3)

        elif method == "Stacking":
            grads = [get_optimal_grads(image, "Sobel"),
                     get_optimal_grads(image, "Naive Gradient"),
                     get_optimal_grads(image, "Scharr"),
                     get_optimal_grads(image, "Beucher")]

            grad = stacking(grads, thresholding=True, threshold=103)

    else:
        filtered = filtering(img)

        if method == "Sobel":
            grad = sobel_edge(filtered)

        elif method == "Naive Gradient":
            grad = naive_gradient(filtered)

        elif method == "Scharr":
            grad = scharr_edge(filtered)

        elif method == "Beucher":
            grad = beucher_edge(filtered)

        elif method == "Canny":
            grad = canny_edge(filtered)

        elif method == "Stacking":
            grads = [get_optimal_grads(image, "Sobel"),
                     get_optimal_grads(image, "Naive Gradient"),
                     get_optimal_grads(image, "Scharr"),
                     get_optimal_grads(image, "Beucher")]

            grad = stacking(grads)

    return grad
