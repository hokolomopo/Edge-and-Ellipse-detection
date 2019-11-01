import cv2
import numpy as np
import tools
import matplotlib.cm as cm
from ipywidgets import interact, fixed

from image import *
from edge import *
from line import *


def update_edges(   method="sobel", image="building", low_filtering=True,
                    low_filter_type="gaussian", low_filtering_kernel_size=5,
                    high_filtering=False, high_filter_type="gaussian",
                    high_filtering_kernel_size=3, strength=2.0, threshold=True,
                    block_size=11, constant=3, edge_threshold=30,
                    low_threshold=50, high_threshold=255, aperture_size=3):

    img = load_gray_img("img/{}.png".format(image))

    img_filtered = filtering(img, low_filtering, low_filter_type, low_filtering_kernel_size,
                        high_filtering, high_filter_type, high_filtering_kernel_size, strength)

    img_threshold = img_filtered
    if threshold:
        img_threshold = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, constant)

    if method == "sobel":
        grad = sobel_edge(img_threshold, thresholding = True, threshold = edge_threshold)

    elif method == "naive_gradient":
        grad = naive_gradient(img_threshold, thresholding = True, threshold = edge_threshold)

    elif method == "scharr":
        grad = scharr_edge(img_threshold, thresholding = True, threshold = edge_threshold)

    elif method == "stacking":
        grad = stacking(img_threshold, thresholding = True, threshold = edge_threshold)

    elif method == "beucher":
        grad = beucher_edge(img_threshold, thresholding = True, threshold = edge_threshold)

    elif method == "canny":
        grad = canny_edge(img, low_threshold = low_threshold, high_threshold = high_threshold, aperture_size = aperture_size)

    tools.multiPlot(1, 4,
            (img, img_filtered, img_threshold, grad),
            ('Original Image', 'Filtered Image', 'Thresholded Image', 'Edges'),
            cmap_tuple=(cm.gray, cm.gray, cm.gray, cm.gray, cm.gray, cm.gray))
