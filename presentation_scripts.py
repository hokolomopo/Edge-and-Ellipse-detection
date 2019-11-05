import numpy as np
import cv2
import argparse
import math

from copy import deepcopy
from image import *
from edge import *
from line import *

import tools
import matplotlib.cm as cm   

FOLDER_NAME = "print/"

def format_file_name(operation, imageName):
    return FOLDER_NAME + imageName + "_" + operation +".png"

def print_main(imageName):
    img = load_gray_img("img/{}.png".format(imageName))

    cv2.imwrite(format_file_name("", imageName), img)

    #Low pass
    filtered = filtering(img, low_filtering=True,
                            low_filter_type="uniform",
                            low_filtering_kernel_size=7,
                            high_filtering=False,
                            high_filter_type="gaussian",
                            high_filtering_kernel_size=3,
                            high_filtering_strength=5)

    cv2.imwrite(format_file_name("LP", imageName), filtered)

    #High pass
    filtered = filtering(img, low_filtering=False,
                            low_filter_type="uniform",
                            low_filtering_kernel_size=7,
                            high_filtering=True,
                            high_filter_type="gaussian",
                            high_filtering_kernel_size=3,
                            high_filtering_strength=1.5)

    cv2.imwrite(format_file_name("HP", imageName), filtered)

    #Threshold
    filtered = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 45, 7)

    cv2.imwrite(format_file_name("Threshold", imageName), filtered)

    #Sobel
    filtered = get_optimal_grads(imageName, imageName, "Sobel")
    cv2.imwrite(format_file_name("Sobel", imageName), filtered)

    #Naive Gradient
    filtered = get_optimal_grads(imageName, imageName, "Naive Gradient")
    cv2.imwrite(format_file_name("Naive", imageName), filtered)

    #Scharr
    filtered = get_optimal_grads(imageName, imageName, "Scharr")
    cv2.imwrite(format_file_name("Scharr", imageName), filtered)

    #Canny
    filtered = get_optimal_grads(imageName, imageName, "Canny")
    cv2.imwrite(format_file_name("Canny", imageName), filtered)

    #Beucher
    filtered = get_optimal_grads(imageName, imageName, "Beucher")
    cv2.imwrite(format_file_name("Beucher", imageName), filtered)

    #Stacking
    filtered = get_optimal_grads(imageName, imageName, "Stacking")
    cv2.imwrite(format_file_name("Stacking", imageName), filtered)

    #Laplacian
    filtered = get_optimal_grads(imageName, imageName, "Laplacian")
    cv2.imwrite(format_file_name("Laplacian", imageName), filtered)

    naive_grad = get_optimal_grads(imageName, imageName, "Naive Gradient")
    sobel_grad = get_optimal_grads(imageName, imageName, "Sobel")
    scharr_grad = get_optimal_grads(imageName, imageName, "Scharr")
    canny_grad = get_optimal_grads(imageName, imageName, "Canny")
    beucher_grad = get_optimal_grads(imageName, imageName, "Beucher")
    stacking_grad = get_optimal_grads(imageName, imageName, "Stacking")
    laplacian_grad = get_optimal_grads(imageName, imageName, "Laplacian")

    tools.multiPlot(2, 4, 
            (img, naive_grad, sobel_grad, scharr_grad, laplacian_grad, beucher_grad, canny_grad, stacking_grad),
            ('Original Image', 'Naive Gradient', 'Sobel', 'Scharr', 'Laplacian', 'Beucher Gradient', 'Canny', 'Stacking'),
            cmap_tuple=(cm.gray, cm.gray, cm.gray, cm.gray, cm.gray, cm.gray, cm.gray, cm.gray),
            printFile=format_file_name("AllEdges", imageName))


