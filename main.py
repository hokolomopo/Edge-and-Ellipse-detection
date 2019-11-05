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

from presentation_scripts import *

def show_main(imageName):
    # Load the image
    img = load_gray_img(imageName)
    display_img(img)

def method_main(imageName, method):
    img = load_gray_img(imageName)
    grad_no_saturate = method(img, False)
    grad_saturate = method(img, True)
    imgList = (img, grad_no_saturate, grad_saturate)
    grad = np.concatenate(imgList, axis = 1)
    display_img(grad, len(imgList), True)

def threshold_main(imageName, method):
    img = load_gray_img(imageName)
    filtered = filtering(img)
    thresholds = [4, 6, 8, 16, 32, 64, 96, 128]
    grads = [method(filtered, threshold = x) for x in thresholds]
    imgList = [img] + grads
    grad = np.concatenate(imgList, axis = 1)
    display_img(grad, len(imgList), True)

def filtering_main(imageName, method):
    img = load_gray_img(imageName)
    no_filter = method(img, filtering = False)
    filter_ = method(img, filtering = True)
    imgList = (img, no_filter, filter_)
    grad = np.concatenate(imgList, axis = 1)
    display_img(grad, len(imgList), True)

def filter_kernel_main(imageName, method):
    img = load_gray_img(imageName)
    kernel_1 = method(img, filtering = True, filtering_kernel_size = 1)
    kernel_3 = method(img, filtering = True, filtering_kernel_size = 3)
    kernel_5 = method(img, filtering = True, filtering_kernel_size = 5)
    kernel_7 = method(img, filtering = True, filtering_kernel_size = 7)
    imgList = (img, kernel_1, kernel_3, kernel_5, kernel_7)
    grad = np.concatenate(imgList, axis = 1)
    display_img(grad, len(imgList), True)

def filter_strength_main(imageName, method):
    img = load_gray_img(imageName)
    strength_1 = method(img, filtering = True, filtering_strength = 1)
    strength_2 = method(img, filtering = True, filtering_strength = 2)
    strength_3 = method(img, filtering = True, filtering_strength = 3)
    strength_5 = method(img, filtering = True, filtering_strength = 5)
    imgList = (img, strength_1, strength_2, strength_3, strength_5)
    grad = np.concatenate(imgList, axis = 1)
    display_img(grad, len(imgList), True)

def compare_gradient_main(imageName):
    img = load_gray_img(imageName)
    grad_sobel = sobel_edge(img)
    grad_naive = naive_gradient(img)
    grad_scharr = scharr_edge(img)
    grad_majority = stacking(img)
    imgList = (img, grad_naive, grad_sobel, grad_scharr, grad_majority)
    grad = np.concatenate(imgList, axis = 1)
    display_img(grad, len(imgList), True)

def lines_main(image_name, edges_method):
    lines_method = "HoughProba"
    kernel_size = 3


    img = load_gray_img("img/soccer.png")

    scale_percent = 0.4
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # img = cv2.resize(img, dim)
    
    #Extract the Edges
    filtered = filtering(img, low_filtering=True,
                            low_filter_type="uniform",
                            low_filtering_kernel_size=7,
                            high_filtering=True,
                            high_filter_type="gaussian",
                            high_filtering_kernel_size=3,
                            high_filtering_strength=1.5)

    edges = sobel_edge(filtered, thresholding=True, threshold=30,
                        kernel_size=5)

    #Create an empty image
    img_w_lines = np.zeros(edges.shape)

    #Get the lines of the images and print them on the empty image
    get_optimal_lines("building", edges, img_w_lines, lines_method)
    
    #Convolve the lines and run the pixel-wise comparison with the edges
    final, conv = get_edges_on_lines(edges, img_w_lines, kernel_size)
    
    tools.multiPlot(1, 4, 
        (edges, img_w_lines, conv, final),
        ('Edges', 'Lines detected', 'Convolved lines', 'Only Edges on Lines'),
        cmap_tuple=(cm.gray, cm.gray, cm.gray, cm.gray))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("-mode", type=str, choices=["show", "sobel", "naive_gradient", "scharr", "stacking", "compare_gradient", "beucher", "compare", "print"], default='show')
    parser.add_argument("-task", type=str, choices=["main", "lines", "threshold", "filter", "kernel_filter", "filter_strength"], default = "main")
    parser.add_argument("-image", type=str, default="building")
    
    args = parser.parse_args()
    mode = args.mode
    task = args.task
    imageName = "img/{}.png".format(args.image)
   
    if mode == "show":
        show_main(imageName)

    if mode == "print":
        print_main(args.image)

    elif mode == "compare_gradient":
        compare_gradient_main(imageName)

    elif mode=="compare":
        img = load_gray_img('img/boat.png')
        # img1 = beucher(img
        img1 = sobel_edge(cv2.GaussianBlur( img, ( 3, 3), 0))
        img2 = high_pass(beucher(cv2.GaussianBlur( img, ( 3, 3), 0)))
        # img2 = np.where(img2 > 70, 255.0, 0)
        display_img(np.concatenate((img1, img2),axis=1))
    
    elif mode == "lines":
        test_lines_main('img/road.png')

    else:
        # Determine method
        if mode == "sobel":
            method = sobel_edge
        
        elif mode == "naive_gradient":
            method = naive_gradient
        
        elif mode == "scharr":
            method = scharr_edge

        elif mode == "stacking":
            method = stacking

        elif mode == "beucher":
            method = beucher


        # Determine task
        if task == "main":
            method_main(imageName, method)

        elif task == "threshold":
            threshold_main(imageName, method)

        elif task == "filter":
            filtering_main(imageName, method)

        elif task == "kernel_filter":
            filter_kernel_main(imageName, method)

        elif task == "filter_strength":
            filter_strength_main(imageName, method)

        elif task == "lines":
            lines_main(imageName, method)


    
