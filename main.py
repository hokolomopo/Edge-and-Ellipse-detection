import numpy as np
import cv2
import argparse
import math

from copy import deepcopy
from image import *
from edge import *

import tools
import matplotlib.cm as cm   

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
    grad_4 = method(img, threshold = 4)
    grad_6 = method(img, threshold = 6)
    grad_8 = method(img, threshold = 8)
    grad_16 = method(img, threshold = 16)
    grad_32 = method(img, threshold = 32)
    grad_64 = method(img, threshold = 64)
    grad_96 = method(img, threshold = 96)
    grad_128 = method(img, threshold = 128)
    imgList = (img, grad_4, grad_6, grad_8, grad_16, grad_32, grad_64, grad_96, grad_128)
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

def beucher_main():
    img = load_gray_img('img/road.png')
    grad = beucher(img)
    display_img(grad)

def test_lines_main(imageName):
    img = load_gray_img(imageName)
    test_lines(img)

def test_lines(img):

    # Find edges
    edges = sobel_edge(img, threshold=10)

    # Apply Hough method
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)

    # Draw image wiht lines foud with Hough
    img_w_lines = np.zeros(edges.shape)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_w_lines, (x1, y1), (x2, y2), (255, 0, 0), 1)  

    # Find Edges points that belong to a line
    final = np.zeros(edges.shape)
    kernel_size = 3
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

    tools.multiPlot(1, 4, 
                (edges, img_w_lines, final, edges-final),
                ('Edges', 'Lines detected', 'Only Edges on Lines', 'Only Edges not on Lines'),
                cmap_tuple=(cm.gray, cm.gray, cm.gray, cm.gray))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("-mode", type=str, choices=["show", "sobel", "naive_gradient", "scharr", "stacking", "compare_gradient", "beucher", "compare", "lines"], default='lines')
    parser.add_argument("-task", type=str, choices=["main", "threshold", "filter", "kernel_filter", "filter_strength"], default = "main")
    parser.add_argument("-image", type=str, default="building")
    
    args = parser.parse_args()
    mode = args.mode
    task = args.task
    imageName = "img/{}.png".format(args.image)
   
    if mode == "show":
        show_main(imageName)

    elif mode == "compare_gradient":
        compare_gradient_main(imageName)

    elif mode == "beucher":
        beucher_main()

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