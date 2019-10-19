import numpy as np
import cv2
import argparse

from image import *
from edge import *

def show_main():
    # Load the image
    img = load_gray_img('img/building.png')
    display_img(img)

def sobel_main():
    img = load_gray_img('img/road.png')
    grad = sobel_edge(img)
    display_img(grad)

def beucher_main():
    img = load_gray_img('img/road.png')
    grad = beucher(img)
    display_img(grad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-mode", type=str, choices=["show", "sobel", "beucher", "compare"], default='show')
    
    args = parser.parse_args()
    mode = args.mode
   
    if mode == "show":
        show_main()
    
    elif mode == "sobel":
        sobel_main()

    elif mode == "beucher":
        beucher_main()

    elif mode=="compare":
        img = load_gray_img('img/road.png')
        img1 = beucher(img)
        img2 = cv2.fastNlMeansDenoising(img1, None,10,7,21)
        # img2 = np.where(img1 > 50, img1, 0)
        display_img(np.concatenate((img1, img2),axis=1))
