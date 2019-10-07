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
    img = load_gray_img('img/building.png')
    grad = sobel_edge(img)
    display_img(grad)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-mode", type=str, choices=["show", "sobel"], default='show')
    
    args = parser.parse_args()
    mode = args.mode
   
    if mode == "show":
        show_main()
    
    elif mode == "sobel":
        sobel_main()