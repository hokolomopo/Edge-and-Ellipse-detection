import numpy as np
import cv2

def resize_img(img, sizeX, sizeY):
    return cv2.resize(img, (sizeX, sizeY))
 
def display_img(img, wait=True):
    """
    Display an image

    Parameters:
    img : the image to display
    wait : if True, wait for the windows to be closed before continuing the program
    """

    cv2.imshow('image', img)
    if wait == True:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":

    # Load the image
    img = cv2.imread('img/building.png', cv2.IMREAD_GRAYSCALE)
    display_img(resize_img(img, 500, 500))
