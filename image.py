import cv2

def display_img(img, wait=True):
    """
    Display an image

    Parameters
    ----------

    img : the image to display
    wait : if True, wait for the windows to be closed before continuing the program
    """

    cv2.imshow('image', img)
    if wait == True:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def load_gray_img(fileName):
    """
    Load an image in gray scale from a file name.

    Parameters
    ----------
    fileName: The name of the file containing the image

    Return
    ------
    An opencv image

    """

    img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (500, 500))

    return img