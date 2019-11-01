import numpy as np
import cv2


def get_edges_on_lines(edges, img_w_lines, kernel_size = 3):
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