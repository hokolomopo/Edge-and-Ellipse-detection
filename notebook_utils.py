#############################################################################
#       This file contains all the functions used to lighten the code       #
#       of the Juypter notebook                                             #
#############################################################################


import cv2
import numpy as np
import tools
import matplotlib.cm as cm
from ipywidgets import interact, fixed, widgets

from image import *
from edge import *
from line import *

def update_method(method="Sobel"):
    kwargs = {
    "image": ["building", "sudoku", "soccer", "road", "pcb"],
    "low_filtering": True,
    "low_filter_type": ["uniform", "median", "gaussian"],
    "low_filtering_kernel_size": (1, 9, 2),
    "high_filtering": True,
    "high_filter_type": ["uniform", "median", "gaussian"],
    "high_filtering_kernel_size": (1, 9, 2),
    "strength": (1.5, 5.0, 0.5),
    "threshold": True,
    "block_size": (3, 27, 4),
    "constant": (1, 11, 2)
    }

    if method == "Sobel":
        interact((lambda **x: update_edges(method = "Sobel", **x)),
                  **dict(kwargs, **{"edge_threshold": (0, 100, 10),
                                    "kernel_size": (1, 7, 2)}))

    elif method == "canny":
        interact((lambda **x: update_edges(method = "Sobel", **x)),
                  **dict(kwargs, **{"low_threshold": (0, 100, 10),
                                    "high_threshold": (200, 250, 10),
                                    "aperture_size": (1, 11, 2)}))

    elif method == "naive_gradient":
        interact((lambda **x: update_edges(method = "Sobel", **x)),
                  **dict(kwargs, **{"edge_threshold": (0, 100, 10)}))

    elif method == "scharr":
        interact((lambda **x: update_edges(method = "Sobel", **x)),
                  **dict(kwargs, **{"edge_threshold": (0, 100, 10)}))

    elif method == "beucher":
        interact((lambda **x: update_edges(method = "Sobel", **x)),
                  **dict(kwargs, **{"edge_threshold": (0, 100, 10)}))


def update_edges(method="Sobel", image="building", low_filtering=True,
                 low_filter_type="gaussian", low_filtering_kernel_size=5,
                 high_filtering=False, high_filter_type="gaussian",
                 high_filtering_kernel_size=3, strength=2.0, threshold=True,
                 block_size=11, constant=3, edge_threshold=30,
                 low_threshold=50, high_threshold=255, aperture_size=3,
                 kernel_size=3, edge_threshold_on=True,
                 low_threshold_fol=240, high_threshold_fol=255):

    if method == "Stacking":
        update_stacking(image, thresholding = edge_threshold_on, threshold = edge_threshold)
        return

    img = load_gray_img("img/{}.png".format(image))

    img_filtered = filtering(img, low_filtering, low_filter_type, low_filtering_kernel_size,
                        high_filtering, high_filter_type, high_filtering_kernel_size, strength)

    img_threshold = img_filtered
    if threshold:
        img_threshold = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, constant)

    if method == "Sobel":
        grad = sobel_edge(img_threshold, thresholding = edge_threshold_on, threshold = edge_threshold, kernel_size = kernel_size)

    elif method == "Naive Gradient":
        grad = naive_gradient(img_threshold, thresholding = edge_threshold_on, threshold = edge_threshold)

    elif method == "Scharr":
        grad = scharr_edge(img_threshold, thresholding = edge_threshold_on, threshold = edge_threshold)

    elif method == "Stacking":
        grad = stacking(img_threshold, thresholding = edge_threshold_on, threshold = edge_threshold)

    elif method == "Beucher":
        grad = beucher_edge(img_threshold, thresholding = edge_threshold_on, threshold = edge_threshold, kernel_size = kernel_size)

    elif method == "Canny":
        grad = canny_edge(img_threshold, low_threshold = low_threshold, high_threshold = high_threshold, aperture_size = aperture_size)

    elif method == "Following":
        grad = following_edge(img_threshold, thresh = low_threshold_fol, maxval = high_threshold_fol)

    elif method == "Laplacian":
        grad = laplacian_edge(img_threshold, thresholding = edge_threshold_on, threshold = edge_threshold, kernel_size = kernel_size)

    tools.multiPlot(1, 4,
            (img, img_filtered, img_threshold, grad),
            ('Original Image', 'Filtered Image', 'Thresholded Image', 'Edges'),
            cmap_tuple=(cm.gray, cm.gray, cm.gray, cm.gray, cm.gray, cm.gray))

def update_stacking(image, thresholding, threshold):
    img = load_gray_img("img/{}.png".format(image))

    grads = [get_optimal_grads(image, image, "Sobel"),
             get_optimal_grads(image, image, "Naive Gradient"),
             get_optimal_grads(image, image, "Scharr"),
             get_optimal_grads(image, image, "Beucher")]

    grad = stacking(grads, thresholding=thresholding, threshold=threshold)

    tools.multiPlot(1, 2,
            (img, grad),
            ('Original Image', 'Edges'),
            cmap_tuple=(cm.gray, cm.gray, cm.gray, cm.gray, cm.gray, cm.gray))

def build_ui_edges(default_method = "Sobel"):
    methodselect = widgets.Dropdown(options = ["Sobel", "Scharr", "Naive Gradient", "Beucher", "Canny", "Stacking", "Following", "Laplacian"], value = default_method)
    imageselect = widgets.Dropdown(options = ["building", "sudoku", "soccer", "road", "pcb"], value = "building")

    #Low Pass Filter Parameters
    lp_on = widgets.ToggleButton(value=False, description = "Apply Low-Pass filter")
    lp_type = widgets.Dropdown(options = ["uniform", "median", "gaussian"], value = "gaussian")

    lp_kernel = widgets.IntSlider(value = 3, min = 1, max = 9, step = 2, continuous_update=False)
    lp_kernel_box = widgets.HBox([widgets.Label(value="LP Kernel Size"), lp_kernel])

    lp_box = widgets.VBox([lp_on, lp_type, lp_kernel_box])

    # High pass filter parameters
    hp_on = widgets.ToggleButton(value=False, description = "Apply High-Pass filter")
    hp_type = widgets.Dropdown(options = ["uniform", "median", "gaussian"], value = "gaussian")

    hp_kernel = widgets.IntSlider(value = 3, min = 1, max = 9, step = 2, continuous_update=False)
    hp_kernel_box = widgets.HBox([widgets.Label(value="HP Kernel Size"), hp_kernel])

    hp_strenght = widgets.FloatSlider(value = 2.0, min = 1.0, max = 9.0, step = 0.5, continuous_update=False)
    hp_strenght_box = widgets.HBox([widgets.Label(value="Strength"), hp_strenght])

    hp_box = widgets.VBox([hp_on, hp_type, hp_kernel_box, hp_strenght_box])

    #Thresholding
    th_on = widgets.ToggleButton(value=False, description = "Apply Thresholding")

    th_block = widgets.IntSlider(value = 11, min = 1, max = 51, step = 2, continuous_update=False)
    th_block_box = widgets.HBox([widgets.Label(value="Thresholding Box Size"), th_block])

    th_const = widgets.IntSlider(value = 3, min = 1, max = 27, step = 2, continuous_update=False)
    th_const_box = widgets.HBox([widgets.Label(value="Thresholding Constant"), th_const])

    th_box = widgets.VBox([th_on, th_block_box, th_const_box])

    #Methods Parameters
    edge_th_on = widgets.ToggleButton(value=True, description = "Apply Thresholding to Edges")
    edge_th_str = widgets.IntSlider(value = 30, min = 1, max = 255, step = 1, continuous_update=False)
    edge_th_str_box = widgets.HBox([widgets.Label(value="Threshold"), edge_th_str])
    edge_kernel = widgets.IntSlider(value = 3, min = 1, max = 9, step = 2, continuous_update=False)
    edge_kernel_box = widgets.HBox([widgets.Label(value="Kernel Size"), edge_kernel])

    edge_low_th = widgets.IntSlider(value = 30, min = 1, max = 255, step = 1, continuous_update=False)
    edge_low_th_box = widgets.HBox([widgets.Label(value="Low Threshold"), edge_low_th])
    edge_low_fol_th = widgets.IntSlider(value = 240, min = 1, max = 255, step = 1, continuous_update=False)
    edge_low_th_fol_box = widgets.HBox([widgets.Label(value="Low Threshold"), edge_low_fol_th])
    edge_high_th = widgets.IntSlider(value = 70, min = 1, max = 255, step = 1, continuous_update=False)
    edge_high_th_box = widgets.HBox([widgets.Label(value="High Threshold"), edge_high_th])
    edge_high_fol_th = widgets.IntSlider(value = 255, min = 1, max = 255, step = 1, continuous_update=False)
    edge_high_th_fol_box = widgets.HBox([widgets.Label(value="High Threshold"), edge_high_fol_th])
    edge_aperture_size = widgets.IntSlider(value = 3, min = 3, max = 9, step = 2, continuous_update=False)
    edge_aperture_size_box = widgets.HBox([widgets.Label(value="Aperture Size"), edge_aperture_size])

    sobel_box = widgets.VBox([edge_th_on, edge_th_str_box, edge_kernel_box])
    beucher_box = widgets.VBox([edge_th_on, edge_th_str_box, edge_kernel_box])
    naive_box = widgets.VBox([edge_th_on, edge_th_str_box])
    scharr_box = widgets.VBox([edge_th_on, edge_th_str_box])
    stacking_box = widgets.VBox([edge_th_on, edge_th_str_box])
    canny_box = widgets.VBox([edge_low_th_box, edge_high_th_box, edge_aperture_size_box])
    following_box = widgets.VBox([edge_low_th_fol_box, edge_high_th_fol_box])
    laplacian_box = widgets.VBox([edge_th_on, edge_th_str_box, edge_kernel_box])

    tabnames = ['Sobel', 'Scharr', 'Naive Gradient', 'Beucher', 'Stacking', 'Canny', 'Following', 'Laplacian']
    tabs = widgets.Tab(children = [sobel_box, scharr_box, naive_box, beucher_box, stacking_box, canny_box, following_box, laplacian_box])
    for i in range(len(tabnames)):
        tabs.set_title(i, tabnames[i])


    accordion = widgets.Accordion(children=[lp_box, hp_box, th_box, tabs])
    accordion.set_title(0, 'Low Pass Filter')
    accordion.set_title(1, 'High Pass Filter')
    accordion.set_title(2, 'Thresholding')
    accordion.set_title(3, 'Method Parameters')


    ui = widgets.VBox([methodselect, imageselect, accordion])


    out = widgets.interactive_output(update_edges, {'method': methodselect,
                                                    'image': imageselect,
                                                    'low_filtering': lp_on,
                                                    'low_filter_type': lp_type,
                                                    'low_filtering_kernel_size': lp_kernel,
                                                    'high_filtering': hp_on,
                                                    'high_filter_type': hp_type,
                                                    'high_filtering_kernel_size': hp_kernel,
                                                    'strength': hp_strenght,
                                                    'threshold': th_on,
                                                    'block_size': th_block,
                                                    'constant': th_const,
                                                    'edge_threshold_on': edge_th_on,
                                                    'edge_threshold': edge_th_str,
                                                    'low_threshold': edge_low_th,
                                                    'high_threshold': edge_high_th,
                                                    'aperture_size': edge_aperture_size,
                                                    'kernel_size': edge_kernel,
                                                    'low_threshold_fol' : edge_low_fol_th,
                                                    'high_threshold_fol' : edge_high_fol_th,
                                                })

    return (ui, out)

# def update_lines(image, method, line_method)
#     img = load_gray_img(image)

#     edges = get_optimal_grads(image, method)

#     # Apply Hough method
#     if(line_method == "Hough")
#         lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
#     else:
#         lines = hough_probabilist(edge_img_0, img, 1, 0.015, 50, 30, 5)

#     # Draw image wiht lines foud with Hough
#     img_w_lines = np.zeros(edges.shape)
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(img_w_lines, (x1, y1), (x2, y2), (255, 0, 0), 1)  


#     final = get_edges_on_lines(edges, img_w_lines)

#     tools.multiPlot(1, 4, 
#             (img, edges, final, edges-final),
#             ('Edges', 'Lines detected', 'Only Edges on Lines', 'Only Edges not on Lines'),
#             cmap_tuple=(cm.gray, cm.gray, cm.gray, cm.gray))