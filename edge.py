import cv2
import numpy as np
import tools
from image import *

def saturate_cast_uint8(img):
	"""
	Taken from practical sessions

	Saturate an image (make the pixels between 0 and 255)

	Parameters
    ----------
	- img: The image to saturate

	Returns
    -------
    The saturated image
	"""
	return np.where(img > 255.0, 255.0,
					np.where( img < 0.0, 0.0, img)).astype(np.uint8)

def clean_grad(grad, saturate, threshold):
	"""
	Clean the gradients by either saturating those (make them be between 0 and
	255) or thresholding those or both.

    Parameters
    ----------
    - saturate:  A boolean indicating whether to saturate the gradients.
    - threshold: A boolean indicating whether to threshold the gradients.

    Returns
    -------
    The cleaned gradients
    """
	if saturate:
		grad = saturate_cast_uint8(grad)

	if threshold is not None:
		grad = np.where(grad > threshold, 255., 0.).astype(np.uint8)

	return grad

def low_pass_filter(img, kernel_size, filter_type):
	"""
	Apply a low pass filter on an image.

	Parameters
    ----------
    - kernel_size: The kernel size of the filter
    - filter_type: The type of filter to apply may be either "uniform",
    			   "median" or "gaussian".

    Returns
    -------
    The low-pass filtered image
	"""
	if filter_type == "uniform":
		ka = np.ones((kernel_size, kernel_size), dtype=float) / 80
		lowPass = cv2.filter2D(img, -1, ka, borderType=cv2.BORDER_CONSTANT)
	if filter_type == "median":
		lowPass = cv2.medianBlur(img, kernel_size)
	elif filter_type == "gaussian":
		lowPass = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

	return lowPass

def high_pass_filter(img, kernel_size, strength, filter_type):
	"""
	Taken from practical sessions

	Apply a high pass filter on an image

	Parameters
    ----------
    - img: 		   The image on which to apply a high pass filter
    - kernel_size: The kernel size of the filter
    - strength:    The strength of the filter
    - filter_type: The type of the filter

    Returns
    -------
    The high-pass filtered image
	"""
	lowPass = low_pass_filter(img, kernel_size, filter_type)
	return saturate_cast_uint8(strength * img - ( strength - 1.0) * lowPass)

def filtering(img, low_filtering=False, low_filter_type=None,
			  low_filtering_kernel_size=None, high_filtering=False,
			  high_filter_type=None, high_filtering_kernel_size=None,
			  high_filtering_strength=None):

	"""
	Filter an image by apply optionnally a low pass filter followed by a high
	pass filter

	Parameters
    ----------
    - img: 		   			   The image on which to apply the filter
    - low_filtering: 		   A boolean indicating whether to apply a low pass
    						   filter
    - low_filter_type: 		   The type of low pass filtering to apply, can be
    						   either "uniform", "median" or "gaussian".
    - low_filter_kernel_size:  The kernel size of the low pass filter
    - high_filtering: 		   A boolean indicating whether to apply a high pass
    						   filter
   	- high_filter_type: 	   The type of high pass filtering to apply, can be
   							   either "uniform", "median" or "gaussian".
    - ligh_filter_kernel_size: The kernel size of the high pass filter
    - high_filtering_strength: The strength of the high pass filter

    Returns
    -------
    The filtered image
	"""
	if low_filtering:
		img = low_pass_filter(img, low_filtering_kernel_size, low_filter_type)

	if high_filtering:
		img = high_pass_filter(img, high_filtering_kernel_size, high_filtering_strength, high_filter_type)

	return img

def sobel_edge(img, thresholding=True, threshold=15, kernel_size=1):
	"""
	https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/
	sobel_derivatives/sobel_derivatives.html

	Compute the edges of an image using the sobel method

	Parameters
    ----------
    - img: 		   			   The image on which to apply the filter
	- thresholding:			   Boolean for wether or not a thresholing is applied to the edges
    - threshold: 			   The threshold at which to consider a pixel as an
    						   edge
    - kernel_size:			   The size of the sobel kernel.

    Returns
    -------
    A opencv image with the pixel corresponding to edges set to 255 and the
    others set to 0
	"""
	gradientX = cv2.Sobel(img, -1, 1, 0, kernel_size)
	gradientY = cv2.Sobel(img, -1, 0, 1, kernel_size)

	absGradientX = cv2.convertScaleAbs(gradientX)
	absGradientY = cv2.convertScaleAbs(gradientY)

	grad = cv2.addWeighted(absGradientX, 0.5, absGradientY, 0.5, 0)

	if thresholding:
		grad = clean_grad(grad, False, threshold)

	return grad

def naive_gradient(img, thresholding=True, threshold=16):
	"""
	Compute the edges of an image computing gradients of the image.

	Parameters
    ----------
    - img: 		   			   The image on which to apply the filter
	- thresholding:			   Boolean for wether or not a thresholing is applied to the edges
    - threshold: 			   The threshold at which to consider a pixel as an
    						   edge

    Returns
    -------
    A opencv image with the pixel corresponding to edges set to 255 and the
    others set to 0
	"""
	xFilter = np.array([[-1, 0, 1]])
	yFilter = np.array([[-1], [0], [1]])

	gradientX = cv2.filter2D(img, cv2.CV_64F, xFilter)
	gradientY = cv2.filter2D(img, cv2.CV_64F, yFilter)

	absGradientX = cv2.convertScaleAbs(gradientX)
	absGradientY = cv2.convertScaleAbs(gradientY)

	grad = cv2.addWeighted(absGradientX, 0.5, absGradientY, 0.5, 0)

	if thresholding:
		grad = clean_grad(grad, False, threshold)

	return grad

def scharr_edge(img, thresholding = True, threshold=64):
	"""
	Compute the edges of an image using the Scharr method.

	Parameters
    ----------
    - img: 		   			   The image on which to apply the filter
	- thresholding:			   Boolean for wether or not a thresholing is applied to the edges
    - threshold: 			   The threshold at which to consider a pixel as an
    						   edge
    Returns
    -------
    A opencv image with the pixel corresponding to edges set to 255 and the
    others set to 0
	"""
	gradientX = cv2.Scharr(img, cv2.CV_32F, 1, 0)/4.0
	gradientY = cv2.Scharr(img, cv2.CV_32F, 0, 1)/4.0

	absGradientX = cv2.convertScaleAbs(gradientX)
	absGradientY = cv2.convertScaleAbs(gradientY)

	grad = cv2.addWeighted(absGradientX, 0.5, absGradientY, 0.5, 0)

	if thresholding:
		grad = clean_grad(grad, False, threshold)

	return grad

def canny_edge(img, low_threshold=50, high_threshold=255, aperture_size=3):
	"""
	Compute the edges of an image using the Canny method.

	Parameters
    ----------

    - img: 		   			   The image on which to apply the filter
    - aperture_size:		   The aperture size of the sobel operator

    Returns
    -------
    A opencv image with the pixel corresponding to edges set to 255 and the
    others set to 0
	"""
	return cv2.Canny(img, low_threshold, high_threshold, None, aperture_size)

def stacking(img, thresholding = False, threshold = 128):
	"""
	"""
	naiveGrad = naive_gradient(img)
	sobel = sobel_edge(img)
	scharr = scharr_edge(img)

	grad = cv2.addWeighted(naiveGrad, 1/3, sobel, 1/3, 0)
	grad = cv2.addWeighted(grad, 1., scharr, 1/3, 0)

	if thresholding:
		grad = clean_grad(grad, False, threshold)

	return grad

def beucher_edge(img, thresholding = True, threshold = 60):
	"""
	Compute the edges of an image using the Canny method.

	Parameters
    ----------
    - img: 		   			   The image on which to apply the filter
	- thresholding:			   Boolean for wether or not a thresholing is
							   applied to the edges
    - aperture_size:		   The aperture size of the sobel operator

    Returns
    -------

    A opencv image with the edges of the original image
	"""
	kernel = np.ones((3,3),np.uint8)
	grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

	if thresholding:
		grad = clean_grad(grad, False, threshold)

	return grad
