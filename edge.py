import cv2
import numpy as np
import tools

def saturate_cast_uint8(img):
	"""
	Taken from practical sessions
	"""
	return np.where( img > 255.0, 255.0, np.where( img < 0.0, 0.0, img)).astype(np.uint8)

def clean_grad(grad, saturate, threshold):
	if saturate:
		grad = saturate_cast_uint8(grad)
	
	if threshold is not None:
		grad = np.where(grad > threshold, 255., 0.).astype(np.uint8)

	return grad

def low_pass_filter(img, kernel_size, filter_type):
	if filter_type == "uniform":
		ka7 = np.ones((kernel_size, kernel_size), dtype=float) / 80
		lowPass = cv2.filter2D(img, -1, ka7, borderType=cv2.BORDER_CONSTANT)
	if filter_type == "median":
		lowPass = cv2.medianBlur(img, kernel_size)
	elif filter_type == "gaussian":
		lowPass = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
	
	return lowPass

def high_pass_filter(img, kernel_size, strength, filter_type):
	"""
	Taken from practical sessions
	"""
	
	lowPass = low_pass_filter(img, kernel_size, filter_type)
	#lowPass = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
	return saturate_cast_uint8(strength * img - ( strength - 1.0) * lowPass)

def filtering(img, low_filtering = False, low_filter_type = None, low_filtering_kernel_size = None, 
			  high_filtering = False, high_filter_type = None, high_filtering_kernel_size = None, 
			  high_filtering_strength = None):
	
	if low_filtering:
		img = low_pass_filter(img, low_filtering_kernel_size, low_filter_type)

	if high_filtering:
		img = high_pass_filter(img, high_filtering_kernel_size, high_filtering_strength, high_filter_type)

	return img

def sobel_edge(img, saturate = True, threshold = 15, low_filtering = True, low_filter_type = "uniform", 
			   low_filtering_kernel_size = 5, high_filtering = True, high_filter_type = "gaussian",  
			   high_filtering_kernel_size = 1 , high_filtering_strength = 2, kernel_size = 1):
	"""
	https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
	TODO: Voir convention calcul des gradients (si ca se met sur la case haut, bas, gauche, droite ...) et indiquer dans le rapport
	"""

	img = filtering(img, low_filtering, low_filter_type, low_filtering_kernel_size,
					high_filtering, high_filter_type, high_filtering_kernel_size, high_filtering_strength)

	gradientX = cv2.Sobel(img, -1, 1, 0, kernel_size)
	gradientY = cv2.Sobel(img, -1, 0, 1, kernel_size)

	absGradientX = cv2.convertScaleAbs(gradientX)
	absGradientY = cv2.convertScaleAbs(gradientY)

	grad = cv2.addWeighted(absGradientX, 0.5, absGradientY, 0.5, 0)

	return clean_grad(grad, saturate, threshold)

def naive_gradient(img, saturate = True, threshold = 16, low_filtering = True, low_filter_type = "uniform", 
			       low_filtering_kernel_size = 5, high_filtering = True, high_filter_type = "gaussian",  
			       high_filtering_kernel_size = 1 , high_filtering_strength = 2):
	img = filtering(img, low_filtering, low_filter_type, low_filtering_kernel_size,
					high_filtering, high_filter_type, high_filtering_kernel_size, high_filtering_strength)

	xFilter = np.array([[-1, 0, 1]])
	yFilter = np.array([[-1], [0], [1]])

	gradientX = cv2.filter2D(img, cv2.CV_64F, xFilter)
	gradientY = cv2.filter2D(img, cv2.CV_64F, yFilter)

	absGradientX = cv2.convertScaleAbs(gradientX)
	absGradientY = cv2.convertScaleAbs(gradientY)

	grad = cv2.addWeighted(absGradientX, 0.5, absGradientY, 0.5, 0)

	return clean_grad(grad, saturate, threshold)

def scharr_edge(img, saturate = True, threshold = 64, low_filtering = True, low_filter_type = "uniform", 
			    low_filtering_kernel_size = 5, high_filtering = True, high_filter_type = "gaussian",  
			    high_filtering_kernel_size = 5 , high_filtering_strength = 2):
	img = filtering(img, low_filtering, low_filter_type, low_filtering_kernel_size,
					high_filtering, high_filter_type, high_filtering_kernel_size, high_filtering_strength)

	gradientX = cv2.Scharr(img, cv2.CV_32F, 1, 0)/4.0
	gradientY = cv2.Scharr(img, cv2.CV_32F, 0, 1)/4.0

	absGradientX = cv2.convertScaleAbs(gradientX)
	absGradientY = cv2.convertScaleAbs(gradientY)

	grad = cv2.addWeighted(absGradientX, 0.5, absGradientY, 0.5, 0)

	return clean_grad(grad, saturate, threshold)

def canny_edge(img, saturate = True, low_threshold = 50, high_threshold = 255, low_filtering = True,
			   low_filter_type = "uniform", low_filtering_kernel_size = 5, high_filtering = True,
			   high_filter_type = "gaussian", high_filtering_kernel_size= 5, high_filtering_strength = 2,
			   aperture_size = 3):
	
	img = filtering(img, low_filtering, low_filter_type, low_filtering_kernel_size,
					high_filtering, high_filter_type, high_filtering_kernel_size, high_filtering_strength)

	return cv2.Canny(img, low_threshold, high_threshold, None, aperture_size)

def stacking(img, saturate = True):
	naiveGrad = naive_gradient(img)
	sobel = sobel_edge(img)
	scharr = scharr_edge(img)

	grad = cv2.addWeighted(naiveGrad, 1/3, sobel, 1/3, 0)
	grad = cv2.addWeighted(grad, 1., scharr, 1/3, 0)

	return clean_grad(grad, saturate, 128)
	
def high_pass(img, strength=3.0, kernel_size=9):
	"""copié collé du cours"""
	img_avg = cv2.GaussianBlur(img, ( kernel_size, kernel_size), 0)

	sharpened = tools.saturate_cast_uint8( strength * img - ( strength - 1.0) * img_avg)
	return sharpened

def beucher(img, low_filtering = True, low_filtering_kernel_size = 1, threshold = 60):
	"""beucher gradient (non linéaire)"""

	if low_filtering:
		img_lp = cv2.GaussianBlur( img, ( low_filtering_kernel_size, low_filtering_kernel_size), 0)

	kernel = np.ones((3,3),np.uint8)
	grad = cv2.morphologyEx(img_lp, cv2.MORPH_GRADIENT, kernel)
	grad = clean_grad(grad, False, threshold)

	return grad