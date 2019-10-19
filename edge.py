import cv2
import numpy as np

def sobel_edge(img):
	"""
	https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
	TODO: Voir convention calcul des gradients (si ca se met sur la case haut, bas, gauche, droite ...) et indiquer dans le rapport
	"""
	gradientX = cv2.Sobel(img, -1, 1, 0, 1)
	gradientY = cv2.Sobel(img, -1, 0, 1, 1)

	absGradientX = cv2.convertScaleAbs(gradientX)
	absGradientY = cv2.convertScaleAbs(gradientY)

	grad = cv2.addWeighted(absGradientX, 0.5, absGradientY, 0.5, 0)
	
	return grad

	
def high_pass(img, strength=3.0, kernel_size=9):
	"""copié collé du cours"""
	img_avg = cv2.GaussianBlur(img, ( kernel_size, kernel_size), 0)

	sharpened = tools.saturate_cast_uint8( strength * img - ( strength - 1.0) * img_avg)
	return sharpened

def beucher(img):
	"""beucher gradient (non linéaire)"""
	kernel = np.ones((3,3),np.uint8)
	grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

	return grad
