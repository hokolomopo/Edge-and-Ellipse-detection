import cv2

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