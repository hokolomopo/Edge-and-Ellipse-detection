from edge import *
from image import *

import cv2

def hough(img, gray):
	edges = sobel_edge(gray)
	#edges = cv2.Canny(gray, 240, 250, None, 3)
	display_img(edges)
	# This returns an array of r and theta values 
	lines = cv2.HoughLines(edges,1,np.pi/180, 220) 
	  
	# The below for loop runs till r and theta values  
	# are in the range of the 2d array 
	for line in lines: 
		
		r = line[0][0]
		theta = line[0][1]

		# Stores the value of cos(theta) in a 
		a = np.cos(theta) 
	  
		# Stores the value of sin(theta) in b 
		b = np.sin(theta) 
		  
		# x0 stores the value rcos(theta) 
		x0 = a*r 

		# y0 stores the value rsin(theta) 
		y0 = b*r 
		  
		# x1 stores the rounded off value of (rcos(theta)-1000sin(theta)) 
		x1 = int(x0 + 1000*(-b))
		  
		# y1 stores the rounded off value of (rsin(theta)+1000cos(theta)) 
		y1 = int(y0 + 1000*(a)) 
	  
		# x2 stores the rounded off value of (rcos(theta)+1000sin(theta)) 
		x2 = int(x0 - 1000*(-b)) 
		  
		# y2 stores the rounded off value of (rsin(theta)-1000cos(theta)) 
		y2 = int(y0 - 1000*(a)) 
		  
		# cv2.line draws a line in img from the point(x1,y1) to (x2,y2). 
		# (0,0,255) denotes the colour of the line to be  
		#drawn. In this case, it is red.
		cv2.line(img,(x1,y1), (x2,y2), (0,0,255),1)

	return img

def lsd(img):

	#Create default parametrization LSD
	lsd = cv2.createLineSegmentDetector(0)

	#Detect lines in the image
	lines = lsd.detect(img)[0] #Position 0 of the returned tuple are the detected lines
	print(lines[0][0][0])

	ver_lines = []

	for line in lines:
	    angletan = math.degrees(math.atan2((round(line[0][3],2) - round(line[0][1],2)), (round(line[0][2],2) - round(line[0][0],2))))

	    if(angletan > 85 and angletan < 95):
	        ver_lines.append(line)

	#Draw detected lines in the image
	drawn_img = lsd.drawSegments(img,np.array(ver_lines))

	return drawn_img