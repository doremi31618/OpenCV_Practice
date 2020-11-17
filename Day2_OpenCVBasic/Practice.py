# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 22:36:49 2020

@author: dorem
"""

import cv2
import imutils
import matplotlib.pyplot as plt


"""
# Loading and Display an image
"""
# load the input image and show its dimensions, keeping in mind that
# images are represented as a multi-dimensional NumPy array with
# shape no. rows (height) x no. columns (width) x no. channels (depth)
image = cv2.imread('jp.png')
(h, w, d) = image.shape
print("width={}, height={}, depth={}".format(w, h, d))

plt.imshow(image)

"""
# Accessing individual pixels
"""
# access the rgb pixel located at x=50, y=100, keeping in mind that 
# opencv store image in BGR format rather than RGB
# and the height is the number of rows and 
# the width is the number of columns 
(B, G, R) = image[100, 50]
print("R={}, G={}, B={}".format(R, G, B))

"""
# Array slicing and cropping
"""
# “regions of interest” (ROIs)

roi = image[60:160, 320:420] #from height 60 pixel to 160 pixel and width 320 pixel to 420 pixel 
plt.imshow(roi, cmap = 'viridis')

"""
# Resizing images
"""
resized = cv2.resize(image, (200,200))
plt.imshow(resized)

#but the display would be a distortion photo 
# so we have to re-calculate the aspect ratio of the original image and use it 
r = 300.0/w # the manification which is 0.5
dim = (300, int(h * r))
resized = cv2.resize(image, dim)
plt.imshow(resized)

#and here comes another solution without calcualte 
# you only need to use imtils.resize(image, width=300)
resiezed = imutils.resize(image, width=300)
plt.imshow(resized)

"""
# Rotating an images
"""
# lets rotate an image 45degrees clockwise using opencv by first
# computing the image center, then constructing the rotation matrix 
# and then finally applying the affine warp
#refence about the affine wrap : https://www.twblogs.net/a/5b7b00e42b7177539c24a869
center = (w//2, h//2) # we use "//"to perform integer math (no floating point values)
M = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated = cv2.warpAffine(image, M, (w,h))
plt.imshow(rotated)

# and still there's an easier way for rotation
rotated = imutils.rotate(image, -45)
plt.imshow(rotated)

# but you will notice that the out-boundary image is cutted and not display the whole image
rotated = imutils.rotate_bound(image, angle=45)
plt.imshow(rotated)

"""
# Smoothing an image (blur an image)
"""
# in many image processing pipline, 
# we must blur an image to reduce hight-frequency noise
blurred = cv2.GaussianBlur(image, (15,15), 0)
plt.imshow(blurred)

"""
# Drawing on an image
"""
# rectangle(image, center, radius, color, thickness)
# img : The output image.
# center : Our circle’s center coordinate. I supplied (300, 150)  which is right in front of Ellie’s eyes.
# radius : The circle radius in pixels. I provided a value of 20  pixels.
# color : Circle color. This time I went with blue as is denoted by 255 in the B and 0s in the G + R components of the BGR tuple, (255, 0, 0) .
# thickness : The line thickness. Since I supplied a negative value (-1 ), the circle is solid/filled in.

output = image.copy()
cv2.rectangle(output, (320, 60), (420, 160), (0, 0, 255), 2)
cv2.line(output, (60,20), (400,200), (0, 0, 255), 5)
plt.imshow(output)

# puText(image, text, pt, font, scale, color, thickness)
# img : The output image.
# text : The string of text we’d like to write/draw on the image.
# pt : The starting point for the text.
# font : I often use the cv2.FONT_HERSHEY_SIMPLEX . The available fonts are listed here.
# scale : Font size multiplier.
# color : Text color.
# thickness : The thickness of the stroke in pixels.
output = image.copy()
cv2.putText(output, "OpenCV + Jurassic Park!!!", (10, 25), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
plt.imshow(output)