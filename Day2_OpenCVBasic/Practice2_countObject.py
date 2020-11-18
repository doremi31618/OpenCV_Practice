# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 12:47:04 2020

@author: dorem
"""

import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
"""
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",type=str,required=True, help="path to input image")
args = ap.parse_args()
image = cv2.imread(args.image)
"""

image = cv2.imread("tetris_blocks.png")
cv2.imshow("origainal",image)
cv2.waitKey(0)

'''
Converting an image to grayscale
'''

#opencv中颜色空间转换函数 cv2.cvtColor()
#https://blog.csdn.net/u012193416/article/details/79312798
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
cv2.imshow("gray",gray)
cv2.waitKey(0)

'''
Edge detection
'''

# 解決cv2.imshow最好的方法就是用coda forge 安裝 opencv
# cv2.Canny(img, minVal, maxVal)
edged = cv2.Canny(gray, 30 , 150)
cv2.imshow("Edged", edged)
cv2.waitKey(0) 

'''
Thresolding
'''
#Image thresholding is an important intermediary step for 
#image processing pipline. Thresholding can help us 
#to remove lighter or darker redions and contours of image 
_, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

'''
Detecting and draw contours


'''
# find contours of the foreground object in the 
# threshld image
# threshold the image by setting all pixel values less than 225
# to 255 (white; foreground) and all pixel values >= 225 to 255
# (black; background), thereby segmenting the image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = image.copy()


for c in cnts:
    cv2.drawContours(output, [c], -1, (240,0,159),3)
    cv2.imshow("Contours", output)
    cv2.waitKey(0)
    
text = "I found {} objects!".format(len(cnts))
cv2.putText(output, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240,0,159),2)
cv2.imshow("Contours", output)
cv2.waitKey(0)

'''
Eroding and dilations

reference : https://shengyu7697.github.io/blog/2020/06/21/Python-OpenCV-erode-dilate/

影像侵蝕的概念:
    就是將影像中白色區域(或高亮)進行細化或縮減，運算完的結果圖比原圖的白色區域更小，也可想像成讓該物體瘦一圈，而這一圈的寬度是由捲積 kernel 的大小所決定的，
    實際上捲積 kernel 沿著影樣滑動並計算，如果捲積 kernel m x n 範圍內所有像素值都是1，那麼新的像素值就保持原來的值，
    否則新的像素值為0，這表示捲積 kernel 掃過的所有像素都會被腐蝕或侵蝕掉(變為0)，所以整張影像的白色區域會變少。

影像膨脹的概念:
    就是將影像中白色區域(或高亮)進行擴張，運算完的結果圖比原圖的白色區域更大，也可想像成讓該物體胖一圈，而這一圈的寬度是由捲積 kernel 的大小所決定的，
'''
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=5)
cv2.imshow("Mask", mask)
cv2.waitKey(0)

mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=5)
cv2.imshow("Dilate", mask)
cv2.waitKey(0)

'''
Masking and bitwise operation
'''
mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Output", output)
cv2.waitKey(0)

