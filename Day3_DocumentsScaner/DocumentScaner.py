# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:19:01 2020

@author: dorem

func ref : https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
@ order_point 
@ four_point_transform 

document scaner ref : https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/
@ 
"""
from skimage.filters import threshold_local
import argparse
import numpy as np
import cv2
import imutils

isUseArgParse=False

def order_points(points):
    # inittialize a list of coordinate that will be ordered 
    # such that : (order in clockwise)
    # the first entry is the top-left 
    # the second entry is the top-right 
    # the third entry is the bottom-right
    # the fourth entry is the bottom-left
    
    #np.zeros : 返回一個指定形狀的零矩陣
    #zeros(shape, dtype=float, order='C')
    rect = np.zeros((4,2), dtype = "float32")
    #[[0. 0]
    # [0. 0]
    # [0. 0]
    # [0. 0]]
    
    #the top-left point will have the smallest sum,
    #whereas the bottom-right point will have the largest sum
    
    #這邊使用的函數是np.sum()
    #np.sum 一共有兩種模式，一種是axis = 0 表示按列相加
    #axis = 1表示按行相加
    #所以底下用axis=1就是要找出哪個座標的x分量與y分量加起來最大最小
    s = points.sum(axis = 1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    
    #now compute the difference between the points,
    #the top-right point will have the smallest diffenece
    #the bottom-right point will have the largest diffencene 
    #axis 1 => y-x
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    
    #return the ordered coordinates
    return rect

def four_point_transform(image, points):
    #obtain a onsistent order of the points and unpack them
    #indivisually
    rect = order_points(points);
    (tl, tr, br, bl) = rect
    
    #compute the width of new image , which will be the 
    #maximum distance between bottom-right and bottom -left 
    #x-coordinates or the top-right and top-left
    widthA = np.sqrt(((bl[0] - br[0])**2) + ((bl[1]-br[1])**2))
    widthB = np.sqrt(((tl[0] - tr[0])**2) + ((tl[1]-tr[1])**2))
    maxWidth = max(int(widthA), int(widthB))
    
    #compute the height of the new image 
    #which will be the maximun distance between the top-right
    #and the bottom right y-coordinate or the bottom-left y coordinate
    heightA = np.sqrt(((bl[0] - tl[0])**2) + ((bl[1]-tl[1])**2))
    heightB = np.sqrt(((br[0] - tr[0])**2) + ((br[1]-tr[1])**2))
    maxHeight = max(int(widthA), int(widthB))
    
    dst = np.array([
        [0,0],
        [maxWidth, 0],
        [maxWidth, maxHeight],
        [0,maxHeight]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    wraped = cv2.warpPerspective(image, M, (maxWidth, maxHeight)) 

    return wraped       
    
def main():
    
    # deal with image file path 
    if isUseArgParse :
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", required = True, help = "path to the image to be scanned")
        args = vars(ap.parse_args())
        imagePath = args["image"]
    else:
        imagePath = "image.jpg"
    
    image = cv2.imread(imagePath)
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)
    
    # step1 edge detection 
    # flow : read image -> compute ratio
    # image processing 
    # 1. turn it into grayscale (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    # 2. blur the grayscale image (cv2.GaussianBlur)
    # 3. find edge (cv2.canny)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    cv2.waitKey(0)
    
    gray = cv2.GaussianBlur(gray, (7,7), 0)
    cv2.imshow("blur", gray)
    cv2.waitKey(0)
    
    edge = cv2.Canny(gray, 75, 200)
    cv2.imshow("edge", edge)
    cv2.waitKey(0)
    
    # step 2 find contours
    # the ref about cv.findContours()
    # link :　https://blog.csdn.net/hjxu2016/article/details/77833336
    cnts = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    #loop over the contours 
    for c in cnts:
        #approximate the controus
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri ,True)
        
        #if our approximated contour has four points, then
        #we can assume that we have found our documanet
        if len(approx) == 4:
            screenCnt = approx
            break
        
    #show the contours (outline of the piece of paper)
    cv2.drawContours(image, [screenCnt], -1, (0,255,0), 2)
    cv2.imshow("outline", image)
    cv2.waitKey(0)
    
    
    #step3 apply a perspective transform & threshold
    warped = four_point_transform(orig, screenCnt.reshape(4,2) * ratio)
    
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset=10, method="gaussian")
    warped = (warped > T).astype("uint8") * 255
    cv2.imshow("sacnned", imutils.resize(warped, height=650))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
main()
