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

import numpy as np
import cv2


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
        [0,maxHeight]])
    
    M = cv2.getPerspectiveTransform(rect, dst)
    wraped = cv2.warpPerspective(image, M, (maxWidth, maxHeight)) 

    return wraped       
    
    