# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:33:21 2020

@author: dorem
"""

from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


def main():
    # Step0  File preprocessing 
    filePath = "image.png"
    if isUseArgParser :
        ap = argparse.ArggumentParser();
        ap.add_argument("-i", "--image", required=true, help="path to the input image");
        args = var(ap.parse_args())
        
    
    #define the answer key which maps the question number 
    #to the correct answer
    # 以下的語法為Python 的字典(Dictionary)
    # Question #1:B
    # Question #2:E
    # Question #3:A
    # Question #4:D
    # Question #5:B
    ANSWER_KEY = { 0: 1, 1: 4, 2: 0, 3:3, 4: 1}
    
    #Step1 image processing 
    # read img -> 
    # convert to gray scale -> 
    # perform gaussian blur -> 
    # detect edge (canny)
    image = cv2.imread(filePath)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale, (5,5), 0)
    edge = cv2.Canny(blur, 75, 200)
    
    #Step2 find document contours
    # find contour -> grab contour -> sort contour
    # find contours in the edge map, then initialize
    # the contour that corresponds to the document 
    cnts = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    #find document contour
    docCnt = None
    #ensure that at least one contour was found 
    if len(cnts) > 0:
        # sort the contour according to their size 
        # in descending orger
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        
        for c in cnts:
            #返回輪廓的周長
            peri = cv2.arcLength(c, True)
            #對Contour做多邊形近似, 並返回近似後的座標集合
            approx = cv2.approxPolyDP(c, 0.02*peri, True)
            
            if len(approx) == 4:
                docCnt = approx;
                break
    
    #cv2.drawContours(image, [doctCnt], -1, (255,0,0), 3)
    #cv2.imshow("image", image)
    
    paper = four_point_transform(image, docCnt.reshape(4,2))
    warped = four_point_transform(grayscale, docCnt.reshape(4,2))
    cv2.imshow("paper", paper)
    cv2.imshow("gray", warped)
    cv2.waitKey(0)
    
    # Step3 : binarization 
    # process of thresholding/segmenting the forgraound form 
    # the background of the image
    # 底下這行不曉得在幹嘛 : 關鍵字-二值化
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]    
    cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)
    
    #Step4 find buble sheet contour 
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts=[]
    
    
    # loop over the contours
    if len(cnts) > 0:
        for c in cnts:
            # compute the bounding box of the contour , 
            # then use the bounding box to derive the aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            #ar = aspect ratio
            ar = w / float(h)
            
            #in order to label the contour as a question, 
            # region should be sufficient wide, sufficiently,
            # and have an aspect ratio approximately  equal to 1
            if w >= 20 and h >=20 and ar >= 0.9 and ar <= 1.1:
                questionCnts.append(c)
                #cv2.drawContours(paper, [c], -1, (255,0,0),3)
                #cv2.imshow("answer select", paper)
                #cv2.waitKey(0)
       
    cv2.destroyAllWindows()
    
######################################################
isUseArgParser = False;
main()
