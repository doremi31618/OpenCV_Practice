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

directUseFilePath = True;

def main():
    # Step0  File preprocessing 
    if directUseFilePath :
        ap = argparse.ArggumentParser();
        ap.add_argument("-i", "--image", required=true, help="path to the input image");
        