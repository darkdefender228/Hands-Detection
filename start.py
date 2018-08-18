#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 00:54:51 2018

@author: romanilechko
"""
import imtools as it
from detecting_object import detecting
import cv2

PATH = 'Test189x110/27734/'

def preparing_img(path):
    original_img = cv2.imread(path)[:,:,0]
    original_img = original_img/255
    return original_img

def main():
    imlist = it.get_imlist(PATH)
    for img_path in imlist:
        original = preparing_img(img_path)
        detecting(original)
        
    