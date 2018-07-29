#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: romanilechko
"""
import imtools as it
import cv2
PATH_FOR_RIGHT = "RightHand50x50/27730/"
PATH_FOR_LEFT = "LeftHand50x50/27731/"

def generator(imlist, save_path, hand):
    for index, imageSource in enumerate(imlist):
        img = cv2.imread(imageSource)
        vertical_img = img.copy()
        vertical_img = cv2.flip(img, 1)
        cv2.imwrite(save_path + hand + str(index) + ".jpg",vertical_img)

#get all images in folder
img_right = it.get_imlist(PATH_FOR_RIGHT)
img_left = it.get_imlist(PATH_FOR_LEFT)

# flipping left hand images and save to right hand images 
# and conversely for right hand images
generator(img_right, PATH_FOR_LEFT, "r")
generator(img_left, PATH_FOR_RIGHT, "l")
