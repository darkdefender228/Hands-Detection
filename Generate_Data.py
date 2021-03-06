#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: romanilechko
"""
import imtools as it
from numpy.random import random_integers
import cv2

PATH_FOR_RIGHT = "RightHand50x50/27730/"
PATH_FOR_LEFT = "LeftHand50x50/27731/"
PATH_FOR_BAD = "Bad50x50/27733/"

def hands_generator(imlist, save_path, hand):
    for index, imageSource in enumerate(imlist):
        img = cv2.imread(imageSource)
        vertical_img = img.copy()
        vertical_img = cv2.flip(img, 1)
        cv2.imwrite(save_path + hand + str(index) + ".jpg",vertical_img)
        
def trash_generator(imlist, save_path, bad='b', first_var=0.8, rand_var=0.7):
    """
    generating new images by choosing image and one random image reducing them and adding 
    """
    count = len(imlist) - 1 #highest limit for randoming second image
    for index, imageSource in enumerate(imlist):
        rand = random_integers(0, count)
        img = cv2.imread(imageSource)[:,:,0] #convert img from (N,M,C) to (N,M)
        img_rand = cv2.imread(imlist[rand])[:,:,0]
        
        n_features_first, n_features_rand = int(img.shape[0]*first_var), int(img_rand.shape[0]*rand_var)
        reduced_first = it.reduce_dim(n_features_first, img)
        reduced_rand = it.reduce_dim(n_features_rand, img_rand)
        genereted = reduced_first + reduced_rand
        cv2.imwrite(save_path + bad + str(index) + ".jpg",genereted)

#get all images in folder
img_right = it.get_imlist(PATH_FOR_RIGHT)
img_left = it.get_imlist(PATH_FOR_LEFT)
img_bad = it.get_imlist(PATH_FOR_BAD)

# flipping left hand images and save to right hand images 
# and conversely for right hand images
hands_generator(img_right, PATH_FOR_LEFT, "r")
hands_generator(img_left, PATH_FOR_RIGHT, "l")
trash_generator(img_bad, PATH_FOR_BAD)