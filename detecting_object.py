#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 12:57:29 2018

@author: romanilechko
"""
import imtools as it
import detectiontools as dt
from object_croptools import generating_boxes
import cv2

SCALE = 2 #for pooling
MIN_DISTANCE = 2 #radius for finding local max
MASK_VALUE = 0.3
PATH = 'Test189x110/27734/'
DISTANCE_IN_CLUSTER = 30
BOX = 53
SHAPE_TEST = (110, 189)


def preparing_img(path):
    original_img = cv2.imread(path)[:,:,0]
    img = original_img/255
    im = dt.make_mask(MASK_VALUE, img)
    return im, original_img

def main():
    found_objects = {}
    images_path = it.get_imlist(PATH)
    
    for image_path in images_path:
        img, original_img = preparing_img(image_path)
        maximum = dt.mean_pooling(img, SCALE)
        
        #finding coordinates peek_local_maxima in scaled img
        coordinates = dt.filtering_max(    
                cv2.peak_local_max(maximum, min_distance=MIN_DISTANCE), maximum, SCALE)
        rescaled_coordinates = (coordinates + MIN_DISTANCE) * SCALE
        n_clusters = dt.count_cluster(
                rescaled_coordinates[0], rescaled_coordinates[1:], DISTANCE_IN_CLUSTER)
        clusters = dt.find_cluster(
                n_clusters, rescaled_coordinates, mode='default', init='k-means++')
        generated = generating_boxes(clusters, original_img, BOX, SHAPE_TEST, optimization=True)
        found_objects[image_path] = generated
    return generated

