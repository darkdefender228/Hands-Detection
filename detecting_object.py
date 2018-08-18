#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: romanilechko
"""
import detectiontools as dt
import object_croptools as ct
import prediction as pr
from hands_classificator import model
from skimage.feature import peak_local_max


SCALE = 2 #for pooling
MIN_DISTANCE = 2 #radius for finding local max
MASK_VALUE = 0.3
DISTANCE_IN_CLUSTER = 30
BOX = 53
SHAPE_TEST = (110, 189)

def detecting(original):
    img = original
    img = dt.make_mask(MASK_VALUE, img)
    polled = dt.mean_pooling(img, SCALE)
    peeks_scaled = dt.filtering_max(
            peak_local_max(polled, min_distance=MIN_DISTANCE), polled, SCALE)
    peeks = (peeks_scaled + MIN_DISTANCE) * SCALE #rescaled peeks
    n_clusters = dt.count_cluster(peeks[0], peeks[1:], DISTANCE_IN_CLUSTER)
    
    clusters = dt.find_cluster(n_clusters, peeks, mode='default', init='k-means++')
    print("anticipated number of clusters ", n_clusters)
    generated = ct.generating_boxes(clusters, original, BOX, SHAPE_TEST, optimization=True)
    pr.predicting(generated, clusters, n_clusters, original, model)

