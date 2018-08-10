#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:12:28 2018

@author: romanilechko
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def show(img, title='my picture'):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()
    
def make_mask(value, img):
    """
    seting the mask on image
    value - float
    img - numpy.ndarray
    """
    mask = img < value
    img[mask] = 0
    return img

def centre(coord, dist):
    centre = []
    for index in range(len(coord[:-1])):
        if np.sum((coord[index] - coord[index + 1])**2)**0.5 < dist:
            centre.append(np.abs(coord[index] + coord[index + 1])//2)
        else:
            centre.append(coord[index])
    return np.asarray(centre)

def filtering_max(coordinates, maximum, scale, value=0.05):
    """
    filtering local maxima of image
    coordinates - numpy.ndarray
    maximum - masked image of type numpy.ndarray
    scale - float
    value - float
    """
    filtered = maximum[coordinates[:, 0], coordinates[:, 1]] > value
    coordinates = coordinates[filtered]
    return coordinates

def mean_pooling(img, scale):
    """
    img - numpy.ndarray
    scale - float
    """
    h_, w_ = 0, 0
    save_mean = []
    for h in range(10, 110, scale):
        h_ = h_ + 1
        for w in range(10, 189, scale):
            save_mean.append(img[h - 10: h, w - 10: w].mean())
    w_ = len(save_mean) // h_
    save_mean = np.asarray(save_mean).reshape(h_, w_)
    return save_mean

def count_cluster(first, dots, radius):
    """
    recirsive fuction for finding numbers of cluster
    first - coordinates of first dot of type numpy.ndarray
    dots - coordinates of dots of type numpy.ndarray
    rdius - float
    
    Find the euclidean distance between first and all other dots, 
    filtered them by radius. After that, function do recursive call 
    for dots which distances more than the radius. 
    Depth of recursion = number of clusters
    """
    lst = []
    first = np.ones(dots.shape)*first
    dist = np.ones(dots.shape[0])
    arr = (first - dots)**2
    
    for index, el in enumerate(arr):
        dist[index] = np.sqrt(np.sum(el))
    index = (dist > radius)
    for j, i in enumerate(index):
        if i:
            lst.append(dots[j])
    if len(lst) > 0:
        return count_cluster(lst[0], np.asarray(lst[1:]), radius) + 1
    else:
        return 1

def find_cluster(n_clusters, coordinates, mode='default', init='k-means++'):
    """
    finds clusters and return a dictionary with keys 'clusters_centres', 'coordinates'
    where 'clusters_centres' are coordinates of clusters centres, 
    'coordinates' are coordinates of local peek maxima
    
    n_clusters - float
    coordinates - numpy.ndarray
    mode - str - is optional, if the mode is +- finds centers for number of clusters  
    [n_clusters - 1, n_clusters, n_clusters + 1] 
    in that case dots['clusters_centres'].shape = (3,)
    init - str - init for Kmeans
    """
    dots = {}
    if mode == 'default':
        kmeans_centers = KMeans(n_clusters=n_clusters, random_state=0, init='k-means++').fit(coordinates).cluster_centers_
    if mode == '+-':
        kmeans_centers = [KMeans(n_clusters=i, random_state=0, init='k-means++').fit(coordinates).cluster_centers_ for i in range(n_clusters - 1, n_clusters + 2)]
        
    dots['clusters_centers'] = np.asarray(kmeans_centers)
    dots['coordinates'] = coordinates
    return dots