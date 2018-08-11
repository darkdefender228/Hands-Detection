#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: romanilechko
"""
import numpy as np

def make_square_box(dots, box_shape, shape, center=False):
    """
    dots - dictionary. keys - ['coordiantes', 'clusters'] - numpy.array
    box_shape - int - weight/height of box
    shape - tuple (y, x) shape of original image
    
    return numpy.array which contains elements of type numpy.array.This element 
    contains 4 boxes (or 5 if center=True).These boxes are generated around 
    one dot. Could contain 1 box if the dot is a cluster center
    """
    y, x = shape
    shift = box_shape//4
    boxes = []
    for dot in dots['coordinates']:
    #left_up
        dot_boxes = []
        check = box(shift, dot, 'l_up')
        right_down_y = crop_number(check[0], (-99999, y))
        right_down_x = crop_number(check[1], (-99999, x))
        left_up_y = crop_number(check[2], (0, 99999))
        left_up_x = crop_number(check[3], (0, 99999))
        dot_boxes.append(np.array([left_up_y, left_up_x, right_down_y, right_down_x]))

    #right_up
        check = box(shift, dot, 'r_up')
        left_down_y = crop_number(check[0], (-99999, y))
        left_down_x = crop_number(check[1], (0, 99999))
        right_up_y = crop_number(check[2], (0, 99999))
        right_up_x = crop_number(check[3], (-99999, x))
        dot_boxes.append(np.array([right_up_y, left_down_x, left_down_y, right_up_x]))
    #left_down
        check = box(shift, dot, 'l_d')
        right_up_y = crop_number(check[0], (0, 99999))
        right_up_x = crop_number(check[1], (-99999, x))
        left_down_y = crop_number(check[2], (-99999, y))
        left_down_x = crop_number(check[3], (0, 99999))
        dot_boxes.append(np.array([left_up_y, left_up_x, right_down_y, right_down_x]))
    #right_down
        check = box(shift, dot, 'r_d')
        right_up_y = crop_number(check[0], (0, 99999))
        rigt_up_x = crop_number(check[1], (0, 99999))
        left_down_y = crop_number(check[2], (-99999, y))
        left_down_x = crop_number(check[3], (-99999, x))
        dot_boxes.append(np.array([right_up_y, left_down_x, left_down_y, rigt_up_x]))
    #center
        if center:
            check = box(box_shape//2, dot, 'center')
            right_up_y = crop_number(check[0], (0, 99999))
            rigt_up_x = crop_number(check[1], (0, 99999))
            left_down_y = crop_number(check[2], (-99999, y))
            left_down_x = crop_number(check[3], (-99999, x))
            dot_boxes.append(np.array([right_up_y, left_down_x, left_down_y, rigt_up_x]))
        boxes.append(np.asarray(dot_boxes, dtype=np.int16))

            
    for cluster in dots['clusters']:
        clusters_dots = cluster.cluster_centers_
        for cluster_dot in clusters_dots:
            cluster_dot = int(cluster_dot[0]), int(cluster_dot[1])
            check = box(box_shape//2, cluster_dot, 'center')
            right_up_y = crop_number(check[0], (0, 99999))
            rigt_up_x = crop_number(check[1], (0, 99999))
            left_down_y = crop_number(check[2], (-99999, y))
            left_down_x = crop_number(check[3], (-99999, x))
            boxes.append(np.array([[right_up_y, left_down_x, left_down_y, rigt_up_x]]))
    return np.asarray(boxes)

def crop_number(numb, limits):
    """
    numb - int
    limits - tuple where limits[0] is a left limit and limits[1] is a right limit
    use to cut coordinate of the box if they bigger than the shape of the image
    or less than 0
    """
    if numb < limits[0]:
        return limits[0]
    else:
        return numb
    if numb > limits[1]:
        return limits[1]
    else:
        return numb

def box(shift, dot, where):
    """
    shift - int
    dot - list [y, x]
    #where - vector where we shift the coordinates l_up, r_up, l_d, r_d, center
    #return -> smaller_diag_y, smaller_diag_x, bigger_diag_y, bigger_diag_x
    """
    if where == 'r_d':
        return (dot[0] - shift, dot[1] - shift, dot[0] + 3*shift, dot[1] + 3*shift)
    if where == 'l_up':
        return (dot[0] + shift, dot[1] + shift, dot[0] - 3*shift, dot[1] - 3*shift)
    if where == 'l_d':
        return (dot[0] - shift, dot[1] + shift, dot[0] + 3*shift, dot[1] - 3*shift)
    if where == 'r_up':
        return (dot[0] + shift, dot[1] - shift, dot[0] - 3*shift, dot[1] + 3*shift)
    if where == 'center':
        return (int(dot[0] - shift), int(dot[1] - shift), int(dot[0] + shift), int(dot[1] + shift))
    
def crop(box, img):
    """
    box - the coordinates for cropping image
    img - numpy.array, original image
    
    return - numpy.array, cropped image
    """
    #condition_is_true if condition else condition_is_false
    y = []
    x = []
    y.append((box[0], box[2])) if box[0] < box[2] else y.append((box[2], box[0])) #y_slicing
    x.append((box[1], box[3])) if box[1] < box[3] else x.append((box[3],box[1])) #x_slicing

    return img[y[0][0]:y[0][1], x[0][0]:x[0][1]]

def generating_boxes(dots, img, box_shape, shape):
    """
    dots - dictionary. keys - ['coordiantes', 'clusters'] - numpy.array
    img - numpy.array
    box_shape - int - weight/height of box
    shape - tuple (y, x) shape of original image
    
    generating the all boxes around dots and around the clusters centers
    
    return - numpy.array - array of cropped images
    """
    croped_image = []
    boxes = make_square_box(dots, box_shape, shape)
    for dot_boxes in boxes:
        for box in dot_boxes:
            croped_image.append(crop(box, img))
    return np.asarray(croped_image)
