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
    if numb > limits[1]:
        return limits[1]
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

def generating_boxes(dots, img, box_shape, shape, optimization=False):
    croped_image = []
    #score = [] #for evalueting

    boxes = make_square_box(dots, box_shape, shape)  #making our boxes for each dot, 
                                                     #each dot has array which includes the box coordinates  which was generated aroud this dot
    for index, dot_boxes in enumerate(boxes):        # dot_boxes - boxes generated around one dot
        dot_boxes_scores = []
        for box in dot_boxes:                        #evalueting all boxes generated from one dot and save this scores in dot_boxes_scores
            if optimization and index < dots['coordinates'].shape[0]: #(evaluate only box generated from max peek)
                dot_boxes_scores.append(evalueting_cropping(box, dots, img, dots['clusters'][0].labels_[index]))
        if optimization and len(dot_boxes_scores) > 0:
            first, second = index_two_min(dot_boxes_scores) # taking index of the boxes with the lowest scores
            values = dot_boxes[first], dot_boxes[second]    # taking these boxes
            
            for value in values:
                dot_boxes.remove(value)                     # removing bad boxes
            
        good = [crop(box, img) for box in dot_boxes]    # cropping good boxes
        croped_image.append(good)
            #score.append(dot_boxes_scores)          #saving in score, in this case, the one element from the score array it is an array 
                                                     #with scores from each box generated from one dot
    return np.asarray(croped_image)

def evalueting_cropping(box, dots, img, cluster):
    coords, clustered = dots['coordinates'], dots['clusters']
    number_self_dots = 0.000000001
    number_alien_dots = 0.000000001
    finded_clusters = []
    count_clusters = 1
    
    summa = np.sum(crop(box, img))
    for index, coord in enumerate(coords):
        if in_box(coord, box):
            cluster_ = clustered[0].labels_[index]
            if cluster != cluster_:
                number_alien_dots = number_alien_dots + 1
                if cluster_ not in finded_clusters:
                    finded_clusters.append(cluster_)
                    count_clusters = count_clusters + 1
            else:
                number_self_dots = number_self_dots + 1
    value = summa * number_self_dots / (number_alien_dots * count_clusters)
    return round(value, 3)

def in_box(coord, box):
    #check is the dot in box or not
    if box[0] < box[2]:
        y = box[0], box[2]
    else:
        y = box[2], box[0]
    if box[1] < box[3]:
        x = box[1], box[3]
    else:
        x = box[3], box[1]
    if  y[0] <= coord[0] and coord[0] <= y[1]:
        if x[0] <= coord[1]  and coord[1] <= x[1]:
            return True
    return False

def index_two_min(arr):
    if arr[0] < arr[1]:
        index_of_min, index_of_second = 0, 1
    else:
        index_of_min, index_of_second = 1, 0
    for index, element in enumerate(arr[2:]):
        if element < arr[index_of_min]:
            index_of_second = index_of_min
            index_of_min = index + 2
        elif element < arr[index_of_second]:
            index_of_second = index + 2
    return index_of_min, index_of_second