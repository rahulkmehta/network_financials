############################################################################
# File Name: objectvision.py                                               #
#                                                                          #
# Developer: Rahul Mehta                                                   #
#                                                                          #
# Designer: Debi Prasad Sahoo, Anshul Prakash Deshkar, Rahul Mehta         #
#                                                                          #
# (c)2016-2020 Copyright Protected,NetworkFinancials Inc.,San Jose(CA),USA #
#                                                                          #
############################################################################

#[METHODOLOGY DRAWN FROM OPEN-SOURCES & PAPERS LISTED IN ACKNOWLEDGEMENTS. IMPROVEMENTS & CHANGES ALSO LISTED IN README.md]

#[PACKAGE IMPORTS]

from __future__ import division
import cv2
import decimal
import operator
import os, glob
import numpy as np
from numpy import array, linspace
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.signal import argrelextrema, argrelmax, find_peaks
import math
import pickle
import pylab as p

#[PRE-PROCESSING DIRECTIVES]
cwd = os.getcwd()

#[HELPER FUNCTION TO DISPLAY IMAGES BEFORE/AFTER PROCESSING]
def display_images(images, cmap = None):
    cols = 2
    rows = (len(images) + 1)//cols
    plt.figure(figsize = (15, 12))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i+1)
        if len(image.shape) == 2:
            cmap = 'gray'
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad = 0, h_pad = 0, w_pad = 0)
    plt.show()

def filter_rgb_and_edge_detection(image, lt = 50, ht = 200): 
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 137, 255, cv2.THRESH_BINARY)
    return cv2.Canny(blackAndWhiteImage, lt, ht)

def filter(image, vertices):
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
    return cv2.bitwise_and(image, mask)

def select_region(image):
    rows, cols = image.shape[:2]
    pt_1  = [cols*0.05, rows*0.90]
    pt_2 = [cols*0.05, rows*0.70]
    pt_3 = [cols*0.30, rows*0.55]
    pt_4 = [cols*0.6, rows*0.15]
    pt_5 = [cols*0.90, rows*0.15] 
    pt_6 = [cols*0.90, rows*0.90]
    vertices = np.array([[pt_1, pt_2, pt_3, pt_4, pt_5, pt_6]], dtype=np.int32)
    return filter(image, vertices)

#[NOTE: THE EDGE DETECTION NEEDED TO BE DONE BEFORE THE HOUGH TRANSFORMATION OTHERWISE THE EFFICIENCY SUFFERS GREATLY.]
#[NOTE: NOISY IMAGES DO NOT WORK AS WELL WITH HOUGH TRANSFORMATIONS]
def hough_transformation(image):
    return cv2.HoughLinesP(image, rho=0.1, theta=np.pi/10, threshold = 15, minLineLength = 7, maxLineGap = 5) #[MLG WAS PREVIOUSLY 4]
 
def draw_hough_transformation(image, lines, color = [255, 0, 0], thickness = 2):
    image = np.copy(image)
    cleaned = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if abs(y2-y1) <=1 and abs(x2-x1) >=25 and abs(x2-x1) <= 55:
                cleaned.append((x1,y1,x2,y2))
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

#[COMPILE LIST OF LINE COORDINATES]
def parse_datapoints(lines):
    datapoints = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            average = decimal.Decimal((x1+x2)/2)
            datapoints.append(average)
    return datapoints

#[FIND LOCAL MAXIMA IN HISTOGRAM WHICH SIGNAL A PARKING LANE]
def find_clusters(datapoints):
    #[HISTOGRAM CREATION]
    npdata = np.array(datapoints, dtype=float)
    w = 10
    n = math.ceil((npdata.max() - npdata.min())/w)
    figure = np.histogram(npdata, bins = n)
    hData = figure[0]
    peaks = argrelextrema(hData, np.greater, order=3)

    #[NOW THAT THE TOP HISTOGRAM VALUES HAVE BEEN FOUND, FIND THE X-VALUES]
    xclusters = []
    for val in peaks:
        for item in val:
            xclusters.append(figure[1][item])

    #[RETURN VALUES]
    return xclusters
            
    
def draw_hough_transformation_withxclustering(image, lines, color=[0, 0, 255], thickness=2, make_copy=True):
    new_image = np.copy(image) # don't want to modify the original
    cleaned = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if abs(y2-y1) <=1 and abs(x2-x1) >=25 and abs(x2-x1) <= 55:
                cleaned.append((x1,y1,x2,y2))
    xlist = sorted(cleaned, key=operator.itemgetter(0,1))
    clusters = {}
    dIndex = 0
    
    #[NOTICE: FOR NOW, THIS IS A HARD-CODED VALUE -- MUST BE CHANGED TO MAKE ALGO ADAPTABLE TO DIFFERING SCOPES]
    clust_dist = 13
    #[END NOTICE]

    for i in range(len(xlist)-1):
        distance = abs(xlist[i+1][0]-xlist[i][0])
        if distance <= clust_dist:
            if not dIndex in clusters.keys(): 
                clusters[dIndex] = []
            clusters[dIndex].append(xlist[i])
            clusters[dIndex].append(xlist[i+1])
        else:
            dIndex += 1
    rects = {}
    i = 0
    for key in clusters:
        all_list = clusters[key]
        cleaned = list(set(all_list))
        if len(cleaned) > 5:
            cleaned = sorted(cleaned, key=lambda tup: tup[1])
            avg_y1 = cleaned[0][1]
            avg_y2 = cleaned[-1][1]
            avg_x1 = 0
            avg_x2 = 0
            for tup in cleaned:
                avg_x1 += tup[0]
                avg_x2 += tup[2]
            avg_x1 = avg_x1/len(cleaned)
            avg_x2 = avg_x2/len(cleaned)
            rects[i] = (avg_x1, avg_y1, avg_x2, avg_y2)
            i += 1
    print("NUM PARKING LANES: ", len(rects))
    buff = 7
    for key in rects:
        tup_topLeft = (int(rects[key][0] - buff), int(rects[key][1]))
        tup_botRight = (int(rects[key][2] + buff), int(rects[key][3]))
        cv2.rectangle(new_image, tup_topLeft,tup_botRight,(0,255,0),3)
    return new_image, rects

def isolate_spots(image, rects, make_copy = True, color=[255, 0, 0], thickness=2, save = True):
    new_image = np.copy(image)
    gap = 15.5
    spot_dict = {}
    tot_spots = 0
    adj_y1 = {0: 20, 1:-10, 2:0, 3:-11, 4:28, 5:5, 6:-15, 7:-15, 8:-10, 9:-30, 10:9, 11:-32}
    adj_y2 = {0: 30, 1: 50, 2:15, 3:10, 4:-15, 5:15, 6:15, 7:-20, 8:15, 9:15, 10:0, 11:30}
    adj_x1 = {0: -8, 1:-15, 2:-15, 3:-15, 4:-15, 5:-15, 6:-15, 7:-15, 8:-10, 9:-10, 10:-10, 11:0}
    adj_x2 = {0: 0, 1: 15, 2:15, 3:15, 4:15, 5:15, 6:15, 7:15, 8:10, 9:10, 10:10, 11:0}
    for key in rects:
        tup = rects[key]
        x1 = int(tup[0] + adj_x1[key])
        x2 = int(tup[2] + adj_x2[key])
        y1 = int(tup[1] + adj_y1[key])
        y2 = int(tup[3] + adj_y2[key])
        cv2.rectangle(new_image, (x1, y1),(x2,y2),(0,255,0),2)
        num_splits = int(abs(y2-y1)//gap)
        for i in range(0, num_splits+1):
            y = int(y1 + i*gap)
            cv2.line(new_image, (x1, y), (x2, y), color, thickness)
        if key > 0 and key < len(rects) -1 :        
            x = int((x1 + x2)/2)
            cv2.line(new_image, (x, y1), (x, y2), color, thickness)
        if key == 0 or key == (len(rects) -1):
            tot_spots += num_splits +1
        else:
            tot_spots += 2*(num_splits +1)
            
        if key == 0 or key == (len(rects) -1):
            for i in range(0, num_splits+1):
                cur_len = len(spot_dict)
                y = int(y1 + i*gap)
                spot_dict[(x1, y, x2, y+gap)] = cur_len +1        
        else:
            for i in range(0, num_splits+1):
                cur_len = len(spot_dict)
                y = int(y1 + i*gap)
                x = int((x1 + x2)/2)
                spot_dict[(x1, y, x, y+gap)] = cur_len +1
                spot_dict[(x, y, x2, y+gap)] = cur_len +2   
    print("TOTAL: ", tot_spots, cur_len)
    return new_image, spot_dict

def assign(image, spot_dict, make_copy = True, color = [0, 0, 255], thickness = 2):
    new_image = np.copy(image)
    for spot in spot_dict.keys():
        (x1, y1, x2, y2) = spot
        cv2.rectangle(new_image, (int(x2), int(y2)), (int(x2), int(y2)), color, thickness)
    return new_image

def test_main():
    test_images = [plt.imread(path) for path in glob.glob('test_images/*.jpg')]
    edge_images = list(map(lambda image: filter_rgb_and_edge_detection(image), test_images))
    roi_images = list(map(select_region, edge_images))
    list_of_lines = list(map(hough_transformation, roi_images))
    line_images = []
    for image, lines in zip(test_images, list_of_lines):
        line_images.append(draw_hough_transformation(image, lines))  
    datapoints = parse_datapoints(lines)
    x_clusters = find_clusters(datapoints)



    #################3
    # rect_images = []
    # rect_coords = []
    # for image, lines in zip(test_images, list_of_lines):
    #     new_image, rects = draw_hough_transformation_withxclustering(image, lines)
    #     rect_images.append(new_image)
    #     rect_coords.append(rects)    
    # display_images(rect_images)
    # delineated = []
    # spot_pos = []
    # for image, rects in zip(test_images, rect_coords):
    #     new_image, spot_dict = isolate_spots(image, rects)
    #     delineated.append(new_image)
    #     spot_pos.append(spot_dict)
    # final_spot_dict = spot_pos[1]
    # marked_spot = list(map(assign(image=test_image[0],spot_dict=final_spot_dict), test_images))
    # display_images(marked_spot)

test_main()
