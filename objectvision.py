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

#[BROAD STROKES DRAWN FROM OPEN-SOURCES & PAPERS LISTED IN README.md. IMPROVEMENTS & CHANGES ALSO LISTED IN README.md]

#[PACKAGE IMPORTS]
from __future__ import division
import cv2
import math
import pickle
import decimal
import operator
import os, glob
import pylab as p
import statistics
import numpy as np
import matplotlib.pyplot as plt
from numpy import array, linspace
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.signal import argrelextrema, argrelmax, find_peaks

#[PRE-SCRIPTING DIRECTIVES]
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

#[BINARY CONVERSION AND CANNY EDGE DETECTION]
def filter_rgb_and_edge_detection(image, lt = 50, ht = 200): 
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 137, 255, cv2.THRESH_BINARY)
    return cv2.Canny(blackAndWhiteImage, lt, ht)

#[REGION OF INTEREST MASKING]
def filter(image, vertices):
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,) * mask.shape[2])      
    return cv2.bitwise_and(image, mask)

#[HARD-CODING THE REGION OF INTEREST]
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

#[PROBABILISTIC HOUGH TRANSFORMATION]
def hough_transformation(image):
    return cv2.HoughLinesP(image, rho = 0.1, theta = np.pi/10, threshold = 15, minLineLength = 7, maxLineGap = 5)
 
 #[DRAWING HOUGH TRANSFORMATION ON IMAGE]
def draw_hough_transformation(image, lines, color = [255, 0, 0], thickness = 2):
    image = np.copy(image)
    cleaned = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if abs(y2-y1) <=1 and abs(x2-x1) >=25 and abs(x2-x1) <= 55:
                cleaned.append((x1,y1,x2,y2))
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

#[COMPILING A USABLE LIST OF LINE COORDINATES]
def parse_datapoints(lines):
    datapoints = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            average = decimal.Decimal((x1 + x2)/2)
            datapoints.append(average)
    return datapoints

#[COMPILING SECONDARY LIST WITH Y-COORDINATES AS WELL AS X-COORDINATES]
def parse_xy(lines):
    datapoints = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            average = decimal.Decimal((x1 + x2)/2)
            datapoints.append((average, y2))
    return datapoints

#[FIND LOCAL MAXIMA IN HISTOGRAM WHICH WOULD THEORETICALLY SIGNAL A PARKING LANE]
def find_clusters(datapoints):
    #[HISTOGRAM CREATION]
    npdata = np.array(datapoints, dtype=float)
    w = 10 #[CHANGE THIS VALUE TO INCREASE/DECREASE THE SIZE OF BINS]
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
            
#[CREATE BOUNDING BOXES THAT WILL EVENTUALLY BE SPLIT INTO SMALLER BOXES (TO BE FED INTO R-CNN MODEL)]
def create_bounding_boxes (xclusters, datapoints):
    #[TAKING THE AVERAGE DISTANCE BETWEEN PEAKS AND THEN DIVIDING IT BY FOUR]
    difflist = []
    for i in range(len(xclusters)-1):
        difflist.append(xclusters[i+1]- xclusters[i])
    buffer = statistics.mean(difflist)/4

    #[LOADING INTO MIN AND MAX Y VALUES FOR EACH X-CLUSTER]
    dictwithmaxandminy = {}
    for val in xclusters:
        dictwithmaxandminy[val] = []
    for val in xclusters:
        min_val = int(str(min(xy[1] for xy in sorted(datapoints) if val - 10 < xy[0] < val + 10)))
        max_val = int(str(max(xy[1] for xy in sorted(datapoints) if val - 10 < xy[0] < val + 10)))
        dictwithmaxandminy[val].append(min_val)
        dictwithmaxandminy[val].append(max_val)
    
    #[RETURN DICTIONARY WTIH BUFFER]
    return dictwithmaxandminy, buffer
 
 #[DRAW THE BOUNDING BOXES ON THE IMAGE]
def draw_dict(image, dictionary, buff, color = [0, 0, 255], thickness = 2, make_copy = True):
    new_image = np.copy(image)
    for key, value in dictionary.items():
        tup_topLeft = (int(key)-int(buff), value[1])
        tup_botRight = (int(key)+int(buff), value[0])
        cv2.rectangle(new_image, tup_topLeft, tup_botRight,(0,255,0), 3)
    return new_image


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

    #[LIST OF LINES NOW CREATED]
    list_of_lines = list(map(hough_transformation, roi_images))
    line_images = []
    for image, lines in zip(test_images, list_of_lines):
        line_images.append(draw_hough_transformation(image, lines))  

    #[WITH DATAPOINTS]
    datapoints = parse_datapoints(lines)
    xclusters = find_clusters(datapoints)
    dictionarywithycoords = parse_xy(lines)
    dictwithminmax, gap = create_bounding_boxes(xclusters, dictionarywithycoords)

    #[RECT IMAGES]
    rect_images = []
    for image in test_images:
        rect_images.append(draw_dict(image, dictwithminmax, gap))

    #[FINAL DISPLAY]
    display_images(rect_images)
    
test_main()
