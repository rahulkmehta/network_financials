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

#[BROAD STROKES DRAWN FROM OPEN-SOURCES & PAPERS LISTED IN README.md. IMPROVEMENTS, CHANGES, AND METHODOLOGIES ALSO LISTED IN README.md]

#[PACKAGE IMPORTS]
from __future__ import division
import cv2
import math
import pickle
import decimal
import operator
import os, glob
import statistics
import numpy as np
import matplotlib.pyplot as plt
from numpy import array, linspace
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.signal import argrelextrema, argrelmax, find_peaks
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

#[PRE-SCRIPTING DIRECTIVES]
cwd = os.getcwd()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
top_model_weights_path = 'car1.h5'
class_dictionary = {}
class_dictionary[0] = 'empty'
class_dictionary[1] = 'occupied'
modelo = load_model(top_model_weights_path)

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
    #[FOR TEST IMAGE 3, TEST IMAGE 4]
    # pt_1  = [cols*0.05, rows*0.87]
    # pt_2 = [cols*0.05, rows*0.05]
    # pt_3 = [cols*0.95, rows*0.05]
    # pt_4 = [cols*0.95, rows*0.95]
    # pt_5 = [cols*0.80, rows*0.95] 
    # pt_6 = [cols*0.70, rows*0.87]

    #[FOR TEST IMAGE 1, TEST IMAGE 2]
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
    #[OLD VERSIONS]
    #return cv2.HoughLinesP(image, rho = 0.1, theta = np.pi/10, threshold = 15, minLineLength = 7, maxLineGap = 5)

    #[COMPROMISE BETWEEN DIFFERENT TYPES OF LOTS]
    return cv2.HoughLinesP(image, rho = 0.75, theta = np.pi/180, threshold = 30, minLineLength = 10, maxLineGap = 5)


#[DRAWING HOUGH TRANSFORMATION ON IMAGE FOR DEVELOPMENT PURPORSES]
def draw_hough_transformation(image, lines, color = [255, 0, 0], thickness = 2):
    image = np.copy(image)
    for line in lines:
        for x1,y1,x2,y2 in line:
            #[OLD CONTINGENCY]
            #if abs(y2-y1) <=1 and abs(x2-x1) >=25 and abs(x2-x1) <= 55:

            #[FILTERS VERTICAL LINES THAT ARE PROBABLY DRAWN ERRONEOUSLY]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
            if abs(angle) < 80:
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

    #[HISTOGRAM SUPPORT]
    plt.hist(npdata, bins = n)
    plt.show()

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
    buffer = statistics.mean(difflist)/3.8

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
    rect_coords = []
    new_image = np.copy(image)
    for key, value in dictionary.items():
        tup_topLeft = (int(key)-int(buff), value[1])
        tup_botRight = (int(key)+int(buff), value[0])
        cv2.rectangle(new_image, tup_topLeft, tup_botRight,(0,255,0), 3)
        rect_coords.append((tup_topLeft[0], tup_topLeft[1], tup_botRight[0], tup_botRight[1]))
    return new_image, rect_coords

#[FOR DEVELOPMENT PURPOSES, DRAWING THE SUB-BOUNDING BOXES]
def draw_parking(rect_coords):
    gap = 15.5
    spot_dict = {}
    tot_spots = 0

    for coord in rect_coords:
        begy = coord[1]
        endy = coord[3]
        x1a = int(coord[0])
        x1b = int((coord[2]-coord[0]) / 2) + x1a
        x1c = int(coord[2])

        while (begy > endy):
            spot_dict[(x1a, begy, x1b, begy-gap)] = tot_spots
            tot_spots += 1
            spot_dict[(x1b, begy, x1c, begy-gap)] = tot_spots
            tot_spots += 1
            begy -= gap
    return spot_dict

#[ASSIGN TO SPOT_DICT]
def assign(image, spot_dict, make_copy = True, color = [0, 0, 255], thickness = 2):
    new_image = np.copy(image)
    for spot in spot_dict.keys():
        (x1, y1, x2, y2) = spot
        cv2.rectangle(new_image, (int(spot[0]), int(spot[1])), (int(spot[2]), int(spot[3])), color, thickness)
    return new_image
   
#[MAKE THE PREDICTION OF RESIZED IMAGE]     
def make_prediction(image, finmodel):
    img = image/255.
    image = np.expand_dims(img, axis=0)
    class_predicted = finmodel.predict(image)
    inID = np.argmax(class_predicted[0])
    label = class_dictionary[inID]
    return label

#[PREDICTS ON IMAGE AND GETS FINAL PERCENTAGE]
def predict_on_image(image, spot_dict, model, make_copy=True, color = [0, 255, 0], alpha=0.5):
    new_image = np.copy(image)
    overlay = np.copy(image)
    cnt_empty = 0
    all_spots = 0
    for spot in spot_dict.keys():
        all_spots += 1
        (x1, y1, x2, y2) = spot
        (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
        spot_img = image[y2:y1, x1:x2]
        spot_img = cv2.resize(spot_img, (48, 48)) 
        label = make_prediction(spot_img, model)
        if label == 'empty':
            cnt_empty += 1
    occupied = all_spots - cnt_empty
    percentage = occupied/all_spots
    print (percentage, "%")

#[DRVER]
def test_main():
    ti = [plt.imread(path) for path in glob.glob('test_images/*.jpg')]
    test_images = []
    for newimg in ti:
        if len(newimg.shape) > 2 and newimg.shape[2] == 4:
            newimg = cv2.cvtColor(newimg, cv2.COLOR_BGRA2BGR)
        test_images.append(newimg)
    
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
    rect_coords = []
    for image in test_images:
         ri, rc = draw_dict(image, dictwithminmax, gap)
         rect_images.append(ri)
         rect_coords.append(rc)

    #[GIVEN THE RECT COORDINATES OF THE PARKING LANES]
    spot_pos = []
    for rc in rect_coords:
         spot_dict = draw_parking(rc)
         spot_pos.append(spot_dict)
    final = []
    final_spot_dict = spot_pos[1]
    for image in test_images:
         final.append(assign(image, final_spot_dict))
    display_images(final)

    #[MAKE FINAL PREDICTION]
    #for image in test_images:
        #predict_on_image(image, spot_dict = final_spot_dict, model = modelo)

test_main()
