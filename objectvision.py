#[AUTHOR: RAHUL MEHTA]
#[ALPHA VERSION: JUNE 1, 2020]
#[SOME METHODOLOGY DRAWN FROM OPEN-SOURCES LISTED IN ACKNOWLEDGEMENTS]

#[PACKAGE IMPORTS]
from __future__ import division
import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np
import pickle
import operator

#[PRE-PROCESSING DIRECTIVES]
cwd = os.getcwd()

#[HELPER FUNCTION TO DISPLAY IMAGES BEFORE/AFTER PROCESSING]
def display_images(images, cmap=None):
    cols = 2
    rows = (len(images)+1)//cols
    plt.figure(figsize=(15, 12))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i+1)
        cmap = 'gray' if len(image.shape)==2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()

def filter_rgb_and_edge_detection(image, lt=50, ht=200): 
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    return cv2.Canny(blackAndWhiteImage, lt, ht)

def define_verts(image):
    rows, cols = image.shape[:2]
    #[TAKEN FROM OPEN-SOURCE AT FACE-VALUE BUT SHOULD BE VERIFIED FOR CORRECTNESS ON OTHER IMAGES]
    #[-----------------------------]
    pt_1  = [cols*0.05, rows*0.90]
    pt_2 = [cols*0.05, rows*0.70]
    pt_3 = [cols*0.30, rows*0.55]
    pt_4 = [cols*0.6, rows*0.15]
    pt_5 = [cols*0.90, rows*0.15] 
    pt_6 = [cols*0.90, rows*0.90]
    #[------------------------------]
    vertices = np.array([[pt_1, pt_2, pt_3, pt_4, pt_5, pt_6]], dtype=np.int32)
    color_mask = np.zeros_like(image)
    if len(color_mask.shape)==2:
        cv2.fillPoly(color_mask, vertices, 255)
    else:
        cv2.fillPoly(color_mask, vertices, (255,) * color_mask.shape[2]) # in case, the input image has a channel dimension        
    return cv2.bitwise_and(image, color_mask)

#[NOTE: THE EDGE DETECTION NEEDED TO BE DONE BEFORE THE HOUGH TRANSFORMATION OTHERWISE THE EFFICIENCY SUFFERS GREATLY]
#[NOTE: NOISY IMAGES DO NOT WORK WITH HOUGH TRANSFORMATIONS]

def hough_transformation(image):
    return cv2.HoughLinesP(image, rho=0.1, theta=np.pi/10, threshold=15, minLineLength=9, maxLineGap=4)

def draw_hough_transformation_withxclustering(image, lines, color=[0, 0, 255], thickness=2, make_copy=True):
    image = np.copy(image) # don't want to modify the original
    cleaned = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if abs(y2-y1) <=1 and abs(x2-x1) >=25 and abs(x2-x1) <= 55:
                cleaned.append((x1,y1,x2,y2))


test_images = [plt.imread(path) for path in glob.glob('test_images/*.jpg')]
edge_images = list(map(lambda image: filter_rgb_and_edge_detection(image), test_images))
masked_images = list(map(define_verts, edge_images))
list_of_lines = list(map(hough_transformation, masked_images))

line_images = []
for image, lines in zip(test_images, list_of_lines):
    line_images.append(draw_hough_transformation(image, lines))

display_images(line_images)
