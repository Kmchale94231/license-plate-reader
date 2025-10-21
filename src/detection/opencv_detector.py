
import os

import numpy as np
import cv2
import math
from matplotlib import pyplot as plt



def detect_candidates(image, cfg):
    img = cv2.imread("C:/Users/Administrator/Desktop/Git/license-plate-reader/data/detector_dataset/images/Cars0.png")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert img_gray is not None, "file could not be read, check with os.path.exists()"

    img_blur = cv2.bilateralFilter(img_gray, 5, 25, 25)

    cv2.imshow('img_blur', img_blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


    edges = cv2.Canny(img_blur, 80, 240)


    cv2.imshow('edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    contours, hierarchy = cv2.findContours(
        edges, 
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
                                        
    candidate_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        candidate_boxes.append((x, y, w, h))

    print("Contours found:", len(contours))
    print("Candidate boxes:", len(candidate_boxes))

    return candidate_boxes

def filter_plate_candidates(contours, img):

    H, W = img.shape[:2]

    MIN_W = 0.05 * W
    MIN_H = 0.03 * H

    AR_MIN, AR_MAX = 2.0, 6.0
    RECT_MIN = 0.50



    candidates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)






# Bounding rectangles, Filtering by aspect ratio + size





"""
Three Functions:

1. detect_candidates(image, cfg) -> list[Box]
- returns a list of condidate boxes (x, y, w, h)

Function one will input image(BGR), convert to grayscale, bilateral filter,
canny edges, find contours, then return all candidates.




2. select_final(boxes, cfg) -> list[Box]

Function will keep the most effective boxes/candidates by area descending, iterate in order
stop once a certain amonut of boxes are there, return the kept boxes.


3. draw_boxes(image, boxes) -> image_with_boxeswhat a

Function will create a list of bounding boxes, loop through boxes, draw a rectangle, use visible color, and then return the modified image.
"""



