# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

class FaceDetector(object):
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)
        
    def detect(self, image, biggest_only=True):
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (30,30)
        biggest_only = True
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else cv2.CASCADE_SCALE_IMAGE
        
        return self.classifier.detectMultiScale(image,
                               scaleFactor=scale_factor,
                               minNeighbors=min_neighbors,
                               minSize=min_size,
                               flags=flags)
        
class VideoCamera(object):
    def __init__(self, index=0):
        self.video = cv2.VideoCapture(index)
        self.index = index
        if self.video.isOpened():
            print("Video started.")
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self, in_grayscale = False):
        _, frame = self.video.read()
        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
    

def cut_faces(image, faces_coord):
    faces = []
    
    for(x, y, w, h) in faces_coord:
        w_rm = int(0.2 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])
    
    return faces

def normalize_intensity(images):
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_norm.append(cv2.equalizeHist(image))
    return images_norm

def resize(images, size=(50,50)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, interpolation = cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, interpolation = cv2.INTER_CUBIC)
    images_norm.append(image_norm)
    
    return images_norm

def normalize_images(frame, frame_coords):
    faces = cut_faces(frame, frame_coords)
    norm_images = normalize_intensity(faces)

    return resize(norm_images)

def draw_rectangle(image, faces_coords):
        for (x, y, w, h) in faces_coords:
            cv2.rectangle(image,(x,y),(x + w,y + h),(150,150,0),8)
        



        
        
    

