#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:53:00 2019

@author: mponsa
"""
import cv2
import os
from face_recognition import VideoCamera, FaceDetector, normalize_images, draw_rectangle

folder = "people/" + input("Person: ").lower()
webcam = VideoCamera()
detector = FaceDetector("xml/haarcascade_frontalface_alt.xml")

cv2.namedWindow("Dataset Builder",cv2.WINDOW_AUTOSIZE)

if not os.path.exists(folder):
    os.mkdir(folder)
    counter = 0
    timer = 0
    while counter < 10 :
        frame = webcam.get_frame()
        faces_coord = detector.detect(frame)
        if len(faces_coord):
            faces = normalize_images(frame, faces_coord)
            cv2.imwrite(folder + '/' + str(counter) + '.jpg', faces[0])
            counter += 1
        draw_rectangle(frame, faces_coord)
        cv2.imshow("Dataset Builder",frame)
        cv2.waitKey(50)
        timer += 50
    cv2.destroyAllWindows()
else:
    print ("This name already exists")

