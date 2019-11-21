# -*- coding: utf-8 -*-

import cv2
import cv2.face
import os
import numpy as np
from face_recognition import VideoCamera, FaceDetector, normalize_images, draw_rectangle, draw_label

def collect_dataset():
    images = []
    labels = []
    labels_dic = {}
    people = [person for person in os.listdir("people/")]
    people = people[1:]
    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir("people/"+ person):
            images.append(cv2.imread("people/" + person + '/' + image, 0))
            labels.append(i)
    
    return(images, np.array(labels), labels_dic)
    
images, labels, labels_dic = collect_dataset()

# Train models.

rec_eig = cv2.face.EigenFaceRecognizer_create()
rec_eig.train(images, labels) #Eigen

rec_fisher = cv2.face.FisherFaceRecognizer_create()
rec_fisher.train(images, labels) #Fisher

rec_lbph = cv2.face.LBPHFaceRecognizer_create()
rec_lbph.train(images, labels) #LBPH

cv2.namedWindow("Face recognition", cv2.WINDOW_AUTOSIZE)
webcam = VideoCamera()
detector = FaceDetector("xml/haarcascade_frontalface_alt.xml")

while True:
    frame = webcam.get_frame()
    faces_coord = detector.detect(frame)
    if len(faces_coord):
        faces = normalize_images(frame, faces_coord)
        for i, face in enumerate(faces):
            collector = cv2.face.StandardCollector_create()
            rec_lbph.predict_collect(face,collector)
            conf = collector.getMinDist()
            pred = collector.getMinLabel()
            treshold = 140
            print ("Prediction: " + str(labels_dic[pred]) + "\nConfidence: " + str(conf))
            os.system('clear')
            draw_label(frame,labels_dic[pred],
                       (faces_coord[i][0], faces_coord[i][1] - 10),
                        conf, treshold)
        draw_rectangle(frame,faces_coord)         
    cv2.imshow("Face recognition", frame)
    if cv2.waitKey(40) & 0xFF == 27:
        webcam.__del__()
        cv2.destroyAllWindows()
        break
