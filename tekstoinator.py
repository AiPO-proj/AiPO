#!/usr/bin/env python
# coding: utf-8

# # Tekstoinator
# 
# Robimy rzeczy nie dlatego, że są proste tylko dlatego, że są na zaliczenie!

# In[1]:


import cv2
import numpy as np

DETECTION_SIZE = (640, 640)

def detect_and_annotate(image):
    inputSize = (640, 640)
    imc = image.copy()
    image_scaled = cv2.resize(imc, (640, 640))
    mean = (122.67891434, 116.66876762, 104.00698793)


    image_processed = image_scaled.copy()

    if image.shape[0] < 640 or image.shape[1] < 640:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        

        Z = image_processed.reshape((-1,3))
        Z = np.float32(Z)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        clusters_n = 4

        ret,label,center=cv2.kmeans(Z,clusters_n,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        
        image_processed = res.reshape((640, 640, 3))

        image_processed = cv2.GaussianBlur(image_processed, (5,5), 2)

    textDetectorDB50= cv2.dnn_TextDetectionModel_DB("./DB_TD500_resnet50.onnx")
    textDetectorDB50.setBinaryThreshold(0.2).setPolygonThreshold(0.4)
    textDetectorDB50.setInputParams(1.0/255, inputSize, mean, False)
    boxes, confidences = textDetectorDB50.detect(image_processed)


    plausible_text = []

    # Process all detected text
    for idx, box in enumerate(boxes):

        x, y, w, h = cv2.boundingRect(box)
        
        if x < 0:
            x = 0
        if y < 0:
            y = 0

        plausible_text.append(image_scaled[y:y+h, x:x+w])
    

    return plausible_text

# For finding road signs, hopefully, probably - not very reliably but good-enoughly
# Why does grass have to be green? Well it is due to the composition of the Sun's light spectrum
# The human eye is also adapted to receiving green for the same reason
# This in turn is likely the reason designers chose green as a background for road signs
# Truly the detection of road signs is made difficult by the laws of physics

def find_green(image):
    returned_images = []

    green_lower = np.array([0, 60, 25])
    green_upper = np.array([50, 255, 120])


    small_kernel = np.ones((5,5),np.uint8)
    big_kernel = np.ones((26,26),np.uint8)

    

    example_frame_masked = cv2.inRange(image, green_lower, green_upper)
    example_frame_masked = cv2.morphologyEx(example_frame_masked, cv2.MORPH_OPEN, small_kernel)
    example_frame_masked = cv2.dilate(example_frame_masked, big_kernel, iterations = 3)
    example_frame_masked = cv2.morphologyEx(example_frame_masked, cv2.MORPH_CLOSE, small_kernel)

    num_labels, labels_im = cv2.connectedComponents(example_frame_masked)
    
    for i in range(1,num_labels):
        mask = (labels_im == i).astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        x, y, w, h = cv2.boundingRect(contours[0])
        x_max, y_max = x + w, y + h

        # Square good

        if w > h:
            if (image.shape[1] >= y + w):
                h = w
        elif h > w:
            if (image.shape[0] >= x + h):
                w = h

        returned_images.append(image[y:y_max,x:x_max,  :])
    
    return returned_images


# In[3]:


import easyocr

reader = easyocr.Reader(['en', 'pl', 'es'])


import cv2
import matplotlib.pyplot as plt


def extract_text_from_video(video_path, frame_interval=25):
    cap = cv2.VideoCapture(video_path)
    k = 0
    found_text = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        k += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if k % frame_interval == 0:
            found_text += reader.readtext(frame)

            example_frames_annotated = detect_and_annotate(frame)
            found_green = find_green(frame)

            for green in found_green:
                example_frames_annotated += detect_and_annotate(green)

            for region in example_frames_annotated:
                found_text += reader.readtext(region)

    cap.release()

    filtered_text = []
    for text in found_text:
        if text[2] > 0.6 and len(text[1]) > 3:
            filtered_text.append(text[1].upper())

    return list(set(filtered_text))

