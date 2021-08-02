# Importing libraries
from statistics import mode
import imutils
import cv2
import numpy as np
from imutils.video import VideoStream
import time
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import datetime as dt
from matplotlib import cm
import pickle as pkl

# Support functions

def detect_faces(detection_model, gray_image_array, conf):
    frame = gray_image_array
    # Grab frame dimention and convert to blob
    (h,w) =  frame.shape[:2]
    # Preprocess input image: mean subtraction, normalization
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
    (300, 300), (104.0, 177.0, 123.0))
    # Set read image as input to model
    detection_model.setInput(blob)

    # Run forward pass on model. Receive output of shape (1,1,no_of_predictions, 7)
    predictions = detection_model.forward()
    coord_list = []
    count = 0
    for i in range(0, predictions.shape[2]):
        confidence = predictions[0,0,i,2]
        if confidence > conf:
            # Find box coordinates rescaled to original image
            box_coord = predictions[0,0,i,3:7] * np.array([w,h,w,h])
            conf_text = '{:.2f}'.format(confidence)
            # Find output coordinates
            xmin, ymin, xmax, ymax = box_coord.astype('int')
            coord_list.append([xmin, ymin, (xmax-xmin), (ymax-ymin)])
            
        #print('Coordinate list:', coord_list)

    return coord_list


def load_detection_model(prototxt, weights):
    detection_model = cv2.dnn.readNetFromCaffe(prototxt, weights)
    return detection_model

font = cv2.FONT_HERSHEY_SIMPLEX

frame_window = 10
face_offsets = (30, 40)
emotion_offsets = (20, 40)
confidence = 0.6

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('video_output.avi',fourcc, 30, (640,352))


def face_mask_prediction(face_detection,prediction_model,video_path):

    # Loading our model 
    model = load_model(prediction_model)

    # Defining image height and width accepted by our model
    img_width, img_height = 224, 224

    dire = "cropped_faces"

    face_detection_size = (40, 40)
    counter = 0

    # starting video streaming

    video_capture = cv2.VideoCapture(video_path)

    # org 
    org = (10, 40)
    class_lable=' '      
    # fontScale 
    fontScale = 1 
    # Line thickness of 4 px 
    thickness = 4 

    st = dt.datetime.today().timestamp()
    i = 0
    fps = 0

    while True:
        ret, bgr_image = video_capture.read()

        i += 1
        time_diff = dt.datetime.today().timestamp() - st
        
        if time_diff >= 1:
            st = dt.datetime.today().timestamp()
            fps = round(i / time_diff)
            #print("fps",fps)
            i=0

        cv2.putText(bgr_image, 'fps : '+str(fps), (10,30), font,fontScale, (27,109,242), thickness, cv2.LINE_AA)
        if ret == False:
            break
        counter += 1
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        faces = detect_faces(face_detection, bgr_image,confidence)
        
        count = 0
        for face_coordinates in faces:
            
            x, y, w, h = face_coordinates
            org = (x-10,y-10)
            rgb_face = bgr_image[y:y+h,x:x+w]

            try:
                cv2.imwrite(dire +"/"+"face_{}".format(counter) + ".jpg",rgb_face)
                rgb_face = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2BGR)
                rgb_face = cv2.resize(rgb_face,(img_width,img_height))
                img = img_to_array(rgb_face)/255
                img = np.expand_dims(img,axis=0)
                pred_prob = model.predict(img)
                pred=np.argmax(pred_prob)
            except:
                pass
                        
            if pred==0:
                #print("User with mask - predic = ",pred_prob[0][0])
                class_lable = "Mask"
                color = (0, 255, 0)
                cv2.rectangle(bgr_image, (x, y), (x+w, y+h), color, 6)
                cv2.putText(bgr_image, class_lable,org, font,fontScale, color, thickness, cv2.LINE_AA)
            else:
                #print('user not wearing mask - prob = ',pred_prob[0][1])
                class_lable = "No Mask"
                color = (0, 0, 255)
                cv2.rectangle(bgr_image, (x, y), (x+w, y+h), color, 6)
                cv2.putText(bgr_image, class_lable, org,font,fontScale, color, thickness, cv2.LINE_AA)

        # display image
        color_img = cv2.resize(bgr_image, (352, 640))
        out.write(color_img)
        cv2.imshow('LIVE face mask detection', bgr_image)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

    return "successful"

# parameters for loading data and images
prototxt = 'ckpt_/deploy.prototxt.txt'
weights = 'ckpt_/res10_300x300_ssd_iter_140000.caffemodel'

prediction_model = 'face_mask_model.h5'

video_path = '1.mp4'

# loading models
face_detection = load_detection_model(prototxt, weights)

# Making prediction 
a = face_mask_prediction(face_detection,prediction_model,video_path)

print("Done", a)