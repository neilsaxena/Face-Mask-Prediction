

import os
import cv2

MY_DIRECTORY = "D:/Neil/college 2/Semester 4/Research/Heena/implementation/Model creation/dataset/valid/without_mask/"

for root, subdirs, files in os.walk(MY_DIRECTORY):
    for f in files:
        image = cv2.imread(MY_DIRECTORY+f, 1)
        image = cv2.resize(image, (224, 224))
        cv2.imwrite(MY_DIRECTORY+f, image)
