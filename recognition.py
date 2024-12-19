import cv2
from ultralytics import YOLO
import os
import numpy as np
from paddleocr import PaddleOCR
from Levenshtein import distance
from PIL import Image
import math
from numpy import asarray

img_path = 'PATH_TO_IMAGES_FOLDER'

# Load ocr model
ocr = PaddleOCR(use_angle_cls=False, lang="en")

#Load YOLO models
ship_det_model = YOLO("PATH_TO_SHIP_DETECTION_MODEL")
text_det_model = YOLO("PATH_TO_TEXT_DETECTION_MODEL")

# Load test images
img_list = [] #image list
files = [] #filename list
for f in sorted(os.listdir(img_path)):
    img_list.append(cv2.imread(img_path+f, cv2.COLOR_BGR2RGB))
    files.append(f)


words_dict_det = {} #prediction dictionary
words_dict_acc = {} #accuracy dictionary
num3 = 0
file_num = 0

for img in img_list:
    filename = files[file_num]
    ship_det = ship_det_model.predict(img, save=False, conf=0.5, iou = 0.6, classes = [3,4])
    cropped1 = []
    words_dict_det[filename] = []
    words_dict_acc[filename] = []
    test_images = []
    for s in ship_det:
        cropping_list1 = s.boxes.data.tolist() #ship detection list
    if len(cropping_list1)>0: #if ships are detected
        boat_start_x = []
        boat_start_y = []
        start_point = []
        end_point = []
        for crop in cropping_list1: #for every ship detected
            start_point.append((int(crop[0]), int(crop[1])))
            end_point.append((int(crop[2]), int(crop[3])))
            boat_start_x.append(int(crop[0]))
            boat_start_y.append(int(crop[1]))
            width = int(crop[3])- int(crop[1])
            height = int(crop[2]) - int(crop[0])
            cropped_test = img[int(crop[1]):int(crop[1])+width, int(crop[0]):int(crop[0]) + height]
            cropped1.append(cropped_test) #cropped ships list
       
        for i in range(len(start_point)): #labeling ships in the original image
            test_img = cv2.rectangle(img, start_point[i], end_point[i], color=(0,165,255), thickness = 2)

        for i in range(len(cropped1)): #for every cropped ship
            text_det = text_det_model(cropped1[i], stream = False, iou = 0.5)
            cropped2 = []
            for t in text_det:
                cropping_list2 = t.boxes.data.tolist() #text detection list
            if len(cropping_list2)>0: #if texts are detected
                box_list = []
                start_point = []
                end_point = []
                for crop in cropping_list2:
                    start_point.append((int(crop[0])+boat_start_x[i], int(crop[1])+boat_start_y[i]))
                    end_point.append((int(crop[2])+boat_start_x[i], int(crop[3])+boat_start_y[i]))
                    width = int(crop[3])- int(crop[1])
                    height = int(crop[2]) - int(crop[0])
                    box_list.append((int(crop[0]+boat_start_x[i]), int(crop[1]+boat_start_y[i]-5)))
                    cropped_test = img[int(crop[1]) +boat_start_y[i]:int(crop[1])+boat_start_y[i]+width, int(crop[0])+boat_start_x[i]:int(crop[0])+boat_start_x[i] + height]
                    h,w,g = cropped_test.shape
                    size = h*w
                    if size<24000: #cropped text is too small
                        ratio = 24000/size
                        factor = math.sqrt(ratio)
                        im = Image.fromarray(cv2.cvtColor(cropped_test, cv2.COLOR_RGB2BGR))
                        im = im.resize((int(factor*w), int(factor*h)))
                        cropped_test2 = np.asarray(im)
                        cropped2.append(cropped_test2) #cropped texts list
                    else:
                        cropped_test2 = cropped_test
                        cropped2.append(cropped_test2)

                for s in range(len(start_point)): #labeling texts in the original image
                    test_img2 = cv2.rectangle(test_img, start_point[s], end_point[s], color=(0,0,255), thickness = 2)

                box_num = 0
                for c in cropped2: #for every cropped text
                    img_text_data = ocr.ocr(c, cls=False, det = False)
                    if img_text_data is not None: #if the text is recognized
                        if img_text_data[0] is not None:
                            rec_text = img_text_data[0][0][0] #recognized text
                            words_dict_det[filename].append(rec_text)
                            acc = img_text_data[0][0][1] #accuracy
                            words_dict_acc[filename].append(acc)
                            cv2.putText(test_img2, rec_text, box_list[box_num], cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                            box_num = box_num + 1
                im = Image.fromarray(cv2.cvtColor(test_img2, cv2.COLOR_RGB2BGR))
                cv2.imshow("Result", test_img2)
                cv2.waitKey(0)
                num3 = num3 + 1

            else: #no text detected
                im = Image.fromarray(cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
                cv2.imshow("Result", test_img)
                cv2.waitKey(0)
                num3 = num3 + 1

    else: #no ships detected
        im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        num3 = num3 + 1
    file_num = file_num + 1