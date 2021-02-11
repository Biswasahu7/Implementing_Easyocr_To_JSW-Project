import matplotlib.pyplot as plt
import easyocr
import cv2
from pylab import array, uint8
import pytesseract
import numpy as np
import re

# DEFINING EASY_OCR with language
reader = easyocr.Reader(['en'])

# Assigning YOLO MODEL into our net variable
net = cv2.dnn.readNet('/home/vert/OCR/yolov3-config/yolov3_training_last_10.weights',
                      '/home/vert/OCR/yolov3-config/yolov3_testing.cfg')

# Assigning class for the model to detect
classes = []
with open("/home/vert/OCR/yolov3-config/classes.txt", "r") as f:
    classes = f.read().splitlines()


# Creating a function passing outside variable to get the number from railway wagon
def easy_ocr(video_file):

    # Reading local video
    cap = cv2.VideoCapture(video_file)

    # VARIABLES
    wagonid=[]
    temp=[]
    coupling_count=0
    cp=0
    terminate=0
    det=0

    #Assigning color to the image
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

    # Running while loop to read the image from the video
    while True:
        _, img = cap.read()

        if img is None:
            continue

        # Scaling(resize) image to display
        scale_percent = 40
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)

        # Resize the original image
        img= cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        (W, H) = (None, None)
        if W is None or H is None:
            (H, W) = img.shape[:2]

        # Converting image to blob format to give our model to work
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        ln = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(ln)

        #Assigning empty list to append all details
        boxes = []
        confidences = []
        classIDs = []

        # Running for loop into the layeroutput which came from blob format to know the score
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # Getting the confidence score from the detected object
                if confidence >= 0.5:

                    # Creating the bounding box
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x_a = int(centerX - (width / 2))
                    y_a = int(centerY - (height / 2))

                    # Appending all the box, confidence and class
                    boxes.append([x_a, y_a, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # Finally here we can come to know the detection for the object
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        # If the len of indexes is greater then 0 means object detected
        if len(idxs)==0:
            terminate+=1

        # If we are getting non fram then we need ot break teh while loop
        if terminate==150:

            # PRINT RESULT & break WHILE LOOP
            print("Result / Wagon IDs - {}".format(wagonid))
            print("Number of couplings - {}".format(coupling_count))
            break

        # If length of idex is > then o that means model has detect something
        if len(idxs) > 0:
            terminate = 0

            # Flatten the detection object
            for i in idxs.flatten():

                # Creating bounding box
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in colors[classIDs[i]]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

                # If the index of class id is coupling then we will increase cp value
                if "coupling" == classes[classIDs[i]]:
                    cp+=1
                    if cp==2:

                        # Increasing coupling count and print the result
                        coupling_count+=1
                        print(coupling_count)
                        print(temp)

                        # just temporary purpose / to check result of easy_ocr
                        temp.clear()

                        # if no wagon id is detected then just paste XXXXX
                        if det!=1:
                            wagonid.append("XXXXXXXXXXX")
                        # RESET det
                        det=0

                if "wagon" == classes[classIDs[i]] and len(idxs)==1:
                    cp=0


                """This is main part for this video once model has detected and if that is 
                                code then we need ot capture the code form the wagaon"""

                if "code" ==classes[classIDs[i]]:

                    # Croping the image
                    img_crop=img[y:y+h,x:x+w]

                    # Once crop done we need to check index value for reading easyocr
                    if img_crop.shape[1] != 0 and img_crop.shape[0] != 0:
                        result = reader.readtext(img_crop)

                    # define list(num) to store result of easy ocr
                    num=[]
                    for r in result:
                        num.append(r[1])
                    temp.append(num)

        cv2.imshow("Image",img)
        cv2.waitKey(1)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# Assigning testing video
video="/home/vert/OCR/Video/video_cctv_WagonSideView1_2020-12-08 07_08_22.707970.avi"

# Calling easy_ocr function to get the number from railway wagon

easy_ocr(vido)