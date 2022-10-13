import cv2 # draw rectangle on face and label image
import os
import face_recognition
import speech_recognition as sr
import matplotlib.pyplot as plt
import numpy as np
import pyttsx3
import pywhatkit
import time
import requests
import pyttsx3
import json

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# ===============================================================================
# distnace measurement
KNOWN_DISTANCE = 45  # INCHES
PERSON_WIDTH = 16  # INCHES
MOBILE_WIDTH = 3.0  # INCHES

# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.31

# colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
# defining fonts
FONTS = cv2.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)


# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id
        color = COLORS[int(classid) % len(COLORS)]

        label = "%s : %f" % (class_names[classid[0]], score)

        # draw rectangle on and label on object
        # cv2.rectangle(image, box, color, 2)
        # cv2.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)

        # getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid == 0:  # person class id
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1] - 2)])
        elif classid == 67:
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1] - 2)])
            # if you want inclulde more classes then you have to simply add more [elif] statements here
            # returning list containing the object data.
    return data_list


def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length


# distance finder function
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance


# reading the reference image from dir
ref_person = cv2.imread('ReferenceImages/image14.png')
ref_mobile = cv2.imread('ReferenceImages/image4.png')

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")

# finding focal length
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1550)
# ==============================================================================
classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()
# ............................................
listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[0].id)

# ------------------------------------------------------------------------------------------------------------------------------------
# news and speech
def talk(text):
    engine.say(text)
    engine.runAndWait()
# engine = pyttsx3.init('sapi5')
# voices = engine.getProperty('voices')
# # print(voices[1].id)
# engine.setProperty('voice', voices[0].id)
# def speak(audio):
#     engine.setProperty("rate", 180)   #--------------> for engine speed control
#     engine.say(audio)
#     engine.runAndWait()
#
# r = requests.get("https://newsapi.org/v2/top-headlines?country=in&category=sports&apiKey=cff18465c61e48dd8e04fb24bb8af554").text
# pa = json.loads(r)
# r1 = requests.get("https://newsapi.org/v2/top-headlines?country=in&category=business&apiKey=cff18465c61e48dd8e04fb24bb8af554").text
# pa1 = json.loads(r1)
# r2 = requests.get("https://newsapi.org/v2/top-headlines?country=in&category=entertainment&apiKey=cff18465c61e48dd8e04fb24bb8af554").text
# pa2 = json.loads(r2)
# r3 = requests.get("https://newsapi.org/v2/top-headlines?country=in&category=science&apiKey=cff18465c61e48dd8e04fb24bb8af554").text
# pa3 = json.loads(r3)
# c = 0
# speak("news for today....lets begin")
# speak("Starting from the bussiness news ")
# for i in pa1['articles']:
#     a = i['description']
#     l = a.split(',')
#     print(l[0])
#     speak(l[0])
#     c = c+1
#     if c==2:
#         break
# c = 0
# speak("Entertainment News")
# for i in pa2['articles']:
#     a = i['description']
#     l = a.split(',')
#     print(l[0])
#     speak(l[0])
#     c = c+1
#     if c==2:
#         break
# c = 0
# speak("Science news")
# for i in pa3['articles']:
#     a = i['description']
#     l = a.split(',')
#     print(l[0])
#     speak(l[0])
#     c = c+1
#     if c==2:
#         break
# speak("Sports news")
# c = 0
# for i in pa['articles']:
#     a = i['description']
#     l = a.split(',')
#     print(l[0])
#     speak(l[0])
#     c = c+1
#     if c==2:
#         break
# speak("thankyou for listing")
#
# # engine.say("this model is created for the purpose to make easy for the blind person to guide them and make their task easier")
# engine.runAndWait()

# ............................................
KNOWN_FACES_DIR = "C:\\Users\\hp\\face_photos"
# UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 0.6 # the lower the tolerance the less chance you have to get false positive
# the more is the tolerance the more there will be matching and it can be more risk of incorrect matching.
# if tolerance is less then matching will be mostly correct and there can be less mtching and more flase negative

FRAME_THICKNESS = 3 # size of frame around face. It depends on how big your image is.
FONT_THICKNESS = 2
MODEL="hog" # or we can use "cnn" model of we use gpu because "ccn" is slow in cpu

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,2000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1550)
font = cv2.FONT_HERSHEY_PLAIN
colors1 = np.random.uniform(0, 255, size=(100, 3))


known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):

    f = None
    # Next we load every file of faces of known person
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        if f == None:
            print(f"No of {name} photos = {len(os.listdir(f'{KNOWN_FACES_DIR}/{name}'))}")
            f = 1

        # Load an image
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')

        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
        encoding = face_recognition.face_encodings(image)[0]

        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(name)

print(known_names)

print("processing unknown faces")
c = 0
l = ["0"]*20
talk("starting AI model........")
xx = int(time.strftime("%S", time.localtime()))
print(xx)
if xx>55:
    xx = xx-60
while True:
    ret, image = cap.read()
    height, width, ret = image.shape
    # ----------------------------------------------------------------
    # distnace
    data = object_detector(image)
    for d in data:
        if d[0] == 'person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'cell phone':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
        else:
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
        # cv2.rectangle(image, (x, y - 3), (x + 150, y + 23), BLACK, -1)
        # cv2.putText(image, f'Dis: {round(distance, 2)} inch', (x + 5, y + 13), FONTS, 0.48, GREEN, 2)

    cv2.imshow('frame', image)
    # -------------------------------------------------------------------
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes) > 0:
        sum = 0
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors1[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label + " " + confidence, (x, y - 20), font, 2, (255, 255, 255), 2)
            cv2.rectangle(image, (x, y - 3), (x + 150, y + 23), BLACK, -1)
            cv2.putText(image, f'Dis: {round(distance, 2)} inch', (x + 5, y + 13), FONTS, 0.48, GREEN, 2)

            ww = str(label)+" detection in front of you"

            if ww not in l:
                talk(str(label)+" detection in front of you")
                print("appended :"+ww)
                l[c] = ww
                c = c+1

            # # time
            yy = int(time.strftime("%S", time.localtime()))
            if (xx + 3 == yy):
                print("destination reach :", yy)
                xx = yy + 3
                if (xx > 60):
                    xx = xx - 60
                print("x value:", xx)
                print(l)
                l.pop(0)
                print(l)


            x1 = float(confidence)
            print(x1)
            if str(label)=='person':
                sum = sum + x1
                if str(label)=='person' and sum > 1.0:
                    ww = str(round(sum))+" persons near you"
                    if ww not in l:
                        talk(str(round(sum)) + " persons near you")
                        print("appended :" + ww)
                        l[c] = ww
                        c = c+1

    # cv2.imshow('Image', image)
    key = cv2.waitKey(1)
    if key == 27:
        break


    locations = face_recognition.face_locations(image, model=MODEL)  # A list of tuples of found face locations in css (top, right, bottom, left) order

    encodings = face_recognition.face_encodings(image, locations)

    if locations == []:
        cv2.putText(image, "No Face is detected in this image", (0 + 10, 0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 0), FONT_THICKNESS)
        ww = "No face detected"
        if ww not in l:
            talk("No face detected")
            print("appended :" + ww)
            l[c] = ww
            c = c+1

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        # time
        yy = int(time.strftime("%S", time.localtime()))
        if (xx + 3 == yy):
            print("destination reach :", yy)
            xx = yy + 3
            if (xx > 60):
                xx = xx - 60
            print("x value:", xx)
            print(l)
            l.pop(0)
            print(l)
        if True in results:
            match = known_names[results.index(True)]
            print(f' - {match} from {results}')
            x = str(match)
            if x=="harshith":
                ww = "harshith in front of you"
                if ww not in l:
                    talk("harshith in front of you")
                    print("appended :" + ww)
                    l[c] = ww
                    c = c+ 1
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])

                color = [0, 255, 0]  # BGR

                cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)

                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

                cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (43, 37, 9), FONT_THICKNESS)
                # talk("candidate verfied as "+x)
                # talk("you can now start your exam")
            # Each location contains positions in order: top, right, bottom, left
        else:
            print(f"Face is not recognised")
            ww = "Face is not recognised"
            if ww not in l:
                talk("Face is not recognised")
                print("appended :" + ww)
                l[c] = ww
                c = c+1
            # top_left = (face_location[3], face_location[0])
            # bottom_right = (face_location[1], face_location[2])
            #
            # color = [0, 255, 0]  # BGR
            #
            # cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
            #
            # top_left = (face_location[3], face_location[2])
            # bottom_right = (face_location[1], face_location[2] + 22)
            #
            # cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            #
            # cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            #             (43, 37, 9), FONT_THICKNESS)

            # Each location contains positions in order: top, right, bottom, left [x-axis = left,right and y-axis=bottom,top]
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            color = [0, 0, 255]  # BGR

            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            cv2.putText(image, "UNKNOWN", (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (43, 37, 9), FONT_THICKNESS)

    cv2.imshow('frame', image)
    if cv2.waitKey(1) == ord('q'):
        break
#     cv2.waitKey(10000)
#     cv2.destroyWindow(filename)

cap.release()
cv2.destroyAllWindows()