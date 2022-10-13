import cv2 # draw rectangle on face and label image
import os
import face_recognition
import speech_recognition as sr
import matplotlib.pyplot as plt
import pyttsx3
import pywhatkit
# ............................................
listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[0].id)

def talk(text):
    engine.say(text)
    engine.runAndWait()

engine.say("THIS IS AN ONLINE PROCTORING SYSTEAM")
engine.say("Rules that every candidate need to follow are - ")
engine.say("The test-takers should not take the test from a public place or a noisy background")
engine.say("Candidates should not be taking the test sitting in an informal posture on a couch or a bed")
engine.say("There should not be any other person present at the test-taking zone")
engine.say("Candidates should not be talking to else apart from the proctor during the test")
engine.say("Candidates should not have any personal belonging while taking BUMAT, such as mobile phones, earphones, scientific calculator, books, stationery items, etc.")
engine.say("if the candidate is found doing any suspicious activity his/her exam will be cancelled out . you will be given only 5 warning before that")
engine.runAndWait()

# ............................................
KNOWN_FACES_DIR = "C:\\Users\\hp\\face_photos"
# UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 0.6 # the lower the tolerance the less chance you have to get false positive
# the more is the tolerance the more there will be matching and it can be more risk of incorrect matching.
# if tolerance is less then matching will be mostly correct and there can be less mtching and more flase negative

FRAME_THICKNESS = 3 # size of frame around face. It depends on how big your image is.
FONT_THICKNESS = 2
MODEL="hog" # or we can use "cnn" model of we use gpu because "ccn" is slow in cpu

video = cv2.VideoCapture(0)

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
d = 0
talk("starting candidate verification ........")
while True:
    if d==5:
        talk("you have reached you maximum limit of warning ,due to illegal activies you exam is cancelled")
        break
    ret, image = video.read()

    locations = face_recognition.face_locations(image, model=MODEL)  # A list of tuples of found face locations in css (top, right, bottom, left) order

    encodings = face_recognition.face_encodings(image, locations)

    if locations == []:
        cv2.putText(image, "No Face is detected in this image", (0 + 10, 0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 0), FONT_THICKNESS)
        d = d + 1
        talk("warning " + str(d) + "suspicious object detected")

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f' - {match} from {results}')
            x = str(match)
            if x=="harshith" and c==0:
                c = 2
                talk("candidate verfied as "+x)
                talk("you can now start your exam")
            # Each location contains positions in order: top, right, bottom, left
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            color = [0, 255, 0]  # BGR

            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (43, 37, 9), FONT_THICKNESS)
        else:
            print(f"Face is not recognised")
            d = d+1
            talk("warning "+d+"suspicious object detected")
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

video.release()
cv2.destroyAllWindows()