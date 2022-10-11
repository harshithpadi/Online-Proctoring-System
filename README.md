# Online-Proctoring-System
In this project, we present a artificial intelligent system that performs automatic online exam proctoring without the help of any human proctoring.

## Abstract -
Human proctoring is the most common approach of evaluation, by either requiring the test taker to visit an examination centre, or by monitoring them visually and acoustically during exams via a webcam. However, such methods are labour – intensive. In this project, we present a artificial intelligent system that performs automatic online exam proctoring without the help of any human proctoring. The system hardware includes one webcam, and a microphone, for the purpose of monitoring the visual and acoustic environment of the testing location

## HOG (Histogram of Oriented Gradients) -
It is used in computer vision and image processing for the purpose of object detection. The technique counts occurrences of gradient orientation in the localized portion of an image. This method is quite similar to Edge Orientation Histograms and Scale Invariant aFeature Transformation

HOG is a simple and powerful feature descriptor. HOG is robust for object detection because object shape is characterized using the local intensity gradient distribution and edge direction.

Steps - 

Step1: The basic idea of HOG is dividing the image into small connected cells

Step2: Computes histogram for each cell.

Step3: Bring all histograms together to form feature vector i.e., it forms one histogram from all small histograms which is unique for each face

The only disadvantage with HOG based face detection is that it doesn’t work on faces at odd angles, it only works with straight and front faces. It is really useful if you use it to detect faces from scanned documents like driver’s license and passport but not a good fit for real-time video.

## YOLO(You Only Look Once) Algorithm:

YOLO algorithm employs convolutional neural networks (CNN) to detect objects in real-time. As the name suggests, the algorithm requires only a single forward propagation through a neural network to detect objects. This means that prediction in the entire image is done in a single algorithm run.

## Steps Involved:

![image](https://user-images.githubusercontent.com/115480440/194987625-649049fc-10e9-4010-ada2-fd752177c334.png)

## CONCLUSION - 

So , online proctoring system can be much more effective by using this model . detecting and recognizing the candidate and continuously verifies the test taker if he is following the rules of online proctoring system .
