'''
OpenCV (Open Source Computer Vision) is a library with functions that mainly aiming real-time computer vision.
'''
# Import the necessary libraries
import cv2
import numpy as np
import argparse
from flask import Flask, render_template, Response, request
from PIL import Image
import io
import os

# Set the flask
UPLOAD_FOLDER = './UPLOAD_FOLDER'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set a helper function to get the input face part
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    ## Grab the frame dimensions and convert it to a blob
    blobInput = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], 
                                        swapRB=True, crop=False)
    ## Pass the blob through the network and obtain the detections and predictions
    net.setInput(blobInput)
    ## net.forward() method detects the faces and stores the data in detects
    detects = net.forward()
    
    ## Set an empty frame for face
    faceBoxes = []
    ## Iterate rectangle on detected face.
    for face in range(detects.shape[2]):  
        ### Extract the confidence (i.e., probability) associated with the prediction
        conf = detects[0, 0, face, 2]
        ### Compare it to the confidence threshold
        if conf > conf_threshold:   
            #### Compute the (x, y)-coordinates of the bounding box for the face if confidence detected face > 0.7
            x1 = int(detects[0, 0, face, 3]*frameWidth)
            y1 = int(detects[0, 0, face, 4]*frameHeight)
            x2 = int(detects[0, 0, face, 5]*frameWidth)
            y2 = int(detects[0, 0, face, 6]*frameHeight)
            #### Drawing the bounding box of the face
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes

# Gives input img to the prg for detection
# Using argparse library which was imported
parser = argparse.ArgumentParser()
# If the input argument is not given it will skip this and open webcam for detection
parser.add_argument('--image')
args = parser.parse_args()

'''
Each model comes with two files: weight file and model file, exported from keras model
The .pb file contains the weights for the actual layers
The .pbtxt file define the model architecture.
'''

# Define function for live detection using webcam
def liveDetect():
    ## Use OpenCV model for face detection helper and define each path
    faceBinBuff = './models/opencv_face_detector_uint8.pb'
    faceTextBuff = './models/opencv_face_detector.pbtxt'
    genderBinBuff = './models/gender_model.pb'
    genderTextBuff = './models/gender_model.pbtxt'
    ageBinBuff = './models/age_model.pb'
    ageTextBuff = './models/age_model.pbtxt'

    ## Load the model networks using DNN OpenCV
    faceNet = cv2.dnn.readNet(faceBinBuff, faceTextBuff)
    genderNet = cv2.dnn.readNet(genderBinBuff, genderTextBuff)
    ageNet = cv2.dnn.readNet(ageBinBuff, ageTextBuff)

    ## Set mean values for RGB
    RGB_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    ## Open a video file for camera stream
    video = cv2.VideoCapture(0)
    padding = 20
    while cv2.waitKey(1) < 0:
        ### Read frame
        hasFrame, frame = video.read()
        if not hasFrame:
            cv2.waitKey()
            break

        ### Detect number of faces from the input frame by defined helper function
        imgResult, faceBoxes = highlightFace(faceNet, frame)
        if not faceBoxes:
            #### Print the message if no face detected   
            print("No face detected")   

        ### Iterate over if there is face detected
        for faceBox in faceBoxes:
            #### Display facebox
            face = frame[max(0, faceBox[1]-padding):min(faceBox[3]+padding, frame.shape[0]-1),
                         max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

            #### Set the input blob for gender predict
            blobGender = cv2.dnn.blobFromImage(
                face, 1.0/255, (224, 224), (0,0,0), 
                swapRB=True, crop=False)
            genderNet.setInput(blobGender)

            #### Detect the gender of each face detected
            genderPreds = genderNet.forward()

            #### Process the result
            if genderPreds[0] > 0.5:
                gender = 'Man'
                col = (255, 0, 0)
            else:
                gender = 'Woman'
                col = (203, 12, 255)

            #### Display the result in the console
            print('Gender: {}'.format(gender)) 

            #### Set the input blob for age predict
            blobAge = cv2.dnn.blobFromImage(
                face, 1.0, (224, 224), (0,0,0), 
                swapRB=True, crop=False)
            ageNet.setInput(blobAge)

            #### Detect the age of each face detected
            agePreds = ageNet.forward()
            
            #### Process the result
            age = int(agePreds[0])
            
            #### Display the result in the console
            print('Age: {}'.format(age)) 

            #### Show the result on the output frame
            cv2.putText(imgResult, '{}, {}'.format(gender, age), (
                faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)

            #### Skip if there is no result
            if imgResult is None:
                continue

            #### Convert the result to bytes
            _, encodedImg = cv2.imencode('.jpg', imgResult)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')


# Define function for image upload 
def gen_frames_photo(img_file):
    ## Use OpenCV model for face detection helper and define each path
    faceBinBuff = './models/opencv_face_detector_uint8.pb'
    faceTextBuff = './models/opencv_face_detector.pbtxt'
    genderBinBuff = './models/gender_model.pb'
    genderTextBuff = './models/gender_model.pbtxt'
    ageBinBuff = './models/age_model.pb'
    ageTextBuff = './models/age_model.pbtxt'

    ## Load the model networks using DNN OpenCV
    faceNet = cv2.dnn.readNet(faceBinBuff, faceTextBuff)
    genderNet = cv2.dnn.readNet(genderBinBuff, genderTextBuff)
    ageNet = cv2.dnn.readNet(ageBinBuff, ageTextBuff)

    ## Set mean values for RGB
    RGB_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    ## Get the input image
    frame = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
    padding = 20
    while cv2.waitKey(1) < 0:
        ### Detect number of faces from the input frame by defined helper function
        imgResult, faceBoxes = highlightFace(faceNet, frame)
        if not faceBoxes:
            #### Print the message if no face detected   
            print("No face detected")   

        ### Iterate over if there is face detected
        for faceBox in faceBoxes:
            #### Display facebox
            face = frame[max(0, faceBox[1]-padding):min(faceBox[3]+padding, frame.shape[0]-1),
                         max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

            #### Set the input blob for gender predict
            blobGender = cv2.dnn.blobFromImage(
                face, 1.0/255, (224, 224), tuple(val/255. for val in RGB_MEAN_VALUES), 
                swapRB=True, crop=False)
            genderNet.setInput(blobGender)

            #### Detect the gender of each face detected
            genderPreds = genderNet.forward()

            #### Process the result
            if genderPreds[0] > 0.5:
                gender = 'Man'
                col = (255, 0, 0)
            else:
                gender = 'Woman'
                col = (203, 12, 255)

            #### Display the result in the console
            print('Gender: {}'.format(gender)) 

            #### Set the input blob for age predict
            blobAge = cv2.dnn.blobFromImage(
                face, 1.0, (224, 224), (0,0,0), 
                swapRB=True, crop=False)
            ageNet.setInput(blobAge)

            #### Detect the age of each face detected
            agePreds = ageNet.forward()
            
            #### Process the result
            age = int(agePreds[0])
            
            #### Display the result in the console
            print('Age: {}'.format(age)) 

            #### Show the result on the output frame
            cv2.putText(imgResult, '{}, {}'.format(gender, age), (
                faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)

            #### Skip if there is no result
            if imgResult is None:
                continue

            #### Convert the result to bytes
            _, encodedImg = cv2.imencode('.jpg', imgResult)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(liveDetect(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['fileToUpload'].read()
        img = Image.open(io.BytesIO(f))
        img_ip = np.asarray(img, dtype="uint8")
        print(img_ip)
        return Response(gen_frames_photo(img_ip), mimetype='multipart/x-mixed-replace; boundary=frame')
        # return 'file uploaded successfully'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
