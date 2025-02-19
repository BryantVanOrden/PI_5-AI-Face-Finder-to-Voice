# FaceCheck Embedded Systems Project

## Overview

This project uses a Rassber P5 with an AI accelerator module and 5 MP camera to capture and Analyze this video feed in real time . It uses a TensorFlow Lite to detect faces and extract facial Landmarks. The system captures a picture when a SENGLE face is into a good view. Later the image is sent to the FaceCheck API and processed. The system also displays a Tiktiner uI UII with live video feed, the last captured face, and log messages.

------------------------------------------------------------

## Project Version

This Project automatically downloads the face_landmark_tly.tfle model from the [Patlevin/face-detection-tfle](https://github.com/patlevin/face-detection-tfle) if not present. As with the model, detection and landmarks are extracted. The code captures a picture when a single face is detected, and then sends the image to the FaceCheck API, processes a text writeup file with the API response. The system also provides Audio feedback via a Bluetooth Headset to take audio notifications.

## Features ---

- * Face detection and Landmarks - Uses the provided model from [Patlevin/face-detection-tfle](https://github.com/patlevin/face-detection-tfle) for face detection and extraction.
-- * FaceCheck API Integration - Sends the captured image to the API using their proscribed two-step process, processes a will avoid duplicates and save results in the captured faces.
-- * UII feedback - Uses ti ktiner Ui UII with live video feed, the last captured image, and log messages.
------------------------------------------------------------

## Requirements


  - Rasspber Pi 5 with a secure Phyon 3.X.
    - Open CV (cv)
    - Pillowe (HTTP)
    - tflie_runtime
    - pyttsx3
    - requests
    - tiktiner 
    - numpy

## Setup
 
1. _clone or Download the repository
    - place the code file (eg. `main.py`.) in your project directory.

2. Install Dependencies
      - OpenCV ((cv) 
      - Pillowe (PILL)
      - tflie_runtime
      - pyttsx3
       - requests
    - tiktiner
 
#3. Configuration
    - Replace Y_AP_TOKEN in the code with your actual API_TOKEN.
    - Optionally ajest the TESTING_MODEE, COLWEND, and other configuration variables as needed.
    - Ensure that your Rassber P is configured with the correct camera interface and Bluetooth Headsettings.

## Running the Application
Run the python application:

    past main.py

## How It Works 

 1. Model Download and Setup
    Script checks for the file 'face_landmark.tfle' and downloads it after it does
    Use the tensorflow model (for example, we use the model from [Patlevin/face-detection-tfle](https://github.com/patlevin/face-detection-tfle)).

2. Face Technology API Integration - Sends the captured image to the FaceCheck API using their prosicbled two-step process. The system polles for result, processes the response , and processes the result to avoid duplicate captures.
 
3. UI feedback & TLC Care - Uses tektiner UI UI with live video feed, the last captured image, and log messages.

 * VI UI and tetts-to-speech finally be saved in the captured_faces directory. 

3# Running the Application

Run the python application:

    past main.py



## How It Works

- Automatically downloads the model (face_landmark.tfle) if not present.

-- Free Site: [Patlevin/face-detection-tfle](https://github.com/patlevin/face-detection-tfle)

## Conditions
- Rassper P 1 with the correct camera and Bluetooth Headsettings.
- Install properly dependencies with python 3.x

## Her Token

CREOW [Your API_TOKEN](RESPOSE with your actual PAI_TOKEN)

## Running the Application


==============================================================

Litkapti UI Application (including the live video feed, the last captured image, and log messages)

-- With the camera and AUII configuration, the application shows the live video feed, detects faces, captures and sends the image to the FaceCheck API, aputs audio feedback.

==============================================================


Conidents,
- * Script contains the face detection and face Landmarks
- * FaceCheck API integration
- * Ui UI feedback to approach duplicate captures and data Logs.
- * Data Logging - Saves captured images and API responses in the captured_faces directory.

==============================================================

## Throubeghtout 


- For Camera Issues : Ensure that your camera is connected and configured.
- For Model Download Failures : Verify the URL and URL updates to be proper.
 

- FACE FACE - UI and TYRTES_TICE_IN support.

## Acknowledgements
- [automatical Mats](https://github.com/patlevin/face-detection-tfle)
- FaceCheck API Documentation
