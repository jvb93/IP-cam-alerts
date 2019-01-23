import sched, time
import cv2
import numpy as np
from collections import deque
from imageai.Detection import ObjectDetection
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.message import EmailMessage
from PIL import Image
import io
import getpass
from multiprocessing import Process, Queue, freeze_support
from ast import literal_eval

############################################################################
#######################Configuring settings from docker env#################

# Settings for the motion detector
HISTORY = int(os.environ['MD_HISTORY'])
THRESHOLD = int(os.environ['MD_THRESHOLD'])
# Threshold number for movement to be detected
BASE_MOVEMENT_THRESHOLD = int(os.environ['MOVEMENT_DETECTION_THRESHOLD'])
# Number of frames to detect objects in after movement is first detected
OD_INTERVAL = int(os.environ['OD_FRAMES'])
# Object detection model configuration
DETECTOR_MODEL = os.environ['OD_MODEL']
DETECTION_SPEED = os.environ['OD_SPEED']
OBJECT_DETECTION_CONFIDENCE_THRESHOLD = os.environ['OD_CONFIDENCE_THRESHOLD']
IMAGE_PROCESSING_RESOLUTION = os.environ['IMAGE_PROCESSING_RESOLUTION']
#IP address of the camera
# TODO: Unfuck this. split this value into username, password, and IP, then concat so we can grab the password securely
CAMERA_IP_ADDRESS = os.environ['CAMERA_IP']
# The address to send alerts to
ALERT_ADDRESS = os.environ['ALERT_ADDRESS']
#############################################################################


def getFrames(q):
    """
    This function is meant to be run as a multiprocessing Process object. Grabs frames from the video capture object
    and stores them in the queue for access outside this process
    :param q: multiprocessing.Queue object
    :return: access q from outside function
    """
    print('getframes')
    cap = cv2.VideoCapture(CAMERA_IP_ADDRESS)
	
    while True:
        print('loopin')
        ret, frame = cap.read()
        if q.qsize() >= 5:
            q.get()
        q.put(frame)

def detectMovement(frame):
    """
    Detects movement in frame supplied by checking against the last few frames

    :param frame: numpy array of the frame to be processed
    :return boolean: True if movement detected above specified threshold, else, false
    """
    fg_mask = bg_subtractor.apply(frame)  # Generate a mask of moved parts of the image
    total_movement = int(np.sum(fg_mask == 255))  # Counts the pixels that have changed to the detector
    print(f'Movement level: {total_movement}')

    return total_movement > BASE_MOVEMENT_THRESHOLD


def findObjects(frame):
    """
    Detects objects in the supplied frame

    :param frame: numpy array of an image.
    :return (debugging_image, objects_detected): a tuple, (<a numpy array image with detections>,
                                                            <a list of objects detected >).
    """
    start_time = time.time()
    debugging_image, detections = detector.detectObjectsFromImage(frame, input_type="array", output_type="array",
                                                                 minimum_percentage_probability=int(OBJECT_DETECTION_CONFIDENCE_THRESHOLD))
    objects_detected = [item['name'] for item in detections]
    print(f'Detected the following objects: {objects_detected}')
    finish_time = time.time()
    total_time = finish_time - start_time
    print(f'Took {total_time} seconds to detect objects')

    return debugging_image, objects_detected


def sendAlertEmail(image_list, detections):
    msg = MIMEMultipart()
    msg['Subject'] = 'Object of interest detected'
    msg['From'] = username
    msg['To'] = ALERT_ADDRESS
    msg_text = MIMEText(f'Objects detected:{detections}')
    msg.attach(msg_text)

    image_number = 1
    for image in image_list:
        image = cv2.imencode('.jpeg', image)[1]
        image = image.tobytes()
        msg_image = MIMEImage(image, name='image')
        msg_image.add_header('Content-ID', f'<image{image_number}>')
        msg_image.add_header('Content-Disposition', 'inline')
        msg.attach(msg_image)
        image_number += 1

    email_server.send_message(msg, username, ALERT_ADDRESS)


def objectDetectionLoop(frame):
    image_list = []  # A list to keep images in that have had objects detected in them
    od_start_time = time.time()
    total_detections = []
    for i in range(OD_INTERVAL):
        detect_frame = frame_queue.get()  # This grabs a frame to be processed from the queue being stocked
        detection_image, detection_list = findObjects(detect_frame)  # This is the actual object detection
        od_start_time = time.time()
        total_detections.append(detection_list)
        print(f'Detection  on frame #{i+1}')
        if 'person' in detection_list:  # If there were any objects detected in that frame we shrink it and add it to the list
            image_list.append(detection_image)
    # If any images were added to the list, that means things were detected, and you should send the alert
    if image_list:

        print(f'Sending alert email with {len(image_list)} images attached')
        sendAlertEmail(image_list, total_detections)  #converting list to set to only send unique items


def analyzeVideo():
    # let's just grab the first frame before things get ahead of themselves
    ret, frame = cap.read()
    start_time = time.time()
    while True:
        frame = frame_queue.get()
        frame = cv2.resize(frame, literal_eval(IMAGE_PROCESSING_RESOLUTION))

        fps = 1 / (time.time() - start_time)
        start_time = time.time()
        #print(f'FPS: {fps}')

        font = cv2.FONT_HERSHEY_SIMPLEX
        if detectMovement(frame):
            # annotate movement detection on the frame to be displayed

            # detecting the next OD_INTERVAL number of frames after motion detection
            print('Motion detected.')
            objectDetectionLoop(frame)
        #time.sleep(.5)

if __name__ == '__main__':
    freeze_support()  #Need this to be able to use MP in a standalone package
    print('Enter login information for the email to use for sending alerts...')
    username = os.environ['SEND_USERNAME']
    password = os.environ['SEND_PASSWORD']

    cap = cv2.VideoCapture(CAMERA_IP_ADDRESS)

    # setting up the object detector
    detector = ObjectDetection()
    print('Setting up detector')
    # Configure the detector based on the settings in config.ini
    if DETECTOR_MODEL == 'yolo':
        print('YOLO chosen')
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath('./models/yolo.h5')
    else:
        print('Tiny-YOLO chosen')
        detector.setModelTypeAsTinyYOLOv3()
        detector.setModelPath('./models/yolo-tiny.h5')

    print('Loading model')
    detector.loadModel(detection_speed=DETECTION_SPEED)
    print('Creating BG subtractor')
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=HISTORY,
                                                       varThreshold=THRESHOLD)  # includes params from settings

    print('Starting email server')
    email_server = smtplib.SMTP('smtp.gmail.com', 587)
    email_server.starttls()
    print('Logging in')
    email_server.login(username, password)
    print('Logged in')
    # setting up the queue that will be used to get data from the stream processing
    frame_queue = Queue(maxsize=10)
    # starting the stream processing multiprocess
    print('Setting up process')
    stream_process = Process(target=getFrames, args=(frame_queue,))
    print('Starting process')
    print(f'analyzing {CAMERA_IP_ADDRESS}')
    stream_process.start()
    print('Process started')
    print('Analyzing video...')

    analyzeVideo()
