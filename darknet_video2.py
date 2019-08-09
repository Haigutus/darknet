#from ctypes import *
#import math
#import random
import os
import cv2
#import numpy as np
import datetime
import darknet
import requests

from uuid import uuid4


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def detections_to_json(detections, detections_dict):

    detections_dict["data"] = []
    #k_width  = detections_dict["meta"]["k_width"]
    #k_height = detections_dict["meta"]["k_height"]

    print(k_width, k_height)

    for detection in detections:

        x, y, w, h = detection[2][0] / k_width,\
                     detection[2][1] / k_height,\
                     detection[2][2] / k_width,\
                     detection[2][3] / k_height

        xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))

        detections_dict["data"].append(
             dict(
             lable = detection[0].decode(),
             confidence = round(detection[1],4),
             xmin = xmin,
             ymin = ymin,
             xmax = xmax,
             ymax = ymax))

    return detections_dict


def cvDrawBoxes(detections, img):
    for detection in detections:

        print(k_width, k_height)

        x, y, w, h = detection[2][0] / k_width,\
                     detection[2][1] / k_height,\
                     detection[2][2] / k_width,\
                     detection[2][3] / k_height

        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img



k_width = int()

def dedect_and_post(source, destiantion = "print", configPath = "./cfg/yolov3.cfg", weightPath = "./yolov3.weights", metaPath   = "./cfg/coco.data"):
    """source = ip to stream or path to video file
       destination = "print" or ip where to post the json detection object"""

    # Test if all paths exist
    for path in [source, configPath, weightPath, metaPath]:

        path_exsists = os.path.exists(path)
        print(path, path_exsists)

        # Stop process if file is missing
        if not path_exsists:
            quit()


    netMain  = darknet.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    metaMain = darknet.load_meta(metaPath.encode("ascii"))

    width = int(darknet.network_width(netMain))
    height = int(darknet.network_height(netMain))
    print("Detection frame size")
    print(width, height)

    darknet_image = darknet.make_image(int(width), int(height),3) # Build darkent image
    cap = cv2.VideoCapture(source)                      # Open stream or file

    original_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))     # float
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # float
    original_FPS    = cap.get(cv2.CAP_PROP_FPS)

    global k_width
    global k_height

    print(width,original_width)

    k_width = float(width)/float(original_width)
    k_height = float(height)/float(original_height)

    print(k_width, k_height)

    print("Stream/file frame size")
    print(original_width, original_height)

    # Check if camera opened successfully
    if (cap.isOpened()== False):
      print("Error opening video stream or file")

    start_time = datetime.datetime.utcnow()

    detections_dict = {}
    detections_dict["meta"] = dict( height = original_height,
                                    width = original_width,
                                    source = source,
                                    stream_start_utc = start_time.isoformat(),
                                    FPS = original_FPS,
                                    #k_width = k_width,
                                    #k_height = k_height
                                    )

    frame_number = 0
    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            frame_resized = cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_LINEAR) # Resize frame to fit darknet

            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)

            #print(detections)



            detections_dict["meta"]["frame_timestamp_utc"]  = datetime.datetime.utcnow().isoformat()
            detections_dict["meta"]["frame_number"]         = frame_number
            detections_dict["meta"]["frame_UUID"]           = str(uuid4())


            meta_json = detections_to_json(detections, detections_dict)


            # send to destination or print the dedection result
            if destiantion == "print":
                print(meta_json)
            else:
                reponse = requests.post(destiantion, json=meta_json)
                print(reponse)


            # image = cvDrawBoxes(detections, frame)
            # Display the resulting frame
            #cv2.imshow('Frame', image)
            # Press Q on keyboard to  exitq
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_number += 1

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()



if __name__ == "__main__":

    source = "ThreodMarduk.mp4"

    #destiantion = "https://ptsv2.com/t/68y5a-1552333596/post"
    destiantion = "print"

    dedect_and_post(source, destiantion)
