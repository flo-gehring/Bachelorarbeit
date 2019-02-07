import json 
import cv2

import argparse


def get_bottom_left(x, y, width, heigth):
    return (x + width, y + heigth)


parser = argparse.ArgumentParser()

parser.add_argument("videoname")

args = parser.parse_args()

name = args.videoname

qualified_videopath = "/home/flo/Videos/" + name

parsed_json = json.load(open(name[0:-4]+".json", "r"))

cap = cv2.VideoCapture(qualified_videopath)

print (qualified_videopath)
ret, frame = cap.read()

framecounter = 0
while ret:
    
    annotations_for_frame = parsed_json[framecounter]["detections"]
    for annotation in annotations_for_frame:

            x = int(annotation["x"])
            y = int(annotation["y"])
            width = int(annotation["width"])
            height = int(annotation["height"])
            

            cv2.rectangle(frame, (x,y), get_bottom_left(x, y, width, height), (102, 0, 255))
    
    framecounter += 1
    frame = cv2.resize(frame, dsize=(1920, 1080), interpolation=cv2.INTER_AREA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    ret, frame = cap.read()


