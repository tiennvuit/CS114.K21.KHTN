import base64
import urllib.parse
import requests
import json
import sys
import argparse
from user_login import get_token
import time
from PIL import Image, ImageDraw
import cv2
import os

url = 'http://service.mmlab.uit.edu.vn/mmlab_api/object_detect'


def get_boundingBoxes(image_path: str):
    """
    Input:

    Output: json file that store informations of objects in image.
    """
    # Get token from API Mmlab.
    token = get_token()
    image = open(image_path, 'rb')
    image_read = image.read()
    encoded = base64.encodebytes(image_read)
    encoded_string = encoded.decode('utf-8')
    data = {
        'api_version': '1.0',
        'data': {
            'method': 'yolov3',
            'model_id': '1575949128.9343169',
            'images': [encoded_string]
        }
    }
    headers = {'Content-type': 'application/json', 'Authorization': "bearer " + token}
    data_json = json.dumps(data)
    response = requests.post(url, data = data_json, headers=headers)
    return response.json()

def draw_bounding_boxes(image, bboxes: json):
    """
    @arguments:
        - image is the image object read by OpenCV
        - bboxes (json) is the information about objects in image get from API
    @return values
        - An image drawed bounding boxes and the name class for each objecs.
    """

    # Get all bounding boxes, class name, scores from response got from API
    bounding_boxes = bboxes['data']['predicts'][0]   # An array contains all bounding boxes of detected object in image.

    # Draw bounding boxes and class name, score to image.
    for bounding_box in bounding_boxes:
        box = bounding_box['bbox']
        box[0] = int(box[0] - box[2]/2)
        box[1] = int(box[1] - box[3]/2)
        label = str(bounding_box['object'].capitalize())
        score = str(round(bounding_box['score'], 2))
        text = label + "-" + score
        cv2.rectangle(image, box, color=(0, 255, 0), thickness=2)
        cv2.putText(image, text, (int(box[0]), int(box[1]-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 1)
    return image

def main(args):
    image_path = str(args.path)

    # Check the file directory whether exit or not.
    if not os.path.exists('input/'+image_path):
        print("\nThe path of image is not exist !\nCheck again in /input folder.")
        exit(1)

    bboxes = get_boundingBoxes(image_path="input/"+image_path)
    image = cv2.imread("input/"+image_path)

    # Get the image after draw bounding boxes
    detected_image = draw_bounding_boxes(image=image, bboxes=bboxes)

    # Save image
    cv2.imwrite('output/detected_'+ image_path, detected_image)

    # Display imagiie
    saved_image = Image.open('output/detected_' + image_path)
    saved_image.show(title="Detect multiable objects")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The program using API of Mmlab-UIT to detect multiable object")
    parser.add_argument("-path", help="The path of image", default="test.png")
    args = parser.parse_args()
    main(args)

    # print("\n---------------------------------------Many thanks for API's MMLAB-UIT----------------------------------")
    # print("<3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3")
