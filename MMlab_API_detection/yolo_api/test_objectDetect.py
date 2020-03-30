import base64
import urllib.parse
import requests
import json
import sys
url = 'http://service.mmlab.uit.edu.vn/mmlab_api/object_detect'
######################
image_path = "test2.jpg"
image = open(image_path, 'rb')
image_read = image.read()
encoded = base64.encodestring(image_read)
encoded_string = encoded.decode('utf-8')
######################
data ={'api_version': '1.0', 'data': {'method': 'yolov3', 'model_id': '1575949128.9343169', 'images': [encoded_string, encoded_string]}}
headers = {'Content-type': 'application/json', 'Authorization': "bearer "+sys.argv[1]}
data_json = json.dumps(data)
response = requests.post(url, data = data_json, headers=headers)
print("response", response)
print(response.json())
################################################################
