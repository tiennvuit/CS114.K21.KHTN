import base64
import urllib.parse
import requests
import json
import timeit
import sys
url = 'http://service.mmlab.uit.edu.vn/mmlab_api/run_model'
####################################
data ={'api_version': '1.2', 'data': {'method': 'yolov3', "model_id": "1575949128.9343169"}}
headers = {'Content-type': 'application/json', 'Authorization': "bearer "+sys.argv[1]}
data_json = json.dumps(data)
response = requests.post(url, data = data_json, headers=headers)
print(response)
response = response.json()
print(response)


