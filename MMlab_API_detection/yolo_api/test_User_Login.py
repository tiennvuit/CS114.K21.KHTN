import base64
import urllib.parse
import requests
import json

url = 'http://service.mmlab.uit.edu.vn/mmlab_api/user_login/post/'
#------------------------------------
data ={'user_name': 'admin', 'password': 'admin'}
headers = {'Content-type': 'application/json'}
data_json = json.dumps(data)
response = requests.post(url, data = data_json, headers=headers)
print("response", response)
print(response.json())
################################################################
