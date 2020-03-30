import base64
import urllib.parse
import requests
import json

url = 'http://service.mmlab.uit.edu.vn/mmlab_api/user_login/post'

def get_token():
    data ={'user_name': 'admin', 'password': 'admin'}
    headers = {'Content-type': 'application/json'}
    data_json = json.dumps(data)
    response = requests.post(url, data = data_json, headers=headers)
    dictionary = response.content.decode()
    index = dictionary.find("token")
    index = index + 8
    return dictionary[index:-3]

def main():
    token = get_token()
    print(token)

if __name__ == "__main__":
    main()
