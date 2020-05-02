import base64
import urllib.parse
import requests
import json

url = 'http://service.mmlab.uit.edu.vn/mmlab_api/user_login/post'

def get_token():
    # The user_name, password we need to provided by administrators of MMlab.
    # To get the account (username, password), let contact Mmlab via email:
    #                     mmlab@uit.edu.vn
    # Use the token providing bellow. 

    data ={'user_name': 'admin', 'password': 'admin'}
    headers = {'Content-type': 'application/json'}
    data_json = json.dumps(data)
    response = requests.post(url, data = data_json, headers=headers)
    dictionary = response.content.decode()
    index = dictionary.find("token")
    index = index + 8
    return dictionary[index:-3]

    # return "eyJhbGciOiJIUzUxMiIsImlhdCI6MTU4NTU4MzkyNSwiZXhwIjoxNjE3MTE5OTI1fQ.eyJ1c2VybmFtZSI6ImFkbWluIn0.NZz--dSYal5wgSpUoNYuK5R37kHCnxC8NeKb9oI42kD0P4dagN0PD1XliKInqQGBNJEcZvp1PBsZFboYD09JYg"

def main():
    token = get_token()
    print(token)

if __name__ == "__main__":
    main()
