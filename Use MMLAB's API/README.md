 # Use API Object Detection of MMLAB - UIT.

<img src="output/detected_test3.png" alt="Detect objects in image" style="width: 60% height: 200px">

---
## 1. The flow working

1. Login into server and get token.

2. Use the token to using API: the sending data is the encoded image object and the response is the file json that containt all bounding boxes, class name, scores of objects in images.

3. Use the json file to draw rectangles in image and display to screen.

---
## 2. Structure of code
```
Use MMLAB's API/
├── detect_objects.py
├── input
│   ├── test1.png
│   ├── test2.png
│   ├── test3.png
│   └── test.png
├── output
│   ├── detected_test1.png
│   ├── detected_test2.png
│   ├── detected_test3.png
│   └── detected_test.png
└── user_login.py
```
---
## 3. Set up
1. Clone project to your local machine.
```
git clone https://github.com/tiennvuit/CS114.K21.KHTN.git
```

2. Go to the project directory
```
cd CS114.K21.KHTN/Use\ MMLAB\'s\ API/
```
3. Create virtual environment.
```bash
python3 -m venv env
```
4. Activating the virtual environment
```bash
source env/bin/activate
```
5. Install requirement packages into virtual environment:
```bash
pip install -r requirements.txt
```

---
## 4. Run program
To run program, just type the following command:

```
python3 detect_objects.py
```

that will use the input image is *test.png*.

If you want to use your image, just move it to the directory `input/` and run the following command:

```
python3 detect_objects.py -path name_image
```

---
## Note
- To use API, you need the account to login into server. Let contact MMlab to request for using via email: *mmlab@uit.edu.vn*
- But, in this project we can use the providing token included in code so you can use API easily by following step by step about.
