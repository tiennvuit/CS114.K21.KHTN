# USAGE
#  python demo.py --model saved_model\model_ --le label_encoded\label_encoded --detector face_detector

# import the necessary packages
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
from config import ENCODED_LABELS
from classifier.hand_crafted_model import LocalBinaryPatterns

def get_arguments():
	# construct the argument parser and parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model", type=str, required=True,
		help="path to trained model")
	# parser.add_argument("-l", "--le", type=str, required=True,
	# 	help="path to label encoder")
	parser.add_argument("-d", "--detector", type=str, default='face_detector',
		help="path to OpenCV's deep learning face detector")
	parser.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	parser.add_argument("-nP", "--points", type=int, default=24,
		help="The number of points for local binary patterns")
	parser.add_argument("-r", "--radius", type=int, default=8,
		help="The radius for local binary patterns")

	return vars(parser.parse_args())


def main(args):
	# load our serialized face detector from disk
	print("[INFO] Loading face detector...")
	protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
	modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
	net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

	# load the liveness detector model and label encoder from disk
	print("[INFO] Loading face anti-spoofing detector...")
	if args['model'].find('deep') != -1:
		model = load_model(args["model"])
	elif args['model'].find('hand') != -1:
		model = pickle.load(open(args['model'], 'rb'))
	# le = pickle.loads(open(args["le"], "rb").read())

	# initialize the video stream and allow the camera sensor to warmup
	print("[INFO] Starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

	# loop over the frames from the video stream
	if args['model'].find('deep') != -1:

		while True:
			# grab the frame from the threaded video stream and resize it
			# to have a maximum width of 600 pixels
			frame = vs.read()
			frame = imutils.resize(frame, width=800)

			# grab the frame dimensions and convert it to a blob
			(h, w) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

			# pass the blob through the network and obtain the detections and
			# predictions
			net.setInput(blob)
			detections = net.forward()

			# loop over the detections
			for i in range(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated with the
				# prediction
				confidence = detections[0, 0, i, 2]

				# filter out weak detections
				if confidence > args["confidence"]:
					# compute the (x, y)-coordinates of the bounding box for
					# the face and extract the face ROI
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					# ensure the detected bounding box does fall outside the
					# dimensions of the frame
					startX = max(0, startX)
					startY = max(0, startY)
					endX = min(w, endX)
					endY = min(h, endY)

					# extract the face ROI and then preproces it in the exact
					# same manner as our training data
					face = frame[startY:endY, startX:endX]
					try:
						face = cv2.resize(face, (64, 64))
					except:
						continue
					face = face.astype("float") / 255.0
					face = img_to_array(face)
					face = np.expand_dims(face, axis=0)

					# pass the face ROI through the trained liveness detector
					# model to determine if the face is "fake" or "real"
					preds = model.predict(face)[0]
					j = np.argmax(preds)
					label = ENCODED_LABELS[j]

					# draw the label and bounding box on the frame
					if label == 'Real':
						label = "{}: {:.4f}".format(label, preds[j])
						cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
					else:
						label = "{}: {:.4f}".format(label, preds[j])
						cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
					cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

			# show the output frame and wait for a key press
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break

		# do a bit of cleanup
		cv2.destroyAllWindows()
		vs.stop()

	# Using hand-crafted models
	else:

		desc = LocalBinaryPatterns(numPoints=args['points'], radius=args['radius'])

		while True:
			# grab the frame from the threaded video stream and resize it
			# to have a maximum width of 600 pixels
			frame = vs.read()
			frame = imutils.resize(frame, width=800)

			# grab the frame dimensions and convert it to a blob
			(h, w) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

			# pass the blob through the network and obtain the detections and
			# predictions
			net.setInput(blob)
			detections = net.forward()

			# loop over the detections
			for i in range(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated with the
				# prediction
				confidence = detections[0, 0, i, 2]

				# filter out weak detections
				if confidence > args["confidence"]:
					# compute the (x, y)-coordinates of the bounding box for
					# the face and extract the face ROI
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					# ensure the detected bounding box does fall outside the
					# dimensions of the frame
					startX = max(0, startX)
					startY = max(0, startY)
					endX = min(w, endX)
					endY = min(h, endY)

					# extract the face ROI and then preproces it in the exact
					# same manner as our training data
					face = frame[startY:endY, startX:endX]
					hist = desc.describe(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)).reshape(1, -1)

					# pass the face ROI through the trained liveness detector
					# model to determine if the face is "fake" or "real"

					preds = model.predict_proba(np.array(hist))[0]
					j = np.argmax(preds)
					label = ENCODED_LABELS[j]

					# draw the label and bounding box on the frame
					if label == 'Real':
						label = "{}: {:.4f}".format(label, preds[j])
						cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
					else:
						label = "{}: {:.4f}".format(label, preds[j])
						cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
					cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

			# show the output frame and wait for a key press
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break

		# do a bit of cleanup
		cv2.destroyAllWindows()
		vs.stop()

if __name__ == "__main__":
	args = get_arguments()
	main(args)
