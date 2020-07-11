# USAGE
# python gather_data.py --input videos/real/real01.mov --output dataset/real/real01 --detector face_detector --skip 5
# python gather_data.py --input videos/fake/fake01.mov --output dataset/fake/fake01 --detector face_detector --skip 5

# import the necessary packages
import numpy as np
import argparse
import cv2
import os

from blur_detection import detect_blur


def get_arguments():
	# construct the argument parser and parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", type=str, required=True,
		help="path to input video")
	parser.add_argument("-o", "--output", type=str, required=True,
		help="path to output directory of cropped faces")
	parser.add_argument("-d", "--detector", type=str, required=True,
		help="path to OpenCV's deep learning face detector")
	parser.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	parser.add_argument("-s", "--skip", type=int, default=5,
		help="# of frames to skip before applying face detection")
	parser.add_argument("-sh", "--show", type=int, default=0,
		help="Show or not show the progress via windows")
	parser.add_argument("-th", "--threshold", type=float, default=100.0,
		help="The threshold to accept the bluring frame")
	args = vars(parser.parse_args())
	return args


def extract_and_save_face(video_path: str, net: object, output_path: str, default_confidence: float, skip: int, show:bool, threshold: float):
	# open a pointer to the video file stream and initialize the total
	# number of frames read and saved thus far
	vs = cv2.VideoCapture(video_path)
	read = 0
	saved = 0

	# loop over frames from the video file stream
	while True:
		# grab the frame from the file
		(grabbed, frame) = vs.read()
		
		if video_path.endswith(".mov"):
			frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE);

		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not grabbed:
			break

		# increment the total number of frames read thus far
		read += 1

		# check to see if we should process this frame
		if read % skip != 0:
			continue

		# grab the frame dimensions and construct a blob from the frame
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
			(300, 300), (104.0, 177.0, 123.0))

		# pass the blob through the network and obtain the detections and
		# predictions
		net.setInput(blob)
		detections = net.forward()

		# ensure at least one face was found
		if len(detections) > 0:
			# we're making the assumption that each image has only ONE
			# face, so find the bounding box with the largest probability
			i = np.argmax(detections[0, 0, :, 2])
			confidence = detections[0, 0, i, 2]

			# ensure that the detection with the largest probability also
			# means our minimum probability test (thus helping filter out
			# weak detections)
			if confidence > default_confidence:
				# compute the (x, y)-coordinates of the bounding box for
				# the face and extract the face ROI
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				face = frame[startY:endY, startX:endX]
				# print(face.shape)
				# input()
				check_blur, _ = detect_blur(image=face, threshold=threshold)
				if check_blur:
					continue

				# write the frame to disk
				p = os.path.sep.join([output_path, str(saved).zfill(4) + ".png"])
				cv2.imwrite(p, face)
				saved += 1
				print("[INFO] saved {} to disk".format(p))

		if show == True:
			cv2.imshow('Frame', frame)
			key = cv2.waitKey(1) & 0xFF
			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break

	# do a bit of cleanup
	vs.release()
	cv2.destroyAllWindows()



def main(args):
	# load our serialized face detector from disk
	print("[INFO] loading face detector...")
	protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
	modelPath = os.path.sep.join([args["detector"],
		"res10_300x300_ssd_iter_140000.caffemodel"])
	net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

	# Create output directory if it is not exist
	if not os.path.exists(args["output"]):
		os.mkdir(args["output"])

	extract_and_save_face(video_path=args["input"],
							net=net,
							output_path=args["output"],
							default_confidence=args["confidence"],
							skip=args["skip"],
							show=args["show"],
							threshold=args["threshold"])


if __name__ == '__main__':
	args = get_arguments()
	main(args=args)
