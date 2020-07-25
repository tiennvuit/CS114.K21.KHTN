import numpy as np
import argparse
import time
import cv2
import os
from gather_data import extract_and_save_face


def get_arguments():
	# construct the argument parser and parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", type=str, default="ROSE_videos",
		help="path to input video")
	parser.add_argument("-o", "--output", type=str, default="ROSE_dataset",
		help="path to output directory of cropped faces")
	parser.add_argument("-d", "--detector", type=str, default="face_detector",
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


def main(args):
    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

	# Create output directory if it is not exist
    if not os.path.exists(args["output"]):
        os.mkdir(args["output"])

    trainPath = os.path.join(args['input'], "train")
    testPath = os.path.join(args['input'], "test")

    # Load train data
    #print("[INFO] Extracting train video ...")
    #for dir in os.listdir(trainPath):
    #    for folderVideo in os.listdir(os.path.join(trainPath, dir)):
    #        for video in os.listdir(os.path.join(trainPath, dir, folderVideo)):
    #            if not os.path.isdir(os.path.join("ROSE_dataset", "train", dir, folderVideo)):
    #                os.mkdir(os.path.join("ROSE_dataset", "train", dir, folderVideo))
    #            video_path = os.path.join(trainPath, dir, folderVideo, video)
    #            output_path = os.path.join("ROSE_dataset", "train", dir, folderVideo, video.split(".")[-2][-2:])
    #            if not os.path.isdir(output_path):
    #                os.mkdir(os.path.abspath(output_path))
    #            print("Extracting the video {} to {}".format(video_path, output_path))
    #            extract_and_save_face(video_path=video_path, net=net,
    #                                         output_path=output_path, default_confidence= args['confidence'],
    #                                         skip=args['skip'], show=args['show'], threshold=args['threshold'])

    print("[INFO] Extracting test video ...")
    time.sleep(3)

    for dir in os.listdir(testPath):
        for folderVideo in os.listdir(os.path.join(trainPath, dir)):
            for video in os.listdir(os.path.join(trainPath, dir, folderVideo)):
                if not os.path.isdir(os.path.join("ROSE_dataset", "test", dir, folderVideo)):
                    os.mkdir(os.path.join("ROSE_dataset", "test", dir, folderVideo))
                video_path = os.path.join(trainPath, dir, folderVideo, video)
                output_path = os.path.join("ROSE_dataset", "test", dir, folderVideo, video.split(".")[-2][-2:])
                if not os.path.isdir(output_path):
                    os.mkdir(os.path.abspath(output_path))
                print("Extracting the video {} to {}".format(video_path, output_path))
                extract_and_save_face(video_path=video_path, net=net,
                                             output_path=output_path, default_confidence= args['confidence'],
                                             skip=args['skip'], show=args['show'], threshold=args['threshold'])

if __name__ == '__main__':
    args = get_arguments()
    main(args)
