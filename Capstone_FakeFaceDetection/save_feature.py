"""
Extract the LBPs feature from dataset and store to disk.
Aim: load the feature vectors more quickly than load image and extract feature.
"""

import os
import argparse
import pickle
from imutils import paths
import cv2
from hand_crafted_model import LocalBinaryPatterns

def get_arguments():
	# construct the argument parser and parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", "-d", default="dataset", help="path to input dataset")
	parser.add_argument('--numPoints', '-nP', type=int, default=24, help='The number of points parameter for LBPs appoach')
	parser.add_argument('--radius', '-r', type=int, default=8, help='The radius parameter for LBPs appoach')
	parser.add_argument('--output', '-o', default='extracted_features/', help='The path of saving file')
	return vars(parser.parse_args())

def main(args):

	desc = LocalBinaryPatterns(args['numPoints'], args['radius'])
	data = []
	labels = []

    # loop over the training images
	for imagePath in paths.list_images(args['dataset']):
        # load the image, convert it to grayscale, and describe it
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		hist = desc.describe(gray)
        # extract the label from the image path, then update the
        # label and data lists
		labels.append(imagePath.split(os.path.sep)[-3])
		data.append(hist)
		print("Extracted the image {} to feature vector.".format(imagePath))

	pickle.dump((data, labels), open(args['output'] + "{}p-{}r.pl".format(args['numPoints'], args['radius']), 'wb'))

if __name__ == '__main__':
    args = get_arguments()
    main(args)
