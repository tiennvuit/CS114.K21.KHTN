# USAGE
# python train.py --dataset dataset --model model --le le.pickle

# set the matplotlib backend so figures can be saved in the background
# import the necessary packages
from classifier.livenessnet import LivenessNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
import datetime
from utils import get_arguments, load_dataset, plot_progress
from config import *


def main(args):
	# Load dataset to memory
	data, labels, le = load_dataset(dataset_path=args["dataset"])

	# convert the data into a NumPy array, then preprocess it by scaling
    # all pixel intensities to the range [0, 1]
	data = np.array(data, dtype="float") / 255.0

	# partition the data into training and testing splits using 75% of
	# the data for training and the default remaining 25% for testing
	(trainX, testX, trainY, testY) = train_test_split(data, labels,
		test_size=TEST_SIZE, random_state=42)

	# construct the training image generator for data augmentation
	aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
		width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
		horizontal_flip=True, fill_mode="nearest")

	# initialize the optimizer and model
	print("[INFO] compiling model...")
	opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model = LivenessNet.build(width=32, height=32, depth=3,
		classes=len(le.classes_))
	model.compile(loss="binary_crossentropy", optimizer=opt,
		metrics=["accuracy"])

	# train the network
	print("[INFO] training network for {} epochs...".format(EPOCHS))
	H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
		validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
		epochs=EPOCHS)

	# evaluate the network
	print("[INFO] evaluating network...")
	predictions = model.predict(x=testX, batch_size=BS)
	print(classification_report(testY.argmax(axis=1),
		predictions.argmax(axis=1), target_names=le.classes_))

	# save the network to disk
	day = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	print("[INFO] serializing network to '{}'...".format("saved_model/model" + day))
	model.save("saved_model/model_" + day, save_format="h5")

	# save the label encoder to disk
	f = open("label_encoded/label_encoded" + day + ".pl", "wb")
	f.write(pickle.dumps(le))
	f.close()

	plot_progress(model=H, name=day)

if __name__ == "__main__":
	args = get_arguments()
	main(args)
