from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from imutils import paths
import numpy as np
import argparse
import pickle
import cv2
import os

def get_arguments():
	# construct the argument parser and parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--dataset", default="dataset",
		help="path to input dataset")
	parser.add_argument("-m", "--model", type=str, default="liveness.model",
		help="path to trained model")
	parser.add_argument("-l", "--le", type=str, default="le.pickle",
		help="path to label encoder")
	parser.add_argument("-p", "--plot", type=str, default="plot.png",
		help="path to output loss/accuracy plot")
	return vars(parser.parse_args())

def load_dataset(dataset_path: str):
    # grab the list of images in our dataset directory, then initialize
    # the list of data (i.e., images) and class images
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(dataset_path))
    data = []
    labels = []

    # loop over all image paths
    for imagePath in imagePaths:
    	# extract the class label from the filename, load the image and
    	# resize it to be a fixed 32x32 pixels, ignoring aspect ratio
    	label = imagePath.split(os.path.sep)[-3]
    	image = cv2.imread(imagePath)
    	image = cv2.resize(image, (32, 32))

    	# update the data and labels lists, respectively
    	data.append(image)
    	labels.append(label)

    # convert the data into a NumPy array, then preprocess it by scaling
    # all pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0

    # encode the labels (which are currently strings) as integers and then
    # one-hot encode them
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = to_categorical(labels, 2)

    return data, labels, le

def plot_progress():
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])
