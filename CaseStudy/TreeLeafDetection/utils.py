import argparse
import os
import numpy as np
from PIL import Image
from config import *
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier

DATASET_DIR = "Folio Leaf Dataset/Folio/"

def getArgument():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-extractor', help='The feature extractor', default="HOG")
    parser.add_argument('-classifier', help="The classifier use", default="kNN")
    return parser.parse_args()

# Load dataset into program
def loadDataset():
    dataset = []
    labelset = []
    # Loop through the label class
    for label in os.listdir(DATASET_DIR):
        for file in os.listdir(DATASET_DIR+label):
            image = Image.open(os.path.join(DATASET_DIR,label,file))
            # Resize the image object
            image = image.resize(IMAGE_SIZE, Image.NEAREST)
            print("{} is resized to {} size".format(file, IMAGE_SIZE))
            # Add the image to the dataset
            dataset.append(image)
            # Add the cooresponding label of the image
            labelset.append(label)

    #getInformationDataSet(dataset=dataset)
    print("Load dataset successfully !")
    return (dataset, labelset)

def featureExtraction(dataset, method):
    feature_vectors = []
    for image in dataset:
        feature_vector = featureExtracting(image, method)
            feature_vectors.append(feature_vector)
    return feature_vectors

def featureExtracting(image, method: str):
    # Use the extracting feature method like HOG, SIFT, Deep Fetures
    if method == "HOG":
        fd, hog_image = hog(image=image, orientations=8, pixels_per_cell=(16,16), 
                            cells_per_block=(1, 1), visualize=True, multichannel=True)
        # Convert hog_image to hog feeature
        # hog_vector = 
        reutrn hog_vector

    if method == "SIFT":
        ### ... 
        return image
    if method == "CNN":
        ### ....
        return image
    print("Feature extractor is not exist !")
    exit(1)


def getInformationDataSet(dataset):
    print("Kích thước tập dữ liệu: {}".format(len(dataset)))
    sizes = {}
    for image in dataset:
        if image.size not in sizes.keys():
            sizes[image.size] = 1
        else:
            sizes[image.size] += 1
    print("Thống kê kích thước của tập dữ liệu: ")
    for size in sizes.keys():
        print("Shape: {} have {} images".format(size, sizes[size]))

    
def trainModel(training_set, classifier: str):
    if classifier == "kNN":
        # Train model using kNN algorithms
        model = KNeighborsClassifier(n_neighbors=32, weights='uniform', algorithm='auto')
        model.fit(X, y)
        return model
    if classifier == 'SVM':
        # Train model using SVM algorithm
