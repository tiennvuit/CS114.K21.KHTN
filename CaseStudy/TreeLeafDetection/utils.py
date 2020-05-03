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
    parser.add_argument('-image_path', help="The test image", default="test0.png")
    return parser.parse_args()

# Load dataset into program
def loadDataset():
    print("Step 1. Loading dataset")
    print("\tLoading dataset ...")
    dataset = []
    labelset = []
    # Loop through the label class
    for label in os.listdir(DATASET_DIR):
        for file in os.listdir(DATASET_DIR+label):
            image = Image.open(os.path.join(DATASET_DIR,label,file))
            # Resize the image object
            image = image.resize(IMAGE_SIZE, Image.NEAREST)
            # Add the image to the dataset
            dataset.append(image)
            # Add the cooresponding label of the image
            labelset.append(label)

    print("\tLoad dataset successfully !")
    print("\tThe size of dataset is {}".format(len(labelset)))
    return (dataset, labelset)

def featureExtraction(dataset, method):
    print("Step 2. Feature extraction")
    print("\tExtracting feature vectors from dataset....")
    feature_vectors = []
    for image in dataset:
        feature_vector = featureExtracting(image, method)
        feature_vectors.append(feature_vector)
    print("\tFeature extraction successfully !")
    return feature_vectors

def featureExtracting(image, method: str):
    # Use the extracting feature method like HOG, SIFT, Deep Fetures
    if method == "HOG":
        fd, hog_image = hog(image=image, orientations=8, pixels_per_cell=(16,16), 
                            cells_per_block=(1, 1), visualize=True, multichannel=True)
        hog_feature = hog_image[0]
        return hog_feature

    if method == "SIFT":
        ### ... 
        return image
    if method == "CNN":
        ### ....
        return image
    print("Feature extractor is not exist !")
    exit(1)
    
def trainModel(dataset, label, classifier: str):
    print("Step 4. Training model using {} algorithm.".format(classifier))
    print("\tTraining process ....")
    if classifier == "kNN":
        # Train model using kNN algorithms
        model = KNeighborsClassifier(n_neighbors=32, weights='uniform', algorithm='auto')
        model.fit(dataset, label)
        print("\tTraining successfully !")
        return model
    if classifier == 'SVM':
        # Train model using SVM algorithm
        pass
    print("The classifier is not exist")
    exit(1)
    
