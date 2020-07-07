import argparse
import os
import numpy as np
import json
from config import *

def getArgument():
    parser = argparse.ArgumentParser(description='Sarcasm detetion')
    parser.add_argument('-extractor', help='The feature extractor', default="BOW")
    parser.add_argument('-classifier', help="The classifier use", default="kNN")
    parser.add_argument('-image_path', help="The test image", default="test0.png")
    return parser.parse_args()

# Load dataset into program
def loadDataset():
    print("Step 1. Loading dataset")
    print("\tLoading dataset ...")
    dataset = []
    labelset = []
    for l in open(DATASET_DIR,'r'):
        data = json.loads(l)
        dataset.append(data['headline'])
        labelset.append(data['is_sarcastic'])

    print("\tLoad dataset successfully !")
    print("\tThe size of dataset is {}".format(len(labelset)))
    return (dataset, labelset)

def featureExtraction(dataset, method):
    print("Step 2. Feature extraction")
    print("\tExtracting feature vectors from dataset....")
    feature_vectors = []
    for headline in dataset:
        feature_vector = featureExtracting(sentense=headline, method=method)
        feature_vectors.append(feature_vector)
    print("\tFeature extraction successfully !")
    return feature_vectors

def featureExtracting(sentense, method: str):
    # Use the extracting feature method like HOG, SIFT, Deep Fetures
    if method == "BOW":
        return None

    print("Feature extractor is not exist !")
    exit(1)

def trainModel(dataset, label, classifier: str):
    print("Step 4. Training model using {} algorithm.".format(classifier))
    print("\tTraining process ....")
    if classifier == "kNN":
        # Train model using kNN algorithms

        model.fit(dataset, label)
        print("\tTraining successfully !")
        return model
    if classifier == 'SVM':
        # Train model using SVM algorithm
        pass
    print("The classifier is not exist")
    exit(1)
