from utils import getArgument, loadDataset, featureExtraction, trainModel
from sklearn.model_selection import train_test_split
from config import *

def print_WorkFlow():
    print("Flow working in this project")
    print("\t - Step 1: Load dataset into memory.")
    print("\t - Step 2: Extract feature vector for each image in the dataset")
    print("\t - Step 3: Split data into training set and test set.")
    print("\t - Step 4: Train model with the explicted classifier.")
    print("\t - Step 5: Testing the model with test set")

def main(args):
    extractor = args.extractor
    classifier = args.classifier
    #print("Use {} is the feature extractor".format(extractor))
    #print("Use {} is the classifier".format(classifier))
    
    # Load dataset into memory
    dataset, labelset = loadDataset()
    # Feature extraction 
    feature_vectors = featureExtraction(dataset=dataset, method=extractor)
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(dataset, labelset, test_size=TEST_SIZE)
    print("Số lượng điểm dữ liệu huấn luyện: {}".format(len(X_train)))
    print("Số lượng điểm dữ liệu kiểm thử: {}".format(len(X_test)))

    # Training model using explicited classifier.
    model = train(dataset=X_train, y_train, classifier=classifier)
    
if __name__ == "__main__":
    args = getArgument()
    print_WorkFlow()
    main(args)

