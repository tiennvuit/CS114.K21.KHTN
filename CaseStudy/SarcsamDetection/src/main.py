from utils import getArgument, loadDataset, featureExtraction, trainModel
from sklearn.model_selection import train_test_split
from config import * # Get hyperparameter
import pickle  # Save model
import datetime


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
    print("Step 3. Split dataset into training data and test data")
    print("\tSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labelset, test_size=TEST_SIZE)
    print("\tSplit dataset successfully !")
    print("\tThe size of training data: {}".format(len(X_train)))
    print("\tThe size of test data: {}".format(len(X_test)))

    # Training model using explicited classifier."
    model = trainModel(dataset=X_train, label=y_train, classifier=classifier)

    # Test model on test data.
    print("Step 5. Test model on test data")
    print("\tTesing model ...")
    accuary = model.score(X_test, y_test)
    print("\tThe accuary of model is {} %".format(accuary*100))

    # Save model
    print("Step 6. Save model into disk")
    print("\tSaving ...")
    #with open('model/{}_{}_{}.pkl'.format(extractor,classifier, datetime.datetime.today().strftime('%d-%m-%Y')), 'wb') as fid:
        #cPickle.dump(gnb, fid)
    with open('model/{}_{}.pkl'.format(extractor,classifier), 'wb') as fid:
        cPickle.dump(gnb, fid)
    print("\tSave model successully with name {}".format(classifier, datetime.datetime.today().strftime('%d-%m-%Y')))


if __name__ == "__main__":
    args = getArgument()
    print_WorkFlow()
    main(args)
