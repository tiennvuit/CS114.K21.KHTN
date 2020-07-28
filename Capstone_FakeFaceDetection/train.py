# USAGE
# python train.py --dataset dataset/Collection_dataset/ --model deeeplearning

# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
import datetime
from utils import get_arguments, load_datasetDeep, load_datasetLBPs, plot_progress, load_extracted_feature
from config import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def main(args):

    # Load dataset to memory follow the selected approach
    if args['model'] == 'deeplearning':
        
        TRAIN_PATH = os.path.join(args["dataset"], "train")
        TEST_PATH = os.path.join(args["dataset"], "test")
        
        from classifier.livenessnet import LivenessNet
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.optimizers import Adam
        
        
        data, labels, le = load_datasetDeep(TRAIN_PATH)
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
        model = LivenessNet.build(width=64, height=64, depth=3,
                                  classes=len(le.classes_))
        model.compile(loss="binary_crossentropy", 
                      optimizer=opt, 
                      metrics=["accuracy", ])
        
		# train the network
        print("[INFO] training network for {} epochs...".format(EPOCHS))
        H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
                      validation_data=(testX, testY), 
                      steps_per_epoch=len(trainX) // BS,
                      epochs=EPOCHS)

		# evaluate the network
        print("[INFO] Evaluating network on the unseen dataset ...")
        test_data, test_labels, le = load_datasetDeep(TEST_PATH)

        predictions = model.predict(x=test_data, batch_size=BS)
        print(classification_report(test_labels.argmax(axis=1),
			predictions.argmax(axis=1), target_names=le.classes_))

		# save the network to disk
        day = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        saving_path = "saved_model/deeplearning_model/model_" + args['dataset'].split(os.path.sep)[1] + '_' + day
        print("[INFO] serializing network to '{}'...".format(saving_path))
        model.save(saving_path, save_format="h5")
        
        name_plot = args['dataset'].split(os.path.sep)[1] + '_' + day
        plot_progress(model=H, name=name_plot)

    else:
        print("[INFO] Loading data ...")
        data, labels = load_extracted_feature(path=args['dataset'])
        print("Size of dataset is {}".format(len(data)))
        (trainX, testX, trainY, testY) = train_test_split(data, labels,
										test_size=TEST_SIZE, random_state=42)
		# Get model
        from hand_crafted_model import hand_crafted_models
        model = hand_crafted_models[args['model']]

		# Training model
        print("[INFO] Training {} model...".format(args['model']))
        model.fit(trainX, trainY)
        
        # Evaluate the model
        print("[INFO] Evaluating network on the test data ...")
        print("\t- The accuracy of model on training set: {}".format(model.score(trainX, trainY)))
        y_pred = model.predict(testX)
        print("\t- The accuary of model on test set: {}".format(accuracy_score(y_true=y_true, y_pred=y_pred)))
        print("\t- The confusion matrix of model on test set: \n{}".format(classification_report(y_true=y_true, y_pred=y_pred)))

		# Save the model to disk
        print("[INFO] Saving model to disk ...")
        filename = 'saved_model/hand-crafted_model/' + args['model'] + "_model_{}_".format(args['path'].split("/")[-1].split(".")[0]) + args['dataset'].split(os.path.sep)[1] + '_' + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
        pickle.dump(model, open(filename, 'wb'))


if __name__ == "__main__":
	args = get_arguments()
	main(args)
