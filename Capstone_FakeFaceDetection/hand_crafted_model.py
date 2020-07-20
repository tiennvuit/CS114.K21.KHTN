from skimage import feature
import numpy as np
import cv2

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist


# Import classifier classes
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# The measure of classification problem
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Define classifiers

logistic_model = LogisticRegression()
kNN_model = KNeighborsClassifier(n_neighbors=2)
randomForest_model = RandomForestClassifier(max_depth=2, random_state=0)
decisionTree_model = tree.DecisionTreeClassifier()
bayes_model = GaussianNB()
svm_model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
neuronNet_model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                hidden_layer_sizes=(64, 32, 32, 64, 2),
                                random_state=1)


hand_crafted_models = {
    'logistic_regression': logistic_model,
    'knn': kNN_model,
    'random_forest': randomForest_model,
    'decision_tree': decisionTree_model,
    'naive_bayes': bayes_model,
    'svm': svm_model,
    'neural_net': neuronNet_model,
}
