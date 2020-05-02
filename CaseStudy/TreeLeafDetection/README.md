# Plant Leaf Classification.

## Dataset information providing by UCI.
- Download dataset at [here](http://archive.ics.uci.edu/ml/datasets/Folio?fbclid=IwAR07TBnKRXAIaCW-YJXSFVlO_nYJArvnQ3Nt7JfdN02WYsRr4CkDtdLa2GQ) and extract it into the same directory with source code.
- The leaves were placed on a white background and then photographed.
- The pictures were taken in broad daylight to ensure optimum light intensity.
- The entire dataset have 

## Flow Working
1. Load dataset and resize every image to the same size (see in the *config.py*) before store them into memory.
2. Using the explicited feature like HOG(Historgram of oriented gradients), SIFT(Scale-invariant feature transform), CNN(Convolutional Neuron Network), ... that you want to extract feacture from each image.
3. Split dataset (feature vectors and labels) into training set and test set with the default ratio is 8/2.
4. Training model using the classifier that you give such as kNN (k Nestest Neighbord), SVM (Suport Vector Machine), Random Forest,... on the training set.
5. Test model on the test set.
6. Save model into memory.

## Run the project
- Clone the project into your local machine
```bash
git clone https://github.com/tiennvuit/CS114.K21.KHTN

cd CaseStudy/TreeLeafDetection
```

- Create new virtual env
```
python -m venv env

source env/bin/activate

pip install --upgarade pip
```

- Install the required dependences.
```bash
pip install -r requirements.txt
```

- Train model.
```bash
python main.py
```

## The result archived
<table style="width:100%">
  <tr>
    <th>Feature</th>
    <th>Classifier</th>
    <th>Training Loss</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <td>HOG</td>
    <td>kNN</td>
    <td></td>
    <td></td>
  </tr>
</table>


## Features
- You can choose the feature approach that you want like HOG, SIFT, CNN features for the classification process.
    + HOG
    + <strike>SIFT</strike>
    + <strike>CNN feature</strike>
- You can choose the classifier (algorithm) that you want to training model like kNN, Random Forest, SVM.
    + kNN
    + <strike>Random Forest</strike>
    + <strike>SVM</strike>



