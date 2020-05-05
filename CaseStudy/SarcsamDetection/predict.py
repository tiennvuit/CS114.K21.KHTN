from utils import getArgument,featureExtracting
from PIL import Image

def predict(image, extractor, classifier):
    # Load the model cooresponding with the feature approach and the classifier.
    try:
        with open('model/{}_{}.pkl'.format(extrator, classifier)', 'rb') as fid:
            model = cPickle.load(fid)
            feature_vector = featureExtracting(image=image, method=extrator)
            label = model.predict(image)
            return label
    except:
        print("Unavailable model using {} and {}".format(extrator, classifier))
        exit(1)

def displayWithLabel(image, label):
    pass


def main(args):
    image_path = args.image
    extractor = arg.extractor
    classifier = arg.classifier

    # Load image into memory
    try:
        image = Image.open(path)
    except:
        print("The path of image is not valid !")
        exit(1)

    # Make prediction with the model
    label = predict(image=image, extractor=extractor, classifier=classifier)

    # Display image with label
    print(label)
    #displayWithLabel(image=image, label=label)


if __name__ == '__main__':
    args = getArgument()
    main(args)