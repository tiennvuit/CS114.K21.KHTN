# USAGE
# python blur_detection.py --images dataset

# import the necessary packages
from imutils import paths
import argparse
import cv2

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def get_arguments():
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", required=True,
    	help="path to input directory of images")
    parser.add_argument("-t", "--threshold", type=float, default=100.0,
    	help="focus measures that fall below this value will be considered 'blurry'")
    return vars(parser.parse_args())


def detect_blur(image, threshold):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	# if the focus measure is less than the supplied threshold,
	# then the image should be considered "blurry"
	if fm < threshold:
		return (True, fm)
	return (False, fm)

def main(args):
    # loop over the input images
    for imagePath in paths.list_images(args["images"]):
    	# load the image, convert it to grayscale, and compute the
    	# focus measure of the image using the Variance of Laplacian
    	# method
        image = cv2.resize(cv2.imread(imagePath), (300, 400), cv2.INTER_CUBIC)
        (check, fm) = detect_blur(image=image, threshold=args["threshold"])

        text = "Not Blurry"
        color = (0, 255, 0)

    	# if the focus measure is less than the supplied threshold,
    	# then the image should be considered "blurry"
        if check:
            text = "Blurry"
            color = (0, 0, 255)

    	# show the image
        cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Image", image)
        key = cv2.waitKey(0)

if __name__ == '__main__':
    args = get_arguments()
    main(args)
