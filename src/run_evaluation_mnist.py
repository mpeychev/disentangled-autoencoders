import os
import sys
import logging
import util
import constants
import get_autoencoder
from tensorflow.examples.tutorials.mnist import input_data
from linear_classifier import LinearClassifier
import numpy as np
import argparse
from sklearn.svm import SVC
import helper
import draw_util

logger = logging.getLogger(__name__)
output_handler = logging.StreamHandler(sys.stderr)
output_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
output_handler.setLevel(logging.INFO)
logger.addHandler(output_handler)
logger.setLevel(logging.INFO)

CLASSIFIER_POSSIBLE_CLASSES_COUNT = 10 # 10 digits to classify
PATIENCE = 10

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

LIMIT = 55000

logger.info('Start fetching.')
train, test, validation = mnist.train.images, mnist.test.images, mnist.validation.images
train_labels, test_labels, validation_labels = mnist.train.labels, mnist.test.labels, \
    mnist.validation.labels
train_labels_digits = np.array([np.argmax(line) for line in train_labels[:LIMIT]])
test_labels_digits = np.array([np.argmax(line) for line in test_labels])
logger.info('Fetching completed.')

def train_classifier(autoencoder, seq_index):
    beta = autoencoder.get_beta()
    logger.info('Beta = {0} | Start training'.format(beta))
    X = autoencoder.get_code(train[:LIMIT], ignore_noise=True)
    classifier = SVC(kernel='rbf')
    classifier.fit(X, train_labels_digits)
    logger.info('Beta = {0} | Training completed'.format(beta))
    X_ = autoencoder.get_code(test, ignore_noise=True)
    predicted = classifier.predict(X_)

    accuracy = 0.0
    for i in range(len(predicted)):
        accuracy += (test_labels_digits[i] == predicted[i])
    accuracy /= len(predicted)
    logger.info('Beta = {0} | Classifier accuracy: {1}'.format(beta, accuracy))
    results_dir = util.get_results_dir()
    accuracy_output_file = os.path.join(results_dir, 'results-' + seq_index + '.txt')
    with open(accuracy_output_file, 'w') as f:
        f.write('Beta = {0} | Classifier accuracy: {1}\n'.format(beta, accuracy))
        f.write('Beta = {0} | Reconstruction: {1}\n'.format(
            beta, autoencoder.calc_reconstruction_accuracy(test)))
    autoencoder.close_session()

def go(architecture, index):
    beta = 0.0
    while beta - 4.0 < 1e-3:
        seq_index = str(index) + '-' + str(beta) + '-mnist'
        if architecture == constants.CONV:
            seq_index += '-conv'
        autoencoder = get_autoencoder.mnist(architecture, beta, False, seq_index)
        autoencoder.restore_model()
        train_classifier(autoencoder, seq_index)
        beta += 0.1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', dest='architecture', required=True,
        help='Autoencoder architecture to test: FC or CONV.')
    args = parser.parse_args()

    for index in range(5):
        go(constants.FC if args.architecture == 'FC' else constants.CONV, index)
