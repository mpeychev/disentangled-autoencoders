import argparse
import logging
import os
import sys

import numpy as np
from sklearn.svm import SVC
from tensorflow.examples.tutorials.mnist import input_data

import constants
import get_autoencoder
import util

logger = logging.getLogger(__name__)
output_handler = logging.StreamHandler(sys.stderr)
output_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
output_handler.setLevel(logging.INFO)
logger.addHandler(output_handler)
logger.setLevel(logging.INFO)

CLASSIFIER_POSSIBLE_CLASSES_COUNT = 10 # 10 digits to classify

mnist = input_data.read_data_sets(os.path.join(os.path.expanduser('~'), 'MNIST_data'), one_hot=True)

logger.info('Start fetching.')
train_all, test, validation = mnist.train.images, mnist.test.images, mnist.validation.images
train_labels_all, test_labels, validation_labels = mnist.train.labels, mnist.test.labels, \
    mnist.validation.labels
test_labels_digits = np.array([np.argmax(line) for line in test_labels])
logger.info('Fetching completed.')

def train_classifier(autoencoder, seq_index, train, train_labels_digits):
    beta = autoencoder.get_beta()
    logger.info('Beta = {0} | Start training'.format(beta))
    X = autoencoder.get_code(train, ignore_noise=True)
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

def go(architecture, run_index, labels_percentage, train, train_labels_digits):
    beta = 0.0
    while beta - 4.0 < 1e-3:
        seq_index = str(run_index) + '-' + str(labels_percentage) + '-' + str(beta) + '-mnist'
        if architecture == constants.CONV:
            seq_index += '-conv'
        autoencoder = get_autoencoder.mnist(architecture, beta, False, seq_index)
        autoencoder.restore_model()
        train_classifier(autoencoder, seq_index, train, train_labels_digits)
        beta += 0.1

def gather_results(architecture, run_index, labels_percentage):
    training_set_indices_file_name = str(run_index) + '-' + str(labels_percentage) + \
        '-training-indices'
    if architecture == constants.CONV:
        training_set_indices_file_name += '-conv'
    training_set_indices_file_name += '.txt'
    training_set_indices = util.read_list_from_file(
        os.path.join(util.get_logs_dir(), training_set_indices_file_name))

    train = train_all[training_set_indices]
    train_labels = train_labels_all[training_set_indices]
    train_labels_digits = np.array([np.argmax(line) for line in train_labels])

    go(constants.FC if architecture == 'FC' else constants.CONV, run_index, labels_percentage,
        train, train_labels_digits)

def summarise_results(architecture, runs, labels_percentage):
    accuracy = []
    for run_index in range(runs):
        beta = 0.0
        while beta - 4.0 < 1e-3:
            result_file_name = 'results-' + str(run_index) + '-' + str(labels_percentage) + '-' + \
                str(beta) + '-mnist'
            if architecture == constants.CONV:
                result_file_name += '-conv'
            result_file_name += '.txt'
            with open(os.path.join(util.get_results_dir(), result_file_name), 'r') as f:
                first_line = f.readline()
                accuracy.append(float(first_line.rstrip().split()[-1]))
            beta += 0.1
    print accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', dest='architecture', required=True, choices=['FC', 'CONV'],
        help='Autoencoder architecture to test.')
    parser.add_argument('--runs', dest='runs', type=int, required=False, default=1,
        help='Number of experiment runs.')
    parser.add_argument('--summarise', action='store_true',
        help='Summarise the results after writing them in files.')
    args = parser.parse_args()

    for labels_percentage in range(20, 101, 20):
        if args.summarise:
            logger.info('labels percentage = {0}'.format(labels_percentage))
            summarise_results(args.architecture, args.runs, labels_percentage)
        else:
            for run_index in range(args.runs):
                gather_results(args.architecture, run_index, labels_percentage)
