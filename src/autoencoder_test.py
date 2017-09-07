#!/usr/local/bin/python

import os
import sys
import logging
import helper
import random
import argparse
from termcolor import colored
import tensorflow as tf
import numpy as np
import get_autoencoder
import constants
from tensorflow.examples.tutorials.mnist import input_data
import run_experiments

logger = logging.getLogger(__name__)
output_handler = logging.StreamHandler(sys.stderr)
output_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
output_handler.setLevel(logging.INFO)
logger.addHandler(output_handler)
logger.setLevel(logging.INFO)

def show_results(architecture, beta, test):
    autoencoder = get_autoencoder.mnist(architecture, beta, False)
    autoencoder.restore_model()
    print('Reconstruction loss: {0}'.format(autoencoder.calc_reconstruction_loss(test)))
    helper.show_images_general_purpose(test[:9], str(beta) + '-orig-test')
    helper.show_images_general_purpose(autoencoder.get_output_layer(test[:9]), str(beta) + '-recon')
    for _ in range(5):
        idx = random.randint(0, len(test))
        helper.show_images_general_purpose(test[idx:idx+1], 'orig-' + str(idx))
        helper.show_images_general_purpose(autoencoder.get_output_layer(test[idx:idx+1]),
            'recon-' + str(idx))
    autoencoder.close_session()

def test_driver(architecture, beta):
    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

    logger.info('Start fetching.')
    train, test, valid = mnist.train.images, mnist.test.images, mnist.validation.images
    logger.info('Fetching completed.')
    helper.show_images_general_purpose(valid[:100], 'orig')

    autoencoder = get_autoencoder.mnist(architecture, beta, True)
    logger.info('Autoencoder built. Start training.')

    if not run_experiments.train_and_log_autoencoder(autoencoder, train, valid, test_mode=True):
        print colored('Training failed.', 'red')
        return
    show_results(architecture, beta, test)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', dest='architecture', required=True,
        help='Autoencoder architecture to test: FC or CONV.')
    parser.add_argument('--beta', dest='beta', required=False, default=0,
        help='Beta.')
    args = parser.parse_args()
    test_driver(constants.FC if args.architecture == 'FC' else constants.CONV,
        beta=float(args.beta))

if __name__ == '__main__':
    main()
