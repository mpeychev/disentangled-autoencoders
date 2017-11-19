import getpass
import json
import logging
import os
import re
import shutil
import sys

import numpy as np
from PIL import Image
import tensorflow as tf

import constants

logger = logging.getLogger(__name__)
output_handler = logging.StreamHandler(sys.stderr)
output_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
output_handler.setLevel(logging.INFO)
logger.addHandler(output_handler)
logger.setLevel(logging.INFO)

# Util functions to manage directories and files.

def get_parent_dir():
    """Returns the root directory of the project."""
    return os.path.abspath(os.path.join(os.getcwd(), os.pardir))

def get_logs_dir():
    """Returns the root directory of the tensorboard log files."""
    logs_dir = os.path.join(get_parent_dir(), 'logs')
    prepare_dir(logs_dir, hard=False)
    return logs_dir

def get_results_dir():
    """Returns the directory for the results file."""
    results_dir = os.path.join(get_parent_dir(), 'results')
    prepare_dir(results_dir, hard=False)
    return results_dir

def get_data_dir():
    """Returns the diretory of the data files."""
    if getpass.getuser() == 'mp739':
        return '/local/scratch/mp739/data'
    else:
        return os.path.join(get_parent_dir(), 'data')

def get_autoencoder_data_dir():
    """Returns the directory of the data to be used for training the autoencoder."""
    return os.path.join(get_data_dir(), 'autoencoder')

def get_classifier_data_dir():
    """Returns the directory of the data to be used for training the linear classifier which
    evaluates the disentanglement level.
    """
    return os.path.join(get_data_dir(), 'classifier')

def prepare_dir(path_to_folder, hard=True):
    """Makes an empty path_to_folder folder."""
    if hard and os.path.exists(path_to_folder):
        shutil.rmtree(path_to_folder)
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)

def write_list_to_file(li, file_name):
    with open(file_name, 'w') as f:
        for item in li:
            print >> f, item

def read_list_from_file(file_name):
    li = []
    with open(file_name, 'r') as f:
        for line in f:
            li.append(int(line.split()[0]))
    return li

def get_autoencoder_data(index=None, get_unique=True):
    """Loads the autoencoder clean data.

    By default the index in run_counter.txt is used - useful for training.
    """
    suffix = ('' if not get_unique else 'unique-') + str(constants.run_index() if index is None \
        else index)
    data_dir = get_autoencoder_data_dir()
    return np.load(os.path.join(data_dir, 'train-' + suffix + '.npy')), \
        np.load(os.path.join(data_dir, 'test-' + suffix + '.npy')), \
        np.load(os.path.join(data_dir, 'validation-' + suffix + '.npy'))

def get_classifier_data(index=None):
    """Loads the classifier data."""
    suffix = str(constants.run_index()) if index is None else str(index)
    data_dir = get_classifier_data_dir()
    with open(os.path.join(data_dir, 'train-' + suffix + '.txt')) as data_file:
        train_data = json.load(data_file)
    with open(os.path.join(data_dir, 'test-' + suffix + '.txt')) as data_file:
        test_data = json.load(data_file)
    with open(os.path.join(data_dir, 'validation-' + suffix + '.txt')) as data_file:
        validation_data = json.load(data_file)
    return train_data, test_data, validation_data

# Tensorflow util functions to use while building tensorflow compute graph.

def fans(shape):
    """Returns fan_in and fan_out according to the shape.

    Assumes the len of shape is 2 or 4.
    """
    assert(len(shape) == 2 or len(shape) == 4)
    if len(shape) == 2:
        return shape[0], shape[1]
    else:
        S = shape[0] * shape[1]
        return S * shape[2], S * shape[3]

def xavier_init(shape):
    """Xavier initialization of network weights.

    Designed to keep the gradients across all layers approximately the same.
    Xavier Glorot and Yoshua Bengio (AISTATS, 2010):
        Understanding the difficulty of training deep feedforward neural networks.
    """
    fan_in, fan_out = fans(shape)
    low  = -np.sqrt(6.0 / (fan_in + fan_out))
    high =  np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform(shape, minval=low, maxval=high)

def get_weights_xavier(shape):
    """Get Xavier initialized weights with the desired shape."""
    return tf.Variable(xavier_init(shape), name='weights')

def he_init(shape):
    """He initialization of network weights for ReLU activations.

    Kaiming He et al.
        Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.
    """
    fan_in, _ = fans(shape)
    stddev = np.sqrt(2.0 / fan_in)
    return tf.random_normal(shape, stddev=stddev)

def get_weights_he(shape):
    """Get He initialized weights with the desired shape."""
    return tf.Variable(he_init(shape), name='weights')

def get_bias(dimension):
    """Get zero initialized bias with the desired dimension."""
    return tf.Variable(tf.zeros([dimension]), name='bias')

# Batch normalization constants.
EPSILON = 1e-4
DECAY = 0.999

def batch_norm(inputs, is_training):
    """Batch normalisation.

    Performs batch normalisation of the layer taking into account whether we are in a training phase
    or not.
    """
    scale_gamma = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    shift_beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    mean_approx = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    var_approx = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        axes = [0, 1, 2] if len(inputs.get_shape().as_list()) == 4 else [0]
        batch_mean, batch_var = tf.nn.moments(inputs, axes)
        train_mean = tf.assign(mean_approx, mean_approx * DECAY + batch_mean * (1 - DECAY))
        train_var = tf.assign(var_approx, var_approx * DECAY + batch_var * (1 - DECAY))

        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, shift_beta,
                scale_gamma, EPSILON)
    else:
        return tf.nn.batch_normalization(inputs, mean_approx, var_approx, shift_beta,
            scale_gamma, EPSILON)
