import os
import sys
import logging
import numpy as np
import tensorflow as tf
import util
import constants
import generate_data
import helper
import argparse
import get_autoencoder
import random
from tensorflow.examples.tutorials.mnist import input_data

# Early stopping constant.
PATIENCE = 5

# Training with mini-batches.
BATCH_SIZE = 1000

# Epoch limit in test mode.
EPOCH_LIMIT = 5

MAX_EPOCH = 0

logger = logging.getLogger(__name__)
output_handler = logging.StreamHandler(sys.stderr)
output_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
output_handler.setLevel(logging.INFO)
logger.addHandler(output_handler)
logger.setLevel(logging.INFO)

def train_and_log_autoencoder(autoencoder, train, validation, test_mode=False):
    global MAX_EPOCH

    tf.set_random_seed(random.randint(0,10000))
    epoch = 0
    best_validation_score = np.inf
    consecutive_decreases = 0
    LEN_TRAIN = len(train)
    while True:
        # Train with mini-batches.
        np.random.shuffle(train)
#        autoencoder.partial_fit(train)

        for start in range(0, LEN_TRAIN, BATCH_SIZE):
            if LEN_TRAIN - start < 2 * BATCH_SIZE:
                autoencoder.partial_fit(train[start:LEN_TRAIN])
                break
            else:
                autoencoder.partial_fit(train[start:start + BATCH_SIZE])

        if test_mode:
            helper.show_images_general_purpose(autoencoder.get_output_layer(validation[:100]),
                'recon-epoch-' + str(epoch))

        # Early stopping.
        validation_score = autoencoder.calc_cost(validation)
        MAX_EPOCH = max(MAX_EPOCH, epoch)
        logger.info('Beta={0} | Epoch {1} validation cost: {2} | max epoch = {3}'.format(
            autoencoder.get_beta(), epoch, validation_score, MAX_EPOCH))
        if np.isnan(validation_score):
            logger.error('Got NaN')
            autoencoder.close_session()
            logger.error('Beta = {0} | Training failed.'.format(autoencoder.get_beta()))
            return False
        if best_validation_score > validation_score:
            logger.info('*** OPTIMAL SO FAR ***')
            autoencoder.save_model()
            best_validation_score = validation_score
            consecutive_decreases = 0
        else:
            consecutive_decreases += 1
            if consecutive_decreases > PATIENCE:
                break

        epoch += 1

        if test_mode and epoch > EPOCH_LIMIT:
            break

    autoencoder.close_session()
    logger.info('Beta = {0} | Training completed.'.format(autoencoder.get_beta()))
    return True

def go_shapes(denoising, architecture, reduced=False):
    # Generate data
    generate_data.gen_autoencoder_data(reduced=reduced)

    # Fetch training data
    logger.info('Start fetching.')
    train, _, validation = util.get_autoencoder_data()
    if len(validation) > 40000:
        validation = validation[:40000]
    logger.info('Train and validation data are read.')

    logger.info('Train set: {0}'.format(len(train)))
    logger.info('Validation set: {0}'.format(len(validation)))

    # Train and log all autoencoders
    current_beta = constants.BETA_LOW
    flag = True
    while True:
        if current_beta - constants.BETA_HIGH > 1e-6:
            break

        # Experiment index
        seq_index = str(constants.run_index()) + '-' + str(current_beta)
        if denoising:
            seq_index += '-denoising'
        if architecture == constants.CONV:
            seq_index += '-conv'

        logger.info('Start building the variational autoencoder architecture.')
        logger.info('Beta = {0}, Seq index = {1}'.format(current_beta, seq_index))

        lr = 0.00003 if current_beta < 1e-3 else 0.001
        autoencoder = get_autoencoder.shapes_set(architecture, current_beta,
            lr, seq_index, denoising)

        is_training_successful = train_and_log_autoencoder(autoencoder, train, validation)
        if not is_training_successful:
            flag = False
            break
        current_beta += constants.BETA_STEP
    if flag:
        constants.increase_index()
    return flag

def go_mnist(architecture, run_index=None):
    mnist = input_data.read_data_sets(os.path.join(os.path.expanduser('~'), 'MNIST_data'),
        one_hot=True)

    logger.info('Start fetching.')
    train_all, _, validation = mnist.train.images, mnist.test.images, mnist.validation.images
    class_indices = [[] for _ in range(10)]
    class_counts = [0] * 10
    for index, label in enumerate(mnist.train.labels):
        digit = np.argmax(label)
        class_indices[digit].append(index)
        class_counts[digit] += 1
    logger.info('Fetching completed.')

    for labels_percentage in range(20, 101, 20):
        training_set_indices = []
        for digit in range(10):
            training_set_indices += random.sample(class_indices[digit],
                (class_counts[digit] * labels_percentage) / 100)
        train = train_all[training_set_indices]
	training_set_indices_file_name = str(run_index) + '-' + str(labels_percentage) + \
	    '-training-indices'
	if architecture == constants.CONV:
	    training_set_indices_file_name += '-conv'
	training_set_indices_file_name += '.txt'
        util.write_list_to_file(training_set_indices,
            os.path.join(util.get_logs_dir(), training_set_indices_file_name))
        flag = True
        beta = 0.0
        while beta - 4.0 < 1e-3:
            seq_index = str(constants.run_index() if run_index is None else run_index) + \
                '-' + str(labels_percentage) + '-' + str(beta) + '-mnist'
            if architecture == constants.CONV:
                seq_index += '-conv'

            logger.info('Start building the variational autoencoder architecture.')
            logger.info('Beta = {0}, Seq index = {1}'.format(beta, seq_index))

            autoencoder = get_autoencoder.mnist(architecture, beta, True, seq_index=seq_index)
            is_training_successful = train_and_log_autoencoder(autoencoder, train, validation)
            if not is_training_successful:
                flag = False
                break
            beta += 0.1
        if flag and run_index is None:
            constants.increase_index()

    return flag

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', required=True, choices=['SHAPES', 'MNIST'],
        help='Run experiments on ShapesSet or MNIST.')
    parser.add_argument('--arch', dest='architecture', required=True, choices=['FC', 'CONV'],
        help='Autoencoder architecture to test.')
    parser.add_argument('--denoising', action='store_true',
        help='Denoising instead of simple.')
    parser.add_argument('--runs', dest='runs', type=int, required=False, default=1,
        help='Number of experiment runs.')
    args = parser.parse_args()

    if args.dataset == 'SHAPES':
        tf.reset_default_graph()
        for i in range(args.runs):
            go_shapes(denoising=args.denoising,
                architecture=constants.FC if args.architecture == 'FC' else constants.CONV)
    else:
        tf.reset_default_graph()
        for i in range(args.runs):
            go_mnist(architecture=constants.FC if args.architecture == 'FC' else constants.CONV,
                run_index=i)
