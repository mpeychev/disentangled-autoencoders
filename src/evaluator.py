import os
import sys
import json
import logging
import util
import draw_util
import numpy as np
import tensorflow as tf
from encoded_image import EncodedImage
from linear_classifier import LinearClassifier
import constants
import math

logger = logging.getLogger(__name__)
output_handler = logging.StreamHandler(sys.stderr)
output_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
output_handler.setLevel(logging.INFO)
logger.addHandler(output_handler)
logger.setLevel(logging.INFO)

# Comparator epsilon.
EPSILON = 1e-6

# Early stopping constant.
PATIENCE = 20

# Training with mini-batches.
BATCH_SIZE = 100

# Classifier constant.
CLASSIFIER_POSSIBLE_CLASSES_COUNT = 4

class Evaluator(object):

    def __init__(self,
        autoencoder_test_data,
        classifier_train_json,
        classifier_test_json,
        classifier_validation_json,
        index=None):

        self.autoencoder_test_data = autoencoder_test_data
        self.classifier_train_json = classifier_train_json
        self.classifier_test_json = classifier_test_json
        self.classifier_validation_json = classifier_validation_json
        
        self.compressed_set = set()
        for img in autoencoder_test_data:
            self.compressed_set.add(draw_util.compress_bits(img))
        self.classifier_accuracy_all = []
        self.cost_all = []
        self.reconstruction_loss_all = []
        self.kl_divergence_all = []
        self.reconstruction_accuracy_all = []

        results_dir = util.get_results_dir()
        util.prepare_dir(results_dir, hard=False)

        self.index = constants.run_index() if index is None else index
        self.classifier_summary_file = os.path.join(results_dir,
            'classifier_accuracy_summary_' + str(self.index) + '.txt')

    def evaluate(self, autoencoder, seq_index):
        results_dir = util.get_results_dir()
        self.stats_output_file = os.path.join(results_dir, 'stats-' + seq_index + '.txt')
        self.accuracy_output_file = os.path.join(results_dir, 'accuracy-' + seq_index + '.txt')

        autoencoder.restore_model()
        logger.info('Model restored.')

        self.cost_all.append(autoencoder.calc_cost(self.autoencoder_test_data))
        self.reconstruction_loss_all.append(
            autoencoder.calc_reconstruction_loss(self.autoencoder_test_data))
        self.kl_divergence_all.append(autoencoder.calc_kl_divergence(self.autoencoder_test_data))
        self.reconstruction_accuracy_all.append(
            autoencoder.calc_reconstruction_accuracy(self.autoencoder_test_data))

        if autoencoder.get_beta() < EPSILON or \
            math.fabs(autoencoder.get_beta() - 1.00) < EPSILON or \
            math.fabs(autoencoder.get_beta() - 4.00) < EPSILON:
            with open(self.stats_output_file, 'w') as f:
                f.write('Evaluate on test data. Cost: {0}\n'.format(self.cost_all[-1]))
                f.write('Reconstruction loss: {0}\n'.format(self.reconstruction_loss_all[-1]))
                f.write('KL-Divergence: {0}\n'.format(self.kl_divergence_all[-1]))
                f.write('Reconstruction accuracy: {0}\n'.format(
                    self.reconstruction_accuracy_all[-1]))
                f.write('\n')

                self._build_stats(autoencoder, f)
                logger.info('Dictionary built.')
                for i in range(10):
                    self._analyse_means(seq_index, i, f)

        self._measure_disentanglement(autoencoder)

    def _build_stats(self, autoencoder, output_file):
        logger.info('Start collecting means and variance statistics.')
        self.learnt_latent_means = {}
        cnt = 0
        learnt_variance = np.zeros(10, dtype=np.float32)
        for shape in ['ellipse', 'square', 'triangle']:
            _cnt = 0
            _learnt_variance = np.zeros(10, dtype=np.float32)
            for position_x in range(constants.POSITION_LIMIT):
                for position_y in range(constants.POSITION_LIMIT):
                    for radius in range(constants.SCALE_LIMIT):
                        for rotation in range(constants.ROTATION_LIMIT):
                            bits = draw_util.encoded_image_to_flattened_bits(
                                EncodedImage(shape, position_x, position_y, radius, rotation))
                            if draw_util.compress_bits(bits) not in self.compressed_set:
                                continue
                            shape_index = 0 if shape == 'ellipse' else 1 if shape == 'square' else 2

                            code_mean = autoencoder.get_code_mean(np.array([bits]))[0]
                            code_variance = autoencoder.get_code_variance(np.array([bits]))[0]

                            self.learnt_latent_means[(shape_index, position_x, position_y, radius,
                                rotation)] = code_mean

                            learnt_variance += code_variance
                            _learnt_variance += code_variance
                            cnt += 1
                            _cnt += 1
                            if cnt % 10000 == 0:
                                logger.info('Beta = {0} | {1}'.format(autoencoder.get_beta(), cnt))
            _learnt_variance /= _cnt
            output_file.write('{0} variance:\n'.format(shape))
            output_file.write(str(_learnt_variance))
            output_file.write('\n')

        learnt_variance /= cnt
        output_file.write('\nVariance all.\n')
        output_file.write(str(learnt_variance) + '\n')

    def _analyse_means(self, seq_index, index, output_file):
        logger.info('Analyse means {0}, index {1}'.format(seq_index, index))
        output_file.write('Analyse means, index {0}\n'.format(index))
        pos = np.zeros((constants.POSITION_LIMIT, constants.POSITION_LIMIT), dtype=np.float32)
        pos_cnt = np.zeros((constants.POSITION_LIMIT, constants.POSITION_LIMIT), dtype=np.int32)

        scale = np.zeros((3, constants.SCALE_LIMIT), dtype=np.float32)
        scale_cnt = np.zeros((3, constants.SCALE_LIMIT), dtype=np.int32)

        rot = np.zeros((3, constants.ROTATION_LIMIT), dtype=np.float32)
        rot_cnt = np.zeros((3, constants.ROTATION_LIMIT), dtype=np.int32)

        for key, z_mean in self.learnt_latent_means.iteritems():
            shape, position_x, position_y, radius, rotation = key
            z_index = z_mean[index]
            pos[position_x][position_y] += z_index
            pos_cnt[position_x][position_y] += 1

            scale[shape][radius] += z_index
            scale_cnt[shape][radius] += 1

            rot[shape][rotation] += z_index
            rot_cnt[shape][rotation] += 1

        pos /= pos_cnt
        scale /= scale_cnt
        rot /= rot_cnt

        output_file.write('\nPosition:\n')
        output_file.write(str(pos) + '\n')

        output_file.write('\nScale:\n')
        output_file.write(str(scale) + '\n')

        output_file.write('\nRotation:\n')
        output_file.write(str(rot) + '\n')
        output_file.write('\n\n')

    def _measure_disentanglement(self, autoencoder):
        beta = autoencoder.get_beta()
        train_classifier_inputs, train_classifier_labels = \
            Evaluator.inputs_and_labels(autoencoder, self.classifier_train_json)
        logger.info('Beta = {0} | Classifier training data processed.'.format(beta))
        test_classifier_inputs, test_classifier_labels = \
            Evaluator.inputs_and_labels(autoencoder, self.classifier_test_json)
        logger.info('Beta = {0} | Classifier test data processed.'.format(beta))
        valid_classifier_inputs, valid_classifier_labels = \
            Evaluator.inputs_and_labels(autoencoder, self.classifier_validation_json)
        logger.info('Beta = {0} | Classifier validation data processed.'.format(beta))

        autoencoder.close_session()

        # Constants.
        CLASSIFIER_INPUT_DIMENSION = autoencoder.get_code_dimension()

        logger.info('Beta = {0} | Create classifier.'.format(beta))
        classifier = LinearClassifier(CLASSIFIER_INPUT_DIMENSION, CLASSIFIER_POSSIBLE_CLASSES_COUNT)

        epoch = 0
        best_validation_score = np.inf
        consecutive_decreases = 0
        LEN_TRAIN = len(train_classifier_inputs)
        while True:
            # Train with mini-batches.
            random_permutation = np.random.permutation(np.arange(LEN_TRAIN))
            shuffled_train_inputs = train_classifier_inputs[random_permutation]
            shuffled_train_labels = train_classifier_labels[random_permutation]
            classifier.partial_fit(shuffled_train_inputs, shuffled_train_labels)

#            for start in range(0, LEN_TRAIN, BATCH_SIZE):
#                end = LEN_TRAIN if LEN_TRAIN - start < 2 * BATCH_SIZE else start + BATCH_SIZE
#                classifier.partial_fit(shuffled_train_inputs[start:end],
#                    shuffled_train_labels[start:end])
#                if end == LEN_TRAIN:
#                    break

            # Early stopping.
            validation_score = classifier.get_cost(valid_classifier_inputs, valid_classifier_labels)
            logger.info('Beta = {0} | Classifier epoch {1} validation cost: {2}'.format(
                beta, epoch, validation_score))
            if best_validation_score > validation_score:
                logger.info('Beta = {0} | *** OPTIMAL SO FAR ***'.format(beta))
                best_validation_score = validation_score
                consecutive_decreases = 0
            else:
                consecutive_decreases += 1
                if consecutive_decreases > PATIENCE:
                    break

            epoch += 1

        logger.info('Beta = {0} | Classifier training completed.'.format(beta))

        accuracy = classifier.accuracy(test_classifier_inputs, test_classifier_labels)
        logger.info('Beta = {0} | Classifier accuracy: {1}'.format(beta, accuracy))
        with open(self.accuracy_output_file, 'w') as f:
            f.write('Beta = {0} | Classifier accuracy: {1}\n'.format(beta, accuracy))
        self.classifier_accuracy_all.append((beta, accuracy))
        classifier.close_session()

    @staticmethod
    def inputs_and_labels(autoencoder, dataset_json):
        inputs = []
        labels = []
        for sample in dataset_json:
            encoded_image_start = EncodedImage.from_dict(sample['s'])
            encoded_image_end = EncodedImage.from_dict(sample['e'])
            z_mean_start = autoencoder.get_code_mean(
                [draw_util.encoded_image_to_flattened_bits(encoded_image_start)])
            z_mean_end = autoencoder.get_code_mean(
                [draw_util.encoded_image_to_flattened_bits(encoded_image_end)])
            z_mean_diff = np.absolute(z_mean_start[0] - z_mean_end[0])
            if np.amax(z_mean_diff) < 1e-3:
                logger.error("Code collision")
            inputs.append(z_mean_diff / np.amax(z_mean_diff))

            change_factor = sample['f']
            one_hot = np.zeros(CLASSIFIER_POSSIBLE_CLASSES_COUNT)
            one_hot[change_factor] = 1.0
            labels.append(one_hot)
        return np.array(inputs), np.array(labels)

    def results_summary(self):
        with open(self.classifier_summary_file, 'w') as f:
            f.write('\nBetas\n')
            for beta, _ in self.classifier_accuracy_all:
                f.write(str(beta) + ' ')
            f.write('\nAccuracy\n')
            for _, accuracy in self.classifier_accuracy_all:
                f.write(str(accuracy) + ' ')
            f.write('\nLoss function as defined\n')
            f.write(str(self.cost_all))
            f.write('\nLoss function as defined - reconstruction component\n')
            f.write(str(self.reconstruction_loss_all))
            f.write('\nLoss function as defined - kl divergence component\n')
            f.write(str(self.kl_divergence_all))
            f.write('\nReconstruction accuracy\n')
            f.write(str(self.reconstruction_accuracy_all))
