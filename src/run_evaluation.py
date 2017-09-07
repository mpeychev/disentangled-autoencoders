import os
import sys
import logging
import util
import constants
import generate_data
import get_autoencoder
from evaluator import Evaluator

logger = logging.getLogger(__name__)
output_handler = logging.StreamHandler(sys.stderr)
output_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
output_handler.setLevel(logging.INFO)
logger.addHandler(output_handler)
logger.setLevel(logging.INFO)

def go(index):
    logger.info('Start evaluation for index = {0}'.format(index))
    denoising = (index in range(7, 12) or index in range(17, 22))
    architecture = constants.FC if index <= 11 else constants.CONV

    generate_data.gen_classifier_data(index=index)

    _, test, _ = util.get_autoencoder_data(index=index)
    logger.info('Index={0} | Test data is read.'.format(index))

    train_classifier_json, test_classifier_json, validation_classifier_json = \
        util.get_classifier_data(index=index)
    logger.info('Index = {0} | Classifier JSON data read.'.format(index))

    evaluator = Evaluator(test,
        train_classifier_json, test_classifier_json, validation_classifier_json, index=index)

    beta = constants.BETA_LOW
    while beta - constants.BETA_HIGH < 1e-6:
        logger.info('Index = {0} | Beta = {1} | Start evaluation'.format(index, beta))
        seq_index = str(index) + '-' + str(beta)
        if denoising:
            seq_index += '-denoising'
        if architecture == constants.CONV:
            seq_index += '-conv'
        autoencoder = get_autoencoder.shapes_set(architecture, beta, None, seq_index, denoising)
        evaluator.evaluate(autoencoder, seq_index)
        beta += constants.BETA_STEP
    evaluator.results_summary()

if __name__ == '__main__':
    for index in range(2, 22):
        go(index)
