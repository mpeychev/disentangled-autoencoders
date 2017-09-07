import os
import sys
import json
import logging
import util
import draw_util
import numpy as np
import tensorflow as tf
from encoded_image import EncodedImage
from fc_autoencoder import FcAutoencoder
from linear_classifier import LinearClassifier

logger = logging.getLogger(__name__)
output_handler = logging.StreamHandler(sys.stderr)
output_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
output_handler.setLevel(logging.INFO)
logger.addHandler(output_handler)
logger.setLevel(logging.INFO)

results_dir = util.get_results_dir()
util.prepare_dir(results_dir, hard=False)

def vary_code(autoencoder, base_encoded_image, suffix):
    bits = draw_util.encoded_image_to_flattened_bits(base_encoded_image)
    stddev = np.sqrt(autoencoder.get_code_variance([bits])[0])
    print('stddev: {0}'.format(stddev))
    Z = autoencoder.get_code([bits])
    pictures = []
    LOWER_BOUND = -1
    UPPER_BOUND = 1
    SHIFT = 0.5
    for i in range(10):
        shift = LOWER_BOUND
#        while shift <= UPPER_BOUND:
        for j in [2, 1, 0, -1, -2]:
            Z[0][i] += j * SHIFT
            new_output_layer = autoencoder.get_output_layer_from_code(Z)
            pictures.append(new_output_layer[0])
            Z[0][i] -= j * SHIFT
            shift += SHIFT
    print len(pictures)
    np.save(os.path.join(results_dir, 'pictures_' + suffix), np.array(pictures))

def produce_numpy_arrays(beta, seq_index):
    logger.info(seq_index)
    autoencoder = FcAutoencoder(beta=beta, seq_index=seq_index, optimizer=None)
    autoencoder.restore_model()
    logger.info('Beta = {0} | Model restored.'.format(beta))

    vary_code(autoencoder, EncodedImage('ellipse', draw_util.POSITION_LIMIT / 2,
        draw_util.POSITION_LIMIT / 2, draw_util.SCALE_LIMIT / 2, 25), seq_index[-3])
    autoencoder.close_session()

if __name__ == '__main__':
    #produce_numpy_arrays(0.0, 'unique3_0.0')
    #produce_numpy_arrays(1.0, 'unique3_1.0')
    produce_numpy_arrays(4.0, 'unique4_4.0')
