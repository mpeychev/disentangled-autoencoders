import os
import sys
import argparse
import logging
import random
import json
import collections
import util
import draw_util
import numpy as np
from encoded_image import EncodedImage
import constants

logger = logging.getLogger(__name__)
output_handler = logging.StreamHandler(sys.stderr)
output_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
output_handler.setLevel(logging.INFO)
logger.addHandler(output_handler)
logger.setLevel(logging.INFO)

TRAIN_RATIO = 70.0
TEST_RATIO = 15.0
VALIDATION_RATIO = 15.0

# Autoencoder data generation.

def statistics_single(dataset):
    logger.info('Length: {0}'.format(len(dataset)))
    logger.info('Shapes: {0}'.format(
        collections.Counter(map(lambda encoded_image: encoded_image.shape, dataset))))
    logger.info('PositionX: {0}'.format(
        collections.Counter(map(lambda encoded_image: encoded_image.position_x, dataset))))
    logger.info('PositionY: {0}'.format(
        collections.Counter(map(lambda encoded_image: encoded_image.position_y, dataset))))
    logger.info('Radius: {0}'.format(
        collections.Counter(map(lambda encoded_image: encoded_image.radius, dataset))))
    logger.info('Rotation: {0}'.format(
        collections.Counter(map(lambda encoded_image: encoded_image.rotation, dataset))))
    print('\n')

def statistics(train, test, validation):
    logger.info('Train stats.')
    statistics_single(train)
    logger.info('Test stats.')
    statistics_single(test)
    logger.info('Validation stats.')
    statistics_single(validation)

def split_for_shape(shape, tot_limit, gen_unique, reduced):
    logger.info('Split for shape {0}'.format(shape))
    all_images = []
    uniques = set()
    for position_x in range(constants.POSITION_LIMIT / (2 ** reduced)):
        for position_y in range(constants.POSITION_LIMIT / (2 ** reduced)):
            for radius in range(constants.SCALE_LIMIT):
                for rotation in range(constants.ROTATION_LIMIT / (2 ** reduced)):
                    enc_img = EncodedImage(shape, position_x, position_y, radius, rotation)
                    if not gen_unique:
                        all_images.append(enc_img)
                    else:
                        img = draw_util.encoded_image_to_flattened_bits(enc_img)
                        t = draw_util.compress_bits(img)
                        if t not in uniques:
                            uniques.add(t)
                            all_images.append(enc_img)
    random.shuffle(all_images)
    if tot_limit != -1:
        all_images = all_images[:min(tot_limit, len(all_images))]
    split_index = int(round(len(all_images) * TRAIN_RATIO / 100.0))
    train_images = all_images[:split_index]
    rest = all_images[split_index:]
    random.shuffle(rest)
    split_index = int(round(len(rest) * TEST_RATIO / (TEST_RATIO + VALIDATION_RATIO)))
    return train_images, rest[:split_index], rest[split_index:]

def gen_autoencoder_data(gen_unique=True, reduced=False):
    target_dir = util.get_autoencoder_data_dir()
    suffix = ('' if not gen_unique else 'unique-') + str(constants.run_index())
    if os.path.isfile(os.path.join(target_dir, 'test-' + suffix + '.npy')) and \
        os.path.isfile(os.path.join(target_dir, 'validation-' + suffix + '.npy')) and \
        os.path.isfile(os.path.join(target_dir, 'train-' + suffix + '.npy')):
        return
    util.prepare_dir(target_dir, hard=False)

    train, test, validation = [], [], []
    tot = -1
    shapes = ['square', 'ellipse', 'triangle'] if not reduced else ['square']
    for shape in shapes:
        _train, _test, _validation = split_for_shape(shape, tot, gen_unique, reduced)
        
        if tot == -1:
            tot = len(_train) + len(_test) + len(_validation)

        random.shuffle(_train)
        random.shuffle(_test)
        random.shuffle(_validation)

        train.extend(_train)
        test.extend(_test)
        validation.extend(_validation)

        random.shuffle(train)
        random.shuffle(test)
        random.shuffle(validation)

    statistics(train, test, validation)

    logger.info('Separation done.')

    np_test = np.array([draw_util.encoded_image_to_flattened_bits(encoded) for encoded in test])
    np_test = np.random.permutation(np_test)
    np.save(os.path.join(target_dir, 'test-' + suffix), np_test)
    logger.info('Test dataset converted.')

    np_validation = np.array([draw_util.encoded_image_to_flattened_bits(encoded)
        for encoded in validation])
    np_validation = np.random.permutation(np_validation)
    np.save(os.path.join(target_dir, 'validation-' + suffix), np_validation)
    logger.info('Validation dataset converted.')

    np_train = np.array([draw_util.encoded_image_to_flattened_bits(encoded) for encoded in train])
    np_train = np.random.permutation(np_train)
    np.save(os.path.join(target_dir, 'train-' + suffix), np_train)
    logger.info('Train dataset converted.')

# Classifier data generation.

def get_random_encoded_image(shape):
    return EncodedImage(shape,
        random.randint(0, constants.POSITION_LIMIT - 1),
        random.randint(0, constants.POSITION_LIMIT - 1),
        random.randint(0, constants.SCALE_LIMIT - 1),
        random.randint(0, constants.ROTATION_LIMIT - 1))

def get_modified_encoded_image(encoded_image):
    encoded_image_base = encoded_image
    t_base = draw_util.compress_bits(draw_util.encoded_image_to_flattened_bits(encoded_image))
    while True:
        encoded_image = encoded_image_base.clone()
        change_factor = random.randint(0, 3)
        change_dir = random.choice([-1, 1])
        if change_factor == 0:
            # Position X
            delta = random.randint(1, constants.POSITION_LIMIT - 1)
            encoded_image.position_x += constants.POSITION_LIMIT + delta * change_dir
            encoded_image.position_x %= constants.POSITION_LIMIT
        elif change_factor == 1:
            # Position Y
            delta = random.randint(1, constants.POSITION_LIMIT - 1)
            encoded_image.position_y += constants.POSITION_LIMIT + delta * change_dir
            encoded_image.position_y %= constants.POSITION_LIMIT
        elif change_factor == 2:
            # Radius
            delta = random.randint(1, constants.SCALE_LIMIT - 1)
            encoded_image.radius += constants.SCALE_LIMIT + delta * change_dir
            encoded_image.radius %= constants.SCALE_LIMIT
        else:
            # Rotation
            delta = random.randint(1, constants.ROTATION_LIMIT - 1)
            encoded_image.rotation += constants.ROTATION_LIMIT + delta * change_dir
            encoded_image.rotation %= constants.ROTATION_LIMIT
        t = draw_util.compress_bits(draw_util.encoded_image_to_flattened_bits(encoded_image))
        if t != t_base:
            return (encoded_image, change_factor)

def create_sample(shape):
    sample = {}
    img = get_random_encoded_image(shape)
    sample['s'] = img.to_dict()
    img, factor = get_modified_encoded_image(img)
    sample['e'] = img.to_dict()
    sample['f'] = factor
    return sample

def create_balanced_dataset(N):
    dataset = []
    for shape in ['ellipse', 'square', 'triangle']:
        for i in range(N / 3 + (1 if N % 3 != 0 else 0)):
            dataset.append(create_sample(shape))
    random.shuffle(dataset)
    return dataset

def gen_classifier_data(index=None):
    target_dir = util.get_classifier_data_dir()
    util.prepare_dir(target_dir, hard=False)
    suffix = str(constants.run_index()) if index is None else str(index)

    if os.path.isfile(os.path.join(target_dir, 'test-' + suffix + '.txt')) and \
        os.path.isfile(os.path.join(target_dir, 'validation-' + suffix + '.txt')) and \
        os.path.isfile(os.path.join(target_dir, 'train-' + suffix + '.txt')):
        return

    TOTAL = 50000
    with open(os.path.join(target_dir, 'train-' + suffix + '.txt'), 'w') \
        as outfile:
        json.dump(create_balanced_dataset(int(TOTAL * TRAIN_RATIO / 100)), outfile)
        logger.info('Train data written.')
    with open(os.path.join(target_dir, 'test-' + suffix + '.txt'), 'w') \
        as outfile:
        json.dump(create_balanced_dataset(int(TOTAL * TEST_RATIO / 100)), outfile)
        logger.info('Test data written.')
    with open(os.path.join(target_dir, 'validation-' + suffix + '.txt'), 'w') \
        as outfile:
        json.dump(create_balanced_dataset(int(TOTAL * VALIDATION_RATIO / 100)), outfile)
        logger.info('Validation data written.')
