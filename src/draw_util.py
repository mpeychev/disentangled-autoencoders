from PIL import Image, ImageDraw
import math
import itertools
import numpy as np
import constants
import skimage
import random

class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def add(self, vec):
        return Point(self.x + vec.x, self.y + vec.y)

    def subtract(self, vec):
        return Point(self.x - vec.x, self.y - vec.y)

IMG_SIZE = 64
WORK_IMG_SIZE = 128

SHIFT = (IMG_SIZE - constants.POSITION_LIMIT) / 2
SCALE_SHIFT = 8

def encoded_image_to_flattened_bits(encoded_image):
    """Generates a flattened bits image from an EncodedImage instance.

    Receives the generative factors packed in an EncodedImage object and returns the binary image as
    a flattened bit vector.

    Args:
        encoded_image: An EncodedImage instance describing the generative factors.
    Returns:
        The resulting binary image as a flattened bit vector stored as a NumPy array.
    """
    shape = encoded_image.shape
    position_x = encoded_image.position_x
    position_y = encoded_image.position_y
    radius = encoded_image.radius
    rotation = encoded_image.rotation

    assert (shape == 'ellipse' or shape == 'square' or shape == 'triangle')
    assert (position_x >= 0 and position_x < constants.POSITION_LIMIT)
    assert (position_y >= 0 and position_y < constants.POSITION_LIMIT)
    assert (radius >= 0 and radius < constants.SCALE_LIMIT)
    assert (rotation >= 0 and rotation < constants.ROTATION_LIMIT)

    position_x = SHIFT + position_x
    position_y = SHIFT + position_y
    radius += SCALE_SHIFT

    O = Point(WORK_IMG_SIZE / 2, WORK_IMG_SIZE / 2)

    im = Image.new('L', (WORK_IMG_SIZE, WORK_IMG_SIZE), 'black')
    draw = ImageDraw.Draw(im)

    # Draw required shape.
    if shape == 'ellipse':
        draw.ellipse([O.x - radius / 2, O.y - radius, O.x + radius / 2, O.y + radius], fill='white')
    elif shape == 'square':
        draw.rectangle([O.x - radius, O.y - radius, O.x + radius, O.y + radius], fill='white')
    elif shape == 'triangle':
        deltas = [Point(radius * math.sqrt(3.0) / 2.0, radius / 2.0), Point(0.0, -radius),
            Point(-radius * math.sqrt(3.0) / 2.0, radius / 2.0)]
        draw.polygon(list(itertools.chain(*[(vec.add(O).x, vec.add(O).y) for vec in deltas])),
            fill='white')
    else:
        raise InvalidShapeException
    del draw

    # Rotate.
    im = im.rotate(rotation * constants.ROTATION_DELTA, Image.BICUBIC, expand=True)

    # Translate.
    O = Point(im.size[0] / 2, im.size[1] / 2)
    left_up = O.subtract(Point(position_x, position_y))
    right_down = left_up.add(Point(IMG_SIZE, IMG_SIZE))

    # Crop and return.
    ma3x = np.asarray(im.crop((left_up.x, left_up.y, right_down.x, right_down.y)).resize(
        (IMG_SIZE, IMG_SIZE), Image.BICUBIC)).flatten()
    map_func = np.vectorize(lambda pixel: (pixel >= 127.5), otypes=[np.float32])
    return map_func(ma3x)

def compress_bits(flattened_bits):
    return tuple(np.packbits(flattened_bits.astype(np.bool_)))

def add_noise(clean):
    ADD_NOISE_PERCENTAGE = 0.9
    border = int(ADD_NOISE_PERCENTAGE * len(clean))
    if border == 0:
        return clean
    return np.concatenate((
        skimage.util.random_noise(clean[:border], mode='s&p', amount=constants.NOISE),
        clean[border:]))

# MNIST support

delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def erase_single(vec):
    vec = np.reshape(vec, (28, 28))
    non_zero_positions = []
    for i in range(28):
        for j in range(28):
            if vec[i][j] > 0.0:
                non_zero_positions.append((i, j))
    P = random.uniform(0.0, 0.15)
    p = int(P * len(non_zero_positions))
    q = [random.choice(non_zero_positions)]
    while p > 0:
        if not q:
            non_zero_positions = list(filter(lambda (x, y): vec[x][y] > 0.0, non_zero_positions))
            q = [random.choice(non_zero_positions)]
        random.shuffle(q)
        (x, y) = q.pop()
        vec[x][y] = 0.0
        for (dx, dy) in delta:
            nxt_x, nxt_y = x + dx, y + dy
            if nxt_x >= 0 and nxt_x < 28 and nxt_y >= 0 and nxt_y < 28 and vec[nxt_x][nxt_y] > 0.0:
                q.append((nxt_x, nxt_y))
        p -= 1
    vec = np.reshape(vec, (28 * 28))
    return vec

def erase(images):
    return np.apply_along_axis(erase_single, 1, images)
