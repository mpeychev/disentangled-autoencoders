import os
import unittest

import numpy as np
import tensorflow as tf
from linear_classifier import LinearClassifier

from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = os.path.join(os.path.expanduser('~'), 'MNIST_data')
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

class LinearClassifierTest(unittest.TestCase):
    """Unit tests class for the LinearClassifier.
    
    MNIST problem used as a benchmark.
    """

    @staticmethod
    def train_classifier(classifier):
        for _ in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            classifier.partial_fit(batch_xs, batch_ys)

    @staticmethod
    def filter_by_target(images, labels, digit):
        return zip(*filter(lambda pair: pair[1][digit], zip(images, labels)))

    @classmethod
    def setUpClass(cls):
        FEATURE_DIMENSION = mnist.train.images.shape[1]
        OUTPUT_DIMENSION = mnist.train.labels.shape[1]
        cls.classifier1 = LinearClassifier(FEATURE_DIMENSION, OUTPUT_DIMENSION, add_bias=False)
        cls.classifier2 = LinearClassifier(FEATURE_DIMENSION, OUTPUT_DIMENSION, add_bias=True)
        LinearClassifierTest.train_classifier(cls.classifier1)
        LinearClassifierTest.train_classifier(cls.classifier2)

    def test_predictions(self):
        all_predictions = LinearClassifierTest.classifier1.predict_batch(mnist.test.images)
        index = 0
        for image in mnist.test.images:
            self.assertEqual(all_predictions[index],
                    LinearClassifierTest.classifier1.predict_single(image))
            index += 1

    def test_accuracy_manually(self):
        all_predictions = LinearClassifierTest.classifier1.predict_batch(mnist.test.images)
        correct = 0
        count = 0
        index = 0
        for label in mnist.test.labels:
            correct += (all_predictions[index] == np.argmax(label))
            count += 1
            index += 1
        self.assertAlmostEqual(LinearClassifierTest.classifier1.accuracy(mnist.test.images,
            mnist.test.labels), float(correct) / float(count))

    def test_accuracy(self):
        ERROR_MESSAGE = 'Classifier accuracy is not good enough.'
        THRESHOLD = 0.9

        self.assertGreater(LinearClassifierTest.classifier1.accuracy(mnist.test.images,
            mnist.test.labels), THRESHOLD, ERROR_MESSAGE)
        self.assertGreater(LinearClassifierTest.classifier1.accuracy(mnist.validation.images,
            mnist.validation.labels), THRESHOLD, ERROR_MESSAGE)
        self.assertGreater(LinearClassifierTest.classifier2.accuracy(mnist.test.images,
            mnist.test.labels), THRESHOLD, ERROR_MESSAGE)
        self.assertGreater(LinearClassifierTest.classifier2.accuracy(mnist.validation.images,
            mnist.validation.labels), THRESHOLD, ERROR_MESSAGE)

    def test_individual_digits_accuracy(self):
        ERROR_MESSAGE = 'Classifier accuracy for digit {0} is not good enough.'
        THRESHOLD = 0.8

        for digit in range(10):
            images, labels = LinearClassifierTest.filter_by_target(mnist.test.images,
                    mnist.test.labels, digit)
            self.assertGreater(LinearClassifierTest.classifier1.accuracy(images, labels), THRESHOLD,
                    ERROR_MESSAGE.format(digit))
            self.assertGreater(LinearClassifierTest.classifier2.accuracy(images, labels), THRESHOLD,
                    ERROR_MESSAGE.format(digit))
            
            images, labels = LinearClassifierTest.filter_by_target(mnist.validation.images,
                    mnist.validation.labels, digit)
            self.assertGreater(LinearClassifierTest.classifier1.accuracy(images, labels), THRESHOLD,
                    ERROR_MESSAGE.format(digit))
            self.assertGreater(LinearClassifierTest.classifier2.accuracy(images, labels), THRESHOLD,
                    ERROR_MESSAGE.format(digit))
 
if __name__ == '__main__':
    unittest.main()
