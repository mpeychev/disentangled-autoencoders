import tensorflow as tf
import numpy as np
import util

class LinearClassifier(object):
    """LinearClassifier which we will use to quantify the level of disentanglement.

    It is implemented as a fully connected neural network with softmax output and cross-entropy cost
    function. It is supposed to have low capacity so that it cannot do the disentanglement itselft.

    Attributes:
        input_dimension: The number of input features passed to the classifier. Corresponds to the
            dimension of the code in the autoencoder.
        possible_classes_count: Dimension of the output layer of the neural network - the number of
            possibilities the classifier has to classify the input. Corresponds to the number of
            generative factors (e.g. position, scale, rotation, etc.) in the dataset.
    """

    def __init__(self, input_dimension, possible_classes_count, add_bias=True,
            optimizer=tf.train.AdamOptimizer()):
        """Initialize the classifier."""
        self.input_dimension = input_dimension
        self.possible_classes_count = possible_classes_count
        self._add_bias = add_bias
        self._optimizer = optimizer

        self._build_network()
        self._define_loss_function()

        # Operation for initializing all tensorflow variables.
        init_op = tf.global_variables_initializer()

        # Launch the default tensorflow session.
        self.sess = tf.InteractiveSession()
        self.sess.run(init_op)

    def _build_network(self):
        """Create the classifier network."""
        self.input_layer = tf.placeholder(tf.float32, [None, self.input_dimension])
        weights = util.get_weights_xavier((self.input_dimension, self.possible_classes_count))

        if self._add_bias:
            bias = util.get_bias(self.possible_classes_count)
            self.prediction_logits = tf.nn.bias_add(tf.matmul(self.input_layer, weights), bias)
        else:
            self.prediction_logits = tf.matmul(self.input_layer, weights)

    def _define_loss_function(self):
        """Define the softmax cross entropy loss function and initialise the tensorflow
        optimizer."""
        self.targets = tf.placeholder(tf.float32, [None, self.possible_classes_count])
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.prediction_logits, labels=self.targets))
        self.train_op = self._optimizer.minimize(self.cost)

    def partial_fit(self, train_inputs, train_targets):
        """Train model based on mini-batches of training data."""
        self.sess.run(self.train_op, feed_dict={self.input_layer: train_inputs,
            self.targets: train_targets})

    def get_cost(self, batch_inputs, batch_targets):
        """Get cross entropy loss evaluated on the batch input."""
        return self.sess.run(self.cost, feed_dict={self.input_layer: batch_inputs,
            self.targets: batch_targets})

    def _get_prediction_logits(self, batch_inputs):
        """Get the values of the output layer of the network."""
        return self.sess.run(self.prediction_logits, feed_dict={self.input_layer: batch_inputs})

    def predict_batch(self, batch_inputs):
        """Return the predictions for a batch of input images."""
        return np.argmax(self._get_prediction_logits(batch_inputs), 1)

    def predict_single(self, input_image):
        """Return the prediction for a single input image."""
        return self.predict_batch([input_image])[0]

    def accuracy(self, test_inputs, test_targets):
        """Evaluate the accuracy of the classifier on the test data."""
        correct_predictions = np.equal(self.predict_batch(test_inputs), np.argmax(test_targets, 1))
        return np.mean(correct_predictions)

    def close_session(self):
        self.sess.close()
        tf.reset_default_graph()
