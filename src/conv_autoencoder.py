import os
import tensorflow as tf
import numpy as np
import util
from base_autoencoder import Autoencoder

MNIST_DIMENSION = 28
SHAPES_DIMENSION = 64

class ConvAutoencoder(Autoencoder):

    def __init__(self,
            image_dimension, # only squared images assumed
            code_dimension,
            beta=None,
            encoder_activation_fn=tf.nn.relu,
            decoder_activation_fn=tf.tanh,
            learning_rate=None,
            experiment_name=None,
            denoising=False):
        print('Construct convolutional autoencoder:')
        print('Image dimension: {0}'.format(image_dimension))
        print('Code dimension: {0}'.format(code_dimension))
        print('Encoder activation function: {0}'.format(encoder_activation_fn))
        print('Decoder activation function: {0}'.format(decoder_activation_fn))
        print('Beta = {0}'.format(beta))
        print('Experiment name: {0}'.format(experiment_name))
        print('Is trainable: {0}'.format(learning_rate is not None))
        print('Is denoising: {0}'.format(denoising))
        print('Learning rate: {0}'.format(learning_rate))
        print('Logs dir: {0}'.format(os.path.join(util.get_logs_dir(), experiment_name)))

        self.image_dimension = image_dimension
        self.code_dimension = code_dimension
        self.beta = beta
        self.encoder_activation_fn = encoder_activation_fn
        self.decoder_activation_fn = decoder_activation_fn
        self.is_training = (learning_rate is not None)
        self.is_denoising = denoising
        self.optimizer = tf.train.AdamOptimizer(learning_rate if learning_rate >= 0 else 0.001) \
            if self.is_training else None
        self.logs_dir = os.path.join(util.get_logs_dir(), experiment_name)

        if self.is_training:
            util.prepare_dir(self.logs_dir)
        else:
            assert (os.path.exists(self.logs_dir))

        self.encoder_get_weights = util.get_weights_he \
            if 'elu' in encoder_activation_fn.__name__ else util.get_weights_xavier
        self.decoder_get_weights = util.get_weights_he \
            if 'elu' in decoder_activation_fn.__name__ else util.get_weights_xavier

        self._describe_autoencoder()
        self._define_loss_function()

        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=1)

        self.sess = tf.InteractiveSession()
        self.sess.run(init_op)

    def _describe_autoencoder(self):
        assert (self.image_dimension == MNIST_DIMENSION or self.image_dimension == SHAPES_DIMENSION)
        if self.image_dimension == MNIST_DIMENSION:
            self._build_input()
            print self.current_layer.get_shape()
            self._stack_conv([3, 3, 1, 8])
            print self.current_layer.get_shape()
            self._stack_conv([3, 3, 8, 16])
            print self.current_layer.get_shape()
            self._stack_fc(512)
            print self.current_layer.get_shape()
            self._build_code()
            print self.current_layer.get_shape()
            self._stack_fc(512)
            print self.current_layer.get_shape()
            self._stack_fc(7 * 7 * 16)
            print self.current_layer.get_shape()
            self._unflatten([7, 7, 16])
            print self.current_layer.get_shape()
            self._stack_deconv([3, 3, 8, 16], [self.batch_size[0], 14, 14, 8])
            print self.current_layer.get_shape()
            self._stack_deconv([3, 3, 4, 8], [self.batch_size[0], 28, 28, 4])
            print self.current_layer.get_shape()
            self._build_output([3, 3, 1, 4])
        else:
            self._build_input()
            print self.current_layer.get_shape()
            self._stack_conv([3, 3, 1, 16])
            print self.current_layer.get_shape()
            self._stack_conv([3, 3, 16, 32])
            print self.current_layer.get_shape()
            self._stack_fc(1024)
            print self.current_layer.get_shape()
            self._build_code()
            print self.current_layer.get_shape()
            self._stack_fc(1024)
            print self.current_layer.get_shape()
            self._stack_fc(8 * 8 * 64)
            print self.current_layer.get_shape()
            self._unflatten([8, 8, 64])
            print self.current_layer.get_shape()
            self._stack_deconv([3, 3, 16, 64], [self.batch_size[0], 16, 16, 16])
            print self.current_layer.get_shape()
            self._stack_deconv([3, 3, 4, 16], [self.batch_size[0], 32, 32, 4])
            print self.current_layer.get_shape()
            self._build_output([3, 3, 1, 4])

    def _build_input(self):
        with tf.name_scope('Input'):
            self.input_layer = tf.placeholder(tf.float32, shape=[None, self.image_dimension ** 2],
                name='flattened_layer')
            self.target_layer = self.input_layer if not self.is_denoising else tf.placeholder(
                tf.float32, shape=[None, self.image_dimension ** 2], name='flattened_target_layer')
            self.batch_size = tf.placeholder(tf.int32, shape=[1], name='batch_size')

        self.current_layer = tf.reshape(self.input_layer, shape=[-1, self.image_dimension,
            self.image_dimension, 1])
        self.code_built = False
        self.encoder_layers_counter = 1

    def _build_code(self):
        self._flatten()
        fan_in = self.current_layer.get_shape().as_list()[1]
        with tf.name_scope('Code'):
            with tf.name_scope('mean'):
                mean_weights = util.get_weights_he((fan_in, self.code_dimension))
                mean_biases = util.get_bias(self.code_dimension)

                self.code_mean = tf.nn.bias_add(
                    tf.matmul(self.current_layer, mean_weights), mean_biases, name='layer')

            with tf.name_scope('stddev'):
                stddev_weights = util.get_weights_he((fan_in, self.code_dimension))
                stddev_biases = util.get_bias(self.code_dimension)

                self.code_log_sigma_sq = tf.nn.bias_add(
                    tf.matmul(self.current_layer, stddev_weights), stddev_biases, name='layer')

            epsilon = tf.random_normal([self.batch_size[0], self.code_dimension])
            self.code = tf.add(self.code_mean,
                tf.multiply(tf.sqrt(tf.exp(self.code_log_sigma_sq)), epsilon), name='layer')

        self.current_layer = self.code
        self.code_built = True
        self.decoder_layers_counter = 1

    def _build_output(self, filter_shape):
        stride = self.image_dimension // self.current_layer.get_shape().as_list()[1]
        with tf.name_scope('Output'):
            W = util.get_weights_xavier(filter_shape)
            b = util.get_bias(filter_shape[2])
            final_layer = tf.nn.bias_add(
                tf.nn.conv2d_transpose(self.current_layer, W,
                    [self.batch_size[0], self.image_dimension, self.image_dimension, 1],
                    [1, stride, stride, 1], 'SAME'),
                b, name='logits')
            self.logits = tf.reshape(final_layer, [self.batch_size[0], self.image_dimension ** 2])
            self.output_layer = tf.sigmoid(self.logits, name='layer')
        del self.current_layer

    def _define_loss_function(self):
        with tf.name_scope('CostFunction'):
            self.reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits, labels=self.target_layer),
                1), name='reconstruction')
            self.kl_divergence = tf.constant(0.0, name='kl_divergence') if self.beta < 1e-3 else \
                self.beta * tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(self.code_mean)
                + tf.exp(self.code_log_sigma_sq) - self.code_log_sigma_sq - 1, 1),
                name='kl_divergence')
            self.cost = tf.add(self.reconstruction_loss, self.kl_divergence, name='cost')

        if self.is_training:
            self.train_op = self.optimizer.minimize(self.cost)

    def _flatten(self):
        shape_list = self.current_layer.get_shape().as_list()
        if len(shape_list) == 4:
            self.current_layer = tf.reshape(self.current_layer, [self.batch_size[0],
                shape_list[1] * shape_list[2] * shape_list[3]])

    def _unflatten(self, dimensions):
        shape_list = self.current_layer.get_shape().as_list()
        assert (len(shape_list) == 2 and len(dimensions) == 3)
        assert (shape_list[1] == dimensions[0] * dimensions[1] * dimensions[2])
        self.current_layer = tf.reshape(self.current_layer, [self.batch_size[0],
            dimensions[0], dimensions[1], dimensions[2]])

    def _stack_conv(self, filter_shape, stride=2):
        assert (not self.code_built)
        with tf.name_scope('EncoderHidden_' + str(self.encoder_layers_counter)):
            W = self.encoder_get_weights(filter_shape)
            next_layer = self.encoder_activation_fn(util.batch_norm(tf.nn.conv2d(self.current_layer,
                W, [1, stride, stride, 1], 'SAME'), self.is_training), name='layer')

        self.current_layer = next_layer
        self.encoder_layers_counter += 1

    def _stack_deconv(self, filter_shape, output_shape, stride=2):
        assert (self.code_built)
        assert (len(self.current_layer.get_shape().as_list()) == 4)
        with tf.name_scope('DecoderHidden_' + str(self.decoder_layers_counter)):
            W = self.decoder_get_weights(filter_shape)
            deconv_layer = tf.nn.conv2d_transpose(self.current_layer, W, output_shape,
                [1, stride, stride, 1], 'SAME')
            deconv_layer = tf.reshape(deconv_layer, output_shape)
            next_layer = self.decoder_activation_fn(util.batch_norm(deconv_layer, self.is_training),
                name='layer')

        self.current_layer = next_layer
        self.decoder_layers_counter += 1

    def _stack_fc(self, dimension):
        self._flatten()
        fan_in = self.current_layer.get_shape().as_list()[-1]
        fan_out = dimension
        if not self.code_built:
            # Encoder.
            with tf.name_scope('EncoderHidden_' + str(self.encoder_layers_counter)):
                W = self.encoder_get_weights((fan_in, fan_out))
                next_layer = self.encoder_activation_fn(util.batch_norm(
                    tf.matmul(self.current_layer, W), self.is_training), name='layer')

            self.encoder_layers_counter += 1
        else:
            # Decoder.
            with tf.name_scope('DecoderHidden_' + str(self.decoder_layers_counter)):
                W = self.decoder_get_weights((fan_in, fan_out))
                next_layer = self.decoder_activation_fn(util.batch_norm(
                    tf.matmul(self.current_layer, W), self.is_training), name='layer')

            self.decoder_layers_counter += 1

        self.current_layer = next_layer
