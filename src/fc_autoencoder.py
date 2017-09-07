import os
import tensorflow as tf
import numpy as np
import util
import draw_util
from base_autoencoder import Autoencoder

class FcAutoencoder(Autoencoder):

    def __init__(self,
            encoder_layers_size=[4096, 1200, 1200, 10],
            decoder_layers_size=[10, 1200, 1200, 1200, 4096],
            beta=None,
            encoder_activation_fn=tf.nn.relu,
            decoder_activation_fn=tf.tanh,
            learning_rate=None,
            seq_index=None,
            denoising=False):
        print('Construct fully connected autoencoder:')
        print('Encoder layers: {0}'.format(encoder_layers_size))
        print('Decoder layers: {0}'.format(decoder_layers_size))
        print('Encoder activation function: {0}'.format(encoder_activation_fn))
        print('Decoder activation function: {0}'.format(decoder_activation_fn))
        print('Beta = {0}'.format(beta))
        print('Seq index = {0}'.format(seq_index))
        print('Is trainable: {0}'.format(learning_rate is not None))
        print('Is denoising: {0}'.format(denoising))
        print('Learning rate: {0}'.format(learning_rate))
        print('Logs dir: {0}'.format(os.path.join(util.get_logs_dir(), seq_index)))

        self.encoder_layers_size = encoder_layers_size
        self.decoder_layers_size = decoder_layers_size
        self.code_dimension = encoder_layers_size[-1]
        self.beta = beta
        self.encoder_activation_fn = encoder_activation_fn
        self.decoder_activation_fn = decoder_activation_fn
        self.is_training = (learning_rate is not None)
        self.is_denoising = denoising
        self.optimizer = tf.train.AdamOptimizer(learning_rate) if self.is_training else None
        self.logs_dir = os.path.join(util.get_logs_dir(), seq_index)

        if self.is_training:
            util.prepare_dir(self.logs_dir)
        else:
            assert (os.path.exists(self.logs_dir))

        self.encoder_get_weights = util.get_weights_he \
            if 'elu' in encoder_activation_fn.__name__ else util.get_weights_xavier
        self.decoder_get_weights = util.get_weights_he \
            if 'elu' in decoder_activation_fn.__name__ else util.get_weights_xavier

        self._build_network()
        self._define_loss_function()

        init_op = tf.global_variables_initializer()
        self.merged_summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=1)

        self.sess = tf.InteractiveSession()
        self.sess.run(init_op)

    def _build_network(self):
        self._build_input()
        self._build_code(self._build_encoder())
        self._build_output(self._build_decoder())

    def _build_input(self):
        with tf.name_scope('Input'):
            self.input_layer = tf.placeholder(tf.float32, shape=[None, self.encoder_layers_size[0]],
                name='layer')
            self.target_layer = self.input_layer if not self.is_denoising else tf.placeholder(
                tf.float32, shape=[None, self.encoder_layers_size[0]], name='target_layer')
            self.batch_size = tf.placeholder(tf.int32, shape=[1], name='batch_size')

    def _build_encoder(self):
        last_layer = self.input_layer
        for i in range(1, len(self.encoder_layers_size) - 1):
            scope = 'EncoderHidden_' + str(i)
            with tf.name_scope(scope):
                W = self.encoder_get_weights((self.encoder_layers_size[i - 1],
                    self.encoder_layers_size[i]))

                current_layer = self.encoder_activation_fn(
                    util.batch_norm(tf.matmul(last_layer, W), self.is_training), name='layer')

            last_layer = current_layer
        return last_layer

    def _build_code(self, last_layer):
        with tf.name_scope('Code'):
            with tf.name_scope('mean'):
                mean_weights = util.get_weights_he((self.encoder_layers_size[-2],
                    self.encoder_layers_size[-1]))
                mean_biases = util.get_bias(self.encoder_layers_size[-1])

                tf.summary.histogram('Code_mean_weights_summary', mean_weights)
                tf.summary.histogram('Code_mean_biases_summary', mean_biases)

                self.code_mean = tf.nn.bias_add(tf.matmul(last_layer, mean_weights), mean_biases,
                    name='layer')

            with tf.name_scope('stddev'):
                stddev_weights = util.get_weights_he((self.encoder_layers_size[-2],
                    self.encoder_layers_size[-1]))
                stddev_biases = util.get_bias(self.encoder_layers_size[-1])

                tf.summary.histogram('Code_stddev_weights_summary', stddev_weights)
                tf.summary.histogram('Code_stddev_biases_summary', stddev_biases)

                self.code_log_sigma_sq = tf.nn.bias_add(tf.matmul(last_layer, stddev_weights),
                    stddev_biases, name='layer')

            epsilon = tf.random_normal([self.batch_size[0], self.encoder_layers_size[-1]])
            self.code = tf.add(self.code_mean,
                tf.multiply(tf.sqrt(tf.exp(self.code_log_sigma_sq)), epsilon), name='layer')

    def _build_decoder(self):
        last_layer = self.code
        for i in range(1, len(self.decoder_layers_size) - 1):
            scope = 'DecoderHidden_' + str(i)
            with tf.name_scope(scope):
                W = self.decoder_get_weights((self.decoder_layers_size[i - 1],
                    self.decoder_layers_size[i]))

                current_layer = self.decoder_activation_fn(
                    util.batch_norm(tf.matmul(last_layer, W), self.is_training), name='layer')

            last_layer = current_layer
        return last_layer

    def _build_output(self, last_layer):
        with tf.name_scope('Output'):
            W = util.get_weights_xavier((self.decoder_layers_size[-2],
                self.decoder_layers_size[-1]))
            b = util.get_bias(self.decoder_layers_size[-1])

            tf.summary.histogram('Output_weights_summary', W)
            tf.summary.histogram('Output_biases_summary', b)

            self.logits = tf.nn.bias_add(tf.matmul(last_layer, W), b, name='logits')
            self.output_layer = tf.sigmoid(self.logits, name='layer')

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

        tf.summary.scalar('cost', self.cost)
        tf.summary.scalar('reconstruction_loss', self.reconstruction_loss)
        tf.summary.scalar('kl_divergence', self.kl_divergence)

    def get_summary(self, targets):
        if not self.is_denoising:
            return self.sess.run(self.merged_summary_op, feed_dict={self.input_layer: targets,
                self.batch_size: [len(targets)]})
        else:
            inputs = draw_util.add_noise(targets)
            return self.sess.run(self.merged_summary_op, feed_dict={self.input_layer: inputs,
                self.target_layer: targets, self.batch_size: [len(targets)]})
