import tensorflow as tf
import numpy as np
import draw_util
import os

class Autoencoder(object):

    def partial_fit(self, targets):
        assert (self.is_training)
        if not self.is_denoising:
            self.sess.run(self.train_op, feed_dict={self.input_layer: targets,
                self.batch_size: [len(targets)]})
        else:
            inputs = draw_util.add_noise(targets) if self.is_denoising == 1 else \
                draw_util.erase(targets)
            self.sess.run(self.train_op, feed_dict={self.input_layer: inputs,
                self.target_layer: targets, self.batch_size: [len(targets)]})

    def calc_reconstruction_loss(self, targets):
        if len(targets) == 40000:
            A = self.calc_reconstruction_loss(targets[:20000])
            B = self.calc_reconstruction_loss(targets[20000:])
            return (A + B) / 2.0
        if not self.is_denoising:
            return self.sess.run(self.reconstruction_loss, feed_dict={self.input_layer: targets,
                self.batch_size: [len(targets)]})
        else:
            inputs = draw_util.add_noise(targets) if self.is_denoising == 1 else \
                draw_util.erase(targets)
            return self.sess.run(self.reconstruction_loss, feed_dict={self.input_layer: inputs,
                self.target_layer: targets, self.batch_size: [len(targets)]})

    def calc_kl_divergence(self, inputs):
        if len(inputs) == 40000:
            A = self.calc_kl_divergence(inputs[:20000])
            B = self.calc_kl_divergence(inputs[20000:])
            return (A + B) / 2.0
        if self.is_denoising == 1:
            inputs = draw_util.add_noise(inputs)
        elif self.is_denoising == 2:
            inputs = draw_util.erase(inputs)
        return self.sess.run(self.kl_divergence, feed_dict={self.input_layer: inputs,
            self.batch_size: [len(inputs)]})

    def calc_cost(self, targets):
        if len(targets) == 40000:
            A = self.calc_cost(targets[:20000])
            B = self.calc_cost(targets[20000:])
            return (A + B) / 2.0
        if not self.is_denoising:
            return self.sess.run(self.cost, feed_dict={self.input_layer: targets,
                self.batch_size: [len(targets)]})
        else:
            inputs = draw_util.add_noise(targets) if self.is_denoising == 1 else \
                draw_util.erase(targets)
            return self.sess.run(self.cost, feed_dict={self.input_layer: inputs,
                self.target_layer: targets, self.batch_size: [len(targets)]})

    def get_code_dimension(self):
        return self.code_dimension

    def get_beta(self):
        return self.beta

    def get_output_layer(self, inputs, ignore_noise=False):
        if not ignore_noise:
            if self.is_denoising == 1:
                inputs = draw_util.add_noise(inputs)
            elif self.is_denoising == 2:
                inputs = draw_util.erase(inputs)
        return self.sess.run(self.output_layer, feed_dict={self.input_layer: inputs,
            self.batch_size: [len(inputs)]})

    def get_output_layer_from_code(self, code):
        return self.sess.run(self.output_layer, feed_dict={self.code: code})

    def get_code(self, inputs, ignore_noise=False):
        if not ignore_noise:
            if self.is_denoising == 1:
                inputs = draw_util.add_noise(inputs)
            elif self.is_denoising == 2:
                inputs = draw_util.erase(inputs)
        return self.sess.run(self.code, feed_dict={self.input_layer: inputs,
            self.batch_size: [len(inputs)]})

    def get_code_mean(self, inputs):
        if self.is_denoising == 1:
            inputs = draw_util.add_noise(inputs)
        elif self.is_denoising == 2:
            inputs = draw_util.erase(inputs)
        return self.sess.run(self.code_mean, feed_dict={self.input_layer: inputs,
            self.batch_size: [len(inputs)]})

    def get_code_variance(self, inputs):
        if self.is_denoising == 1:
            inputs = draw_util.add_noise(inputs)
        elif self.is_denoising == 2:
            inputs = draw_util.erase(inputs)
        code_log_sigma_sq = self.sess.run(self.code_log_sigma_sq, feed_dict = {
            self.input_layer: inputs, self.batch_size: [len(inputs)]})
        return np.exp(code_log_sigma_sq)

    def calc_reconstruction_accuracy(self, targets):
        inputs = targets if not self.is_denoising else (draw_util.add_noise(targets) if \
            self.is_denoising == 1 else draw_util.erase(targets))
        predicted_images = self.get_output_layer(inputs)
        return np.mean(np.sqrt(np.sum(np.square(predicted_images - targets), axis=1)))

    def save_model(self):
        self.saver.save(self.sess, os.path.join(self.logs_dir, 'model'))

    def restore_model(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.logs_dir))

    def close_session(self):
        self.sess.close()
        tf.reset_default_graph()
