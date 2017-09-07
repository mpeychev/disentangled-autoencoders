from fc_autoencoder import FcAutoencoder
from conv_autoencoder import ConvAutoencoder
import constants

def shapes_set(architecture, beta, lr, seq_index, denoising):
    assert (architecture == constants.FC or architecture == constants.CONV)
    if architecture == constants.FC:
        return FcAutoencoder(
            beta=beta,
            learning_rate=lr,
            seq_index=seq_index,
            denoising=denoising)
    else:
        return ConvAutoencoder(64, 10,
            beta=beta,
            learning_rate=lr,
            experiment_name=seq_index,
            denoising=denoising)

def mnist(architecture, beta, is_trainable, seq_index=None, denoising=False):
    if architecture == constants.FC:
        return FcAutoencoder(
            encoder_layers_size=[28 * 28, 512, 512, 16],
            decoder_layers_size=[16, 512, 512, 512, 28 * 28],
            beta=beta,
            learning_rate=0.001 if is_trainable else None,
            seq_index='fc_test' if seq_index is None else seq_index,
            denoising=denoising)
    else:
        return ConvAutoencoder(28, 16,
            beta=beta,
            learning_rate=0.001 if is_trainable else None,
            experiment_name='conv_test' if seq_index is None else seq_index,
            denoising=denoising)
