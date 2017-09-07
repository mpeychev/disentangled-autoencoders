from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def show_images_general_purpose(images, name=None):
    dim = images.shape[0]
    n = int(np.ceil(np.sqrt(images.shape[1])))
    n_image_rows = int(np.ceil(np.sqrt(dim)))
    n_image_cols = int(np.ceil(dim * 1.0 / n_image_rows))
    gs = gridspec.GridSpec(n_image_rows, n_image_cols, top=1., bottom=0., right=1., left=0.,
        hspace=0.05, wspace=0.)
    for g, count in zip(gs, range(int(dim))):
        ax = plt.subplot(g)
        ax.imshow(images[count,:].reshape((n, n)))
        ax.set_xticks([])
        ax.set_yticks([])
    if name is not None:
        plt.savefig(name + '.png')
    else:
        plt.show()
