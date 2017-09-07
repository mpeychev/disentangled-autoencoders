from PIL import Image
import os
import util
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



def show_images(images, save_name, hard=False):
    print images.shape
    dim = images.shape[0]
    if hard:
        images = np.array(
            map(lambda image: map(lambda pixel: 0.0 if pixel < 0.5 else 1.0, image), images))
    n_image_rows = images.shape[0] / 10
    n_image_cols = 10
    gs = gridspec.GridSpec(n_image_rows, n_image_cols, hspace=0., wspace=0.)
    for i in range(n_image_rows):
        for j in range(n_image_cols):
            ax = plt.subplot(gs[i * n_image_cols + j])
            ax.imshow(images[i * n_image_cols + j].reshape((64, 64)))
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0 and j == 0:
                ax.set_title('scale')
            if i == 0 and j == 3:
                ax.set_title('y')
            if i == 0 and j == 4:
                ax.set_title('x')
            if i == 0 and j == 7:
                ax.set_title('rotation')
            if j == 0 and i == 0:
                ax.set_ylabel('+1.0')
            if j == 0 and i == 1:
                ax.set_ylabel('+0.5')
            if j == 0 and i == 2:
                ax.set_ylabel('base')
            if j == 0 and i == 3:
                ax.set_ylabel('-0.5')
            if j == 0 and i == 4:
                ax.set_ylabel('-1.0')
            if i == n_image_rows - 1:
                ax.set_xlabel('z{0}'.format(j))
            ax.set_aspect('equal')
            plt.subplots_adjust(wspace=None, hspace=None)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig(save_name + '_vis.png')

results_dir = util.get_results_dir()
images = np.load(os.path.join(results_dir, 'pictures_4.npy'))
indexes = []
SHIFT_RANGE = len(images) / 10
for shift in range(SHIFT_RANGE):
    for i in range(10):
        indexes.append(i * SHIFT_RANGE + shift)
new_images = []
for index in indexes:
    new_images.append(images[index])
show_images(np.array(new_images), 'all')
