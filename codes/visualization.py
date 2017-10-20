import numpy as np
import matplotlib.pyplot as plt
from output_paths import img_arrays_path


def vis_square(data, path):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
       source code: http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb
    """

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    plt.savefig(path)
    plt.axis('off')

image_info = np.load('imgs_after_1_epochs.npz')
images = image_info['imgs']
images = images.reshape(images.shape[0], images.shape[2], images.shape[3])
conv_outputs = image_info['conv_outputs']
conv_outputs = conv_outputs.transpose(0, 2, 3, 1)
vis_square(images, 'origin_images.png')
plt.gcf().clear()
vis_square(conv_outputs, 'after_conv.png')
print('finished. ')