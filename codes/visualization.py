import numpy as np
import matplotlib.pyplot as plt

img_path_prefixes = ['imgs_after_1_epochs']  # , 'imgs_after_10_epochs', 'imgs_after_100_epochs']


def vis_square(data, path):
    """
    
        Take an array of shape (n, height, width) or (n, height, width, 3)
        and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
        refer to: http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb
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

for img_path_prefix in img_path_prefixes:
    img_info = np.load(img_path_prefix + '.npz')
    imgs = img_info['imgs']
    imgs = imgs.reshape(imgs.shape[0], imgs.shape[2], imgs.shape[3])
    conv1_outputs = img_info['conv1_outputs']
    conv1_outputs = conv1_outputs.transpose(0, 2, 3, 1)
    conv2_outputs = img_info['conv2_outputs']
    conv2_outputs = conv2_outputs.transpose(0, 2, 3, 1)
    plt.gcf().clear()
    vis_square(imgs, img_path_prefix + '_origin.png')
    plt.gcf().clear()
    vis_square(conv1_outputs, img_path_prefix + '_after_conv1.png')
    plt.gcf().clear()
    vis_square(conv2_outputs, img_path_prefix + '_after_conv2.png')
