from network import Network
from layers import Relu, Linear, Conv2D, AvgPool2D, Reshape
from utils import LOG_INFO
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_4d
from datetime import datetime
import csv
from solve_net import data_iterator
import numpy as np
from output_paths import *

train_data, test_data, train_label, test_label = load_mnist_4d('data')
img_record_num = 4

# Your model defintion here
# You should explore different model architecture
model = Network()
model.add(Conv2D('conv1', 1, 4, 3, 1, 0.01))
model.add(Relu('relu1'))
model.add(AvgPool2D('pool1', 2, 0))  # output shape: N x 4 x 14 x 14
model.add(Conv2D('conv2', 4, 4, 3, 1, 0.01))
model.add(Relu('relu2'))
model.add(AvgPool2D('pool2', 2, 0))  # output shape: N x 4 x 7 x 7
model.add(Reshape('flatten', (-1, 196)))
model.add(Linear('fc3', 196, 10, 0.1))

loss = SoftmaxCrossEntropyLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.01,
    'weight_decay': 0.0,
    'momentum': 0.9,
    'batch_size': 100,
    'max_epoch': 1,
    'disp_freq': 5,
    'test_epoch': 5
}

start = datetime.now()
display_start = str(start).split(' ')[1][:-3]
log_list = []
current_iter_count = 0


for epoch in range(config['max_epoch']):
    current_iter_count, loss_values = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'], current_iter_count)
    log_list = log_list + loss_values
    if epoch == config['max_epoch'] / 100 - 1 or epoch == config['max_epoch'] / 10 - 1 or epoch == config['max_epoch'] - 1:
        inputs, labels = data_iterator(train_data, train_label, config['batch_size']).next()
        model.forward(inputs)
        batch_conv_output = model.layer_list[1]._forward_output
        indices = set()
        for i in range(len(labels)):
            if labels[i] not in indices and len(indices) < img_record_num:
                indices.add(i)
            elif len(indices) == img_record_num:
                break
        indices = list(indices)
        imgs = [inputs[j] for j in indices]
        conv_outputs = [batch_conv_output[k] for k in indices]
        np.savez('imgs_after_' + str(epoch + 1) + '_epochs.npz', imgs=imgs, conv_outputs=conv_outputs)

    if epoch == config['max_epoch'] - 1:
        # save conv + relu' s output
        inputs, labels = data_iterator(train_data, train_label, config['batch_size']).next()
        model.forward(inputs)
        batch_conv_output = model.layer_list[1]._forward_output
        indices = set()
        for i in range(len(labels)):
            if labels[i] not in indices and len(indices) < img_record_num:
                indices.add(i)
            elif len(indices) == img_record_num:
                break
        indices = list(indices)
        imgs = [inputs[j] for j in indices]
        conv_outputs = [batch_conv_output[k] for k in indices]
        np.savez(img_arrays_path, imgs=imgs, conv_outputs=conv_outputs)

        acc_value = test_net(model, loss, test_data, test_label, config['batch_size'])

now = datetime.now()
display_now = str(now).split(' ')[1][:-3]
logfile = file(logpath, 'wb')
writer = csv.writer(logfile)
writer.writerow(['iter', 'loss'])
data = [(log['iter'], log['loss']) for log in log_list]
writer.writerows(data)
data = [(display_start, ), (display_now, ), (acc_value, )]
writer.writerows(data)
logfile.close()
