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
save_parameters = True
use_parameters = False

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


# try
if use_parameters:
    parameters = np.load('parameters.npz')
    model.layer_list[0].W = parameters['conv1_w']
    model.layer_list[0].b = parameters['conv1_b']
    model.layer_list[3].W = parameters['conv2_w']
    model.layer_list[3].b = parameters['conv2_b']
    model.layer_list[-1].W = parameters['fc3_w']
    model.layer_list[-1].b = parameters['fc3_b']
    acc_value = test_net(model, loss, test_data, test_label, config['batch_size'])
    print('acc = ' + str(acc_value))


for epoch in range(config['max_epoch']):
    current_iter_count, loss_values = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'], current_iter_count)
    log_list = log_list + loss_values
    if epoch == config['max_epoch'] / 100 - 1 or epoch == config['max_epoch'] / 10 - 1 or epoch == config['max_epoch'] - 1:
        inputs, labels = data_iterator(train_data, train_label, config['batch_size']).next()
        model.forward(inputs)
        batch_conv1_output = model.layer_list[1]._forward_output
        batch_conv2_output = model.layer_list[4]._forward_output
        indices = []
        labels_set = set()
        for i in range(len(labels)):
            if labels[i] not in labels_set and len(indices) < img_record_num:
                labels_set.add(labels[i])
                indices.append(i)
            elif len(indices) == img_record_num:
                break
        # debug
        print(str(indices))
        print(str([labels[i] for i in indices]))
        imgs = [inputs[j] for j in indices]
        conv1_outputs = [batch_conv1_output[k] for k in indices]
        conv2_outputs = [batch_conv2_output[l] for l in indices]
        np.savez('imgs_after_' + str(epoch + 1) + '_epochs.npz', imgs=imgs, conv1_outputs=conv1_outputs, conv2_outputs=conv2_outputs)

    if epoch == config['max_epoch'] - 1:
        # accuracy on testset
        acc_value = test_net(model, loss, test_data, test_label, config['batch_size'])

        # save parameters
        if save_parameters:
            np.savez('parameters.npz', conv1_w=model.layer_list[0].W, conv1_b=model.layer_list[0].b, conv2_w=model.layer_list[3].W, conv2_b=model.layer_list[3].b, fc3_w=model.layer_list[-1].W, fc3_b=model.layer_list[-1].b)

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
