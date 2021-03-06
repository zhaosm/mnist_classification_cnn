import numpy as np


def conv2d_forward(input, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        input_cols, output: shape = n (#sample) x c_out (#output channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after convolution
    '''
    input_cols = im2col_indices(x=input, field_height=kernel_size, field_width=kernel_size, padding=pad,stride=1)  # (c_in * kernel_size * kernel_size) * (h_out * w_out * n)
    n, h, w = input.shape[0], input.shape[2], input.shape[3]
    h_out = h + 2 * pad - kernel_size + 1
    w_out = w + 2 * pad - kernel_size + 1
    c_out, c_in = W.shape[0], W.shape[1]
    output_conv_transposed = np.dot(W.reshape(c_out, -1), input_cols)
    output_conv_transposed = output_conv_transposed + b.reshape(-1, 1)
    return output_conv_transposed.reshape(c_out, h_out, w_out, n).transpose(3, 0, 1, 2)


def conv2d_backward(input, grad_output, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_out (#output channel) x h_out x w_out
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input
    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_W: gradient of W, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        grad_b: gradient of b, shape = c_out
    '''
    input_cols = im2col_indices(x=input, field_height=kernel_size, field_width=kernel_size, padding=pad, stride=1)
    c_out = grad_output.shape[1]
    grad_output_transposed = grad_output.transpose(1, 2, 3, 0).reshape(c_out, -1)  # inverse process for transpose and reshape in forward
    grad_w_transposed = np.matmul(grad_output_transposed, input_cols.T)  # same as linear layer's backward
    grad_w = grad_w_transposed.reshape(W.shape)

    grad_input_cols = np.matmul(W.reshape(c_out, -1).T, grad_output_transposed)  # same as linear's backward
    grad_input = col2im_indices(cols=grad_input_cols, x_shape=input.shape, field_height=kernel_size,
                                field_width=kernel_size, padding=pad, stride=1)  # calculate gradient for input by summing corresponding gradients in input_cols_grad
    grad_b = np.sum(grad_output, axis=(0, 2, 3))  # sum on axis except c_out

    return grad_input, grad_w, grad_b.ravel()


def avgpool2d_forward(input, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_in (#input channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after average pooling over input
    '''
    n, c_in, h_in, w_in = input.shape
    assert (h_in + 2 * pad) % kernel_size == 0
    assert (w_in + 2 * pad) % kernel_size == 0
    input_reshaped = input.reshape(n * c_in, 1, h_in, w_in)
    input_reshaped_cols = im2col_indices(input_reshaped, kernel_size, kernel_size, pad, kernel_size)  # (1 * kernel_size * kernel_size) * (h_out * w_out * (n * c_in))
    h_out = (h_in + 2 * pad) / kernel_size
    w_out = (w_in + 2 * pad) / kernel_size
    return np.mean(input_reshaped_cols, axis=0).reshape(h_out, w_out, n, c_in).transpose(2, 3, 0, 1)


def avgpool2d_backward(input, grad_output, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_in (#input channel) x h_out x w_out
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    '''
    n, c_in, h_in, w_in = input.shape
    assert (h_in + 2 * pad) % kernel_size == 0
    assert (w_in + 2 * pad) % kernel_size == 0
    grad_output_reshaped = grad_output.transpose(2, 3, 0, 1).ravel()  # inverse process for transpose and reshape in forward
    input_cols_grad = np.zeros((kernel_size * kernel_size, grad_output_reshaped.shape[0]))
    input_cols_grad[:, range(grad_output_reshaped.shape[0])] = grad_output_reshaped / kernel_size
    input_grad = col2im_indices(input_cols_grad, (n * c_in, 1, h_in, w_in), kernel_size, kernel_size, padding=pad, stride=kernel_size)
    return input_grad.reshape(input.shape)


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    """

    refer to: https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py
    """
    # first figure out what the size of the output should be
    n, c, h, w = x_shape
    assert (h + 2 * padding - field_height) % stride == 0
    assert (w + 2 * padding - field_height) % stride == 0
    out_height = (h + 2 * padding - field_height) / stride + 1
    out_width = (w + 2 * padding - field_width) / stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, c)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * c)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(c), field_height * field_width).reshape(-1, 1)
    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ 

        An implementation of im2col based on some fancy indexing
        refer to: https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py
    """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0)
    cols = cols.reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    """ 

        An implementation of col2im based on fancy indexing and np.add.at
        refer to: https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py
    """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
