import matplotlib.pyplot as plt
import numpy as np


def print_images(labels, images, _figsize=(10, 10)):
    """
        Function to plot a set of images
    :param labels: the labels
    :param images: the images
    :param _figsize: the size of the figure
    :return: None
    """
    # assert labels.shape == images.shape
    f, axarr = plt.subplots(labels.shape[0], labels.shape[1], figsize=_figsize)

    rows, cols = labels.shape
    # Check the dimension of axarr and adjust accordingly
    if rows == 1 and cols == 1:
        axarr = np.array([[axarr]])  # Convert to 2D array
    elif rows == 1 or cols == 1:
        axarr = axarr.reshape(rows, cols)  # Convert to 2D array

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            axarr[i, j].set_title(labels[i, j])
            axarr[i, j].imshow(images[i, j], cmap='gray', vmin=0, vmax=255)

    plt.show()


def get_offsets(kernel, origin):
    """
        Function to get the offsets of a kernel.
        Compute the offsets that should be used based on the origin of the kernel
        Later I will use this to convolute the image regarding the origin, so I should pad properly
    :param kernel: the kernel
    :param origin: the origin of the kernel
    :return: the offsets
    """
    # get the kernel size
    k1, k2 = kernel.shape
    # get the origin
    o1, o2 = origin

    # get the offsets
    left = o2
    right = k2 - o2 - 1
    top = o1
    bottom = k1 - o1 - 1

    return left, right, top, bottom


def padding(img, left, right, top, bottom):
    """
        Function to pad an image with zeros based on the kernel size
    :param img: the image to be padded
    :param kernel_size: tuple (m, n). Remember m = 2a +1, n = 2b + 1
    :return: the padded image
    """
    l_img = np.pad(img, ((top, bottom), (left, right)), 'constant', constant_values=0)
    return l_img


def _padding(img, kernel_size):
    """
        Function to pad an image with zeros based on the kernel size
    :param img: the image to be padded
    :param kernel_size: tuple (m, n). Remember m = 2a +1, n = 2b + 1
    :return: the padded image
    """

    pad_ud = int((kernel_size[0] - 1) / 2)
    pad_lr = int((kernel_size[1] - 1) / 2)
    return padding(img, pad_lr, pad_lr, pad_ud, pad_ud)


def convolution(img, kernel):
    """
        Function to apply convolution to an image
    :param img: the image to be convoluted
    :param kernel: the kernel to be used
    :return: the convoluted image
    """
    # get the kernel size
    k1, k2 = kernel.shape
    # get the image size
    i1, i2 = img.shape
    # create a new image with the same size of the original
    new_img = np.zeros((i1, i2))
    # pad the image ( I am considering symmetric padding - square kernel)
    l_padded = _padding(img, kernel.shape)
    # apply convolution
    for i in range(i1):
        for j in range(i2):
            new_img[i, j] = np.sum(l_padded[i:i + k1, j:j + k2] * kernel)

    return new_img


def gradient(img_1, img_2):
    """
        Function to calculate the gradient magnitude
    :param img_1: first image of gradient
    :param img_2: second image of gradient
    :return: the gradient magnitude
    """
    return np.sqrt(img_1 ** 2 + img_2 ** 2)


# calculando histograma de uma imagem
def histogram(img, max_value):
    print(img.shape)
    l_hist = np.zeros(max_value + 1, dtype=np.uint32)
    np.add.at(l_hist, img, 1)
    return l_hist


def threshold_image(img1, threshold):
    """
        Function to threshold an image
    :param img1: the image to be thresholded
    :param threshold: the threshold value
    :return: the thresholded image
    """
    img1[img1 > threshold] = 255
    img1[img1 <= threshold] = 0
    return img1

