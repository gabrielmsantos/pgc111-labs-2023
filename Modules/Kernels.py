# KERNELS
import matplotlib.pyplot as plt
import numpy as np


def mean_kernel(size):
    """
        Function to create a mean kernel
    :param size: tuple (m, n). Remember m = 2a +1, n = 2b + 1
    :return: the mean kernel
    """
    return np.ones(size) / (size[0] * size[1])


def gaussian_kernel(size, sigma=1, K=1):
    """
        Function to create a gaussian kernel
    :param K: constant multiplier
    :param sigma: standard deviation
    :param size: tuple (m, n). Remember m = 2a +1, n = 2b + 1
    :return: the gaussian kernel
    """
    # create the kernel
    kernel = np.zeros(size)
    # get the center of the kernel
    center = (int((size[0] - 1) / 2), int((size[1] - 1) / 2))
    # fill the kernel
    for i in range(size[0]):
        for j in range(size[1]):
            kernel[i, j] = K * np.exp(-((i - center[0]) ** 2 + (j - center[1]) ** 2) / (2 * sigma ** 2))
    # normalize the kernel
    kernel /= np.sum(kernel)
    return kernel


def sobel_kernels():
    """
        Function to return the sobel kernel
    :return: the sobel kernels vertical and horizontal
    """
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]]), np.array([[-1, -2, -1],
                                             [0, 0, 0],
                                             [1, 2, 1]]),


def roberts_cross_kernel():
    """
        Function to return the roberts cross kernel
    :return:
    """
    # livro do gonzalez:
    return np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]]), np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    # slide do professor:
    # return np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]]), np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])


def laplacian_kernel(diagonals=False):
    """
        Function to return the laplacian kernel
    :return: the laplacian kernel
    """
    if diagonals:
        return np.array([[1, 1, 1],
                         [1, -8, 1],
                         [1, 1, 1]])
    else:
        return np.array([[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]])