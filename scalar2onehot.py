import numpy as np
from matplotlib import pyplot as plt


def scalar2onehot(mask):

    # ones = np.zeros(mask.shape)
    ones = np.where(mask == 1.0, 1, 0)
    ones = np.array(ones[:, :, np.newaxis])
    # zeros = np.ones(mask.shape)
    zeros = np.where(mask == 1.0, 0, 1)
    zeros = np.array(zeros[:, :, np.newaxis])

    # plt.imshow(ones, cmap='gray')
    # plt.show()
    # plt.imshow(zeros, cmap='gray')
    # plt.show()

    output = np.concatenate((ones, zeros), axis=-1)
    return output