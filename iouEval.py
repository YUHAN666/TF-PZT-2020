""" Calculate the IoU between mask and model output """
import numpy as np


class iouEval:

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.fp = 0
        self.fn = 0
        self.tp = 0

    def reset(self):
        self.fp = 0
        self.fn = 0
        self.tp = 0

    def addBatch(self, label, output):
        self.output = output
        self.label = label
        classes = self.output.shape[-1]
        for batch in range(self.batch_size):
            # for channel in range(classes):
                # output = np.squeeze(output[batch, :, :, :], axis=-1)
                # label = np.squeeze(label[batch, :, :, :], axis=-1)
            output = self.output[batch, :, :]
            label = self.label[batch, :, :]

            output = np.where(output > 0.5, 1.0, 0.0)

            tp = np.sum(np.multiply(output, label))
            fp = np.sum(output)-tp
            fn = np.sum(label)-tp

            self.tp += tp
            self.fp += fp
            self.fn += fn

    def getIoU(self):

        return self.tp/(self.fp+self.tp+self.fn)
