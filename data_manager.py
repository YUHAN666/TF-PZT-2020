import re
import os
import numpy as np
import cv2
from config import *
from random import shuffle
import tensorflow as tf
from my_func import rotate
from skimage import exposure


class DataManager(object):
    def __init__(self, imageList, maskList, param, shuffle=True):
        """
        """
        self.shuffle = shuffle
        self.__Param = param
        self.image_list = imageList
        self.mask_list = maskList
        self.data_size = len(imageList)
        self.data_dir = param["data_dir"]
        self.epochs_num = param["epochs_num"]
        if param["mode"] == "visualization":
            self.batch_size = param["batch_size_inference"]
        else:
            self.batch_size = param["batch_size"]
        self.number_batch = int(np.floor(len(self.image_list) / self.batch_size))
        self.next_batch = self.get_next()

    def get_next(self):
        dataset = tf.data.Dataset.from_generator(self.generator, (tf.float32, tf.float32, tf.float32, tf.string))
        dataset = dataset.repeat(self.epochs_num)
        # if self.shuffle:
        #     dataset = dataset.shuffle(1)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        out_batch = iterator.get_next()
        return out_batch

    def generator(self):
        rand_index = np.arange(len(self.image_list))
        np.random.shuffle(rand_index)
        for index in range(len(self.image_list)):
            image_path = self.image_list[rand_index[index]]
            mask_path = self.mask_list[rand_index[index]]
            # image_path = os.path.join(self.data_dir, file_basename_image)
            if image_path.split('/')[-1].split('_')[0] == 'n':
                label = np.array([0.0])
            else:
                label = np.array([1.0])

            # label = image_path.split('/')
            # label = label[-2]
            # label = label_dict[label]

            image, mask = self.read_data(image_path, mask_path)

            # image = tf.cast(image, tf.float32)
            # mask = tf.cast(mask, tf.float32)
            image = image / 255
            mask = mask / 255
            # image = (np.array(image[:, :, np.newaxis]))
            mask = (np.array(mask[:, :, np.newaxis]))

            if self.__Param["mode"] == "train_decision" or self.__Param["mode"] == "train_segmentation":
                aug_random = np.random.uniform()
                if aug_random > 0.6:
                    expo = np.random.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3])
                    # image = exposure.adjust_gamma(image, expo)
            #                    angle=np.random.randint(1,72)*5
            #                    image=rotate(image,angle)

            # if IMAGE_MODE == 0:  # expanding dimension is needed only in mono mode
            #     image = (np.array(image[:, :, np.newaxis]))

            yield image, mask, label, image_path

    def read_data(self, image_path, mask_path):

        pad_x = np.random.randint(0, 100, 2)
        pad_y = np.random.randint(0, 200, 2)

        img = cv2.imread(image_path, 1)  # /255.#read the gray image
        # img = np.pad(img, ((pad_y[0], pad_y[1]), (pad_x[0], pad_x[1]), (0, 0)), 'constant')
        img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))

        msk = cv2.imread(mask_path, 0)  # /255.#read the gray image
        # msk = np.pad(msk, ((pad_y[0], pad_y[1]), (pad_x[0], pad_x[1])), 'constant')
        msk = cv2.resize(msk, (IMAGE_SIZE[1], IMAGE_SIZE[0]))

        # image augmentation by randomly flip
        # p = np.random.random(1)
        # if p < 0.25:
        #     img = cv2.flip(img, 1)
        #     msk = cv2.flip(msk, 1)
        # elif 0.25 < p < 0.5:
        #     img = cv2.flip(img, 0)
        #     msk = cv2.flip(msk, 0)
        # elif 0.5 < p < 0.75:
        #     img = cv2.flip(img, -1)
        #     msk = cv2.flip(msk, -1)
        # else:
        #     pass

        # img = img.swapaxes(0, 1)
        # image = (np.array(img[:, :, np.newaxis]))
        return img, msk

    # def label_preprocess(self,label):
    #     label = cv2.resize(label, (int(IMAGE_SIZE[1]/8), int(IMAGE_SIZE[0]/8)))
    #     label_pixel=self.ImageBinarization(label)
    #     label=label.sum()
    #     if label>0:
    #         label=1
    #     return  label_pixel, label
    #
    # def ImageBinarization(self, img, threshold=1):
    #     img = np.array(img)
    #     image = np.where(img > threshold, 1, 0)
    #     return image

    # def label2int(self,label):  # label shape (num,len)
    #     # seq_len=[]
    #     target_input = np.ones((MAX_LEN_WORD), dtype=np.float32) + 2  # 初始化为全为PAD
    #     target_out = np.ones(( MAX_LEN_WORD), dtype=np.float32) + 2  # 初始化为全为PAD
    #     target_input[0] = 0  # 第一个为GO
    #     for j in range(len(label)):
    #         target_input[j + 1] = VOCAB[label[j]]
    #         target_out[j] = VOCAB[label[j]]
    #         target_out[len(label)] = 1
    #     return target_input, target_out

    # def int2label(self,decode_label):
    #     label = []
    #     for i in range(decode_label.shape[0]):
    #         temp = ''
    #         for j in range(decode_label.shape[1]):
    #             if VOC_IND[decode_label[i][j]] == '<EOS>':
    #                 break
    #             elif decode_label[i][j] == 3:
    #                 continue
    #             else:
    #                 temp += VOC_IND[decode_label[i][j]]
    #         label.append(temp)
    #     return label

    # def get_label(self,f):
    #     return f.split('.')[-2].split('_')[1]
