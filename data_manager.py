import re
import os
import numpy as np
import cv2
from config import *
from random import shuffle
import tensorflow as tf
from my_func import rotate
from skimage import exposure
from config import IMAGE_MODE
from scalar2onehot import scalar2onehot



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

            # mask = scalar2onehot(mask)
            # image = (np.array(image[:, :, np.newaxis]))
            # mask = np.array(mask[:, :, np.newaxis])

            # if self.__Param["mode"] == "train_decision" or self.__Param["mode"] == "train_segmentation":
            #     aug_random = np.random.uniform()
            #     if aug_random > 0.6:
            #         expo = np.random.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3])
                    # image = exposure.adjust_gamma(image, expo)
            #                    angle=np.random.randint(1,72)*5
            #                    image=rotate(image,angle)
            #########################################################################################
            # if IMAGE_MODE == 0:  # expanding dimension is needed only in mono mode
            if self.__Param["mode"] == "train_decision" or self.__Param["mode"] == "train_segmentation":
                aug_random = np.random.uniform()
                if aug_random > 0.7:

                    # adjust_gamma
                    if np.random.uniform() > 0.7:
                        expo = np.random.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3])
                        image = exposure.adjust_gamma(image, expo)

                    # flip
                    if np.random.uniform() > 0.7:
                        aug_seed = np.random.randint(-1, 2)
                        image = cv2.flip(image, aug_seed)
                        mask = cv2.flip(mask, aug_seed)

                    # # rotate
                    # if np.random.uniform() > 0.7:
                    #     angle = np.random.randint(-5, 5)
                    #     image = rotate(image, angle)
                    #     mask = rotate(mask, angle)

                    # GassianBlur
                    if np.random.uniform() > 0.7:
                        image = cv2.GaussianBlur(image, (5, 5), 0)

                    # # shift
                    # if np.random.uniform() > 0.7:
                    #     dx = np.random.randint(-5, 5)  # width*5%
                    #     dy = np.random.randint(-5, 5)  # Height*10%
                    #     rows, cols = image.shape[:2]
                    #     M = np.float32([[1, 0, dx], [0, 1, dy]])  # (x,y) -> (dx,dy)
                    #     image = cv2.warpAffine(image, M, (cols, rows))
                    #     mask = cv2.warpAffine(mask, M, (cols, rows))

            # mask = self.ImageBinarization(mask)
            if IMAGE_MODE == 0:  # expanding dimention is needed only in mono mode
                image = image[:,:,0]
                image = (np.array(image[:, :, np.newaxis]))
            mask = (np.array(mask[:, :, np.newaxis]))  # needed only when label is mono
            #     image = (np.array(image[:, :, np.newaxis]))
           ###########################################################




            yield image, mask, label, image_path

    def read_data(self, image_path, mask_path):

        pad_x = np.random.randint(0, 100, 2)
        pad_y = np.random.randint(0, 200, 2)

        # if IMAGE_MODE == 0:
        #     img = cv2.imread(image_path, 0)  # /255.#read the gray image
        #     # img = np.pad(img, ((pad_y[0], pad_y[1]), (pad_x[0], pad_x[1]), (0, 0)), 'constant')
        #
        #     img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
        #     img = np.array(img[:, :, np.newaxis])
        #
        # else:
        img = cv2.imread(image_path, 1)  # /255.#read the gray image
        # img = np.pad(img, ((pad_y[0], pad_y[1]), (pad_x[0], pad_x[1]), (0, 0)), 'constant')

        img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))

        try:
            msk = cv2.imread(mask_path, 0)  # /255.#read the gray image
            # msk = np.pad(msk, ((pad_y[0], pad_y[1]), (pad_x[0], pad_x[1])), 'constant')
            msk = cv2.resize(msk, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
            _, msk = cv2.threshold(msk, 0, 255, cv2.THRESH_BINARY)
            # msk = np.array(msk[:, :, np.newaxis])
        except:
            msk = np.zeros((img.shape[0], img.shape[1]))

        # image augmentation by randomly flip

        # if self.__Param["mode"] == 'train_segmentation' or self.__Param["mode"] == 'train_decision':
        #     p = np.random.random(1)
        #     if p < 0.25:
        #         img = cv2.flip(img, 1)
        #         msk = cv2.flip(msk, 1)
        #     elif 0.25 < p < 0.5:
        #         img = cv2.flip(img, 0)
        #         msk = cv2.flip(msk, 0)
        #     elif 0.5 < p < 0.75:
        #         img = cv2.flip(img, -1)
        #         msk = cv2.flip(msk, -1)
        #     else:
        #         pass
        #
        #     # p2 = np.random.random(1)
        #     # if p2 < 0.25:
        #     #     img = cv2.GaussianBlur(img, (7, 7), 1.5)
        #     # elif 0.25 < p2 < 0.5:
        #     #     img = cv2.GaussianBlur(img, (5, 5), 0)
        #     # elif 0.5 < p2 < 0.75:
        #     #     img = cv2.GaussianBlur(img, (9, 9), 1.5)
        #     # else:
        #     #     pass
        #
        # # img = img.swapaxes(0, 1)
        # # image = (np.array(img[:, :, np.newaxis]))
        # if IMAGE_MODE == 0:
        #     img = img[:, :, 0]
        #     img = np.array(img[:, :, np.newaxis])
        #
        # if len(msk.shape) == 2:
        #     msk = np.array(msk[:, :, np.newaxis])

        return img, msk

    def ImageBinarization(self,img, threshold=1):
        img = np.array(img)
        image = np.where(img > threshold, 1, 0)
        return image

    def rotate(self, image, angle, center=None, scale=1.0):  # 1
        (h, w) = image.shape[:2]  # 2
        if center is None:  # 3
            center = (w // 2, h // 2)  # 4

        M = cv2.getRotationMatrix2D(center, angle, scale)  # 5

        rotated = cv2.warpAffine(image, M, (w, h))  # 6
        return rotated  # 7

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
