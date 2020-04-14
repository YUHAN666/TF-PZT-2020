import tensorflow as tf
import numpy as np
# import cv2
import os
# import time
from data_manager import DataManager
from model import Model
from config import IMAGE_SIZE, TRAIN_MODE_IN_TRAIN, TRAIN_MODE_IN_TEST, TRAIN_MODE_IN_VALID, IMAGE_MODE, TEST_RATIO
import utils
# from datetime import datetime
from tqdm import tqdm
from timeit import default_timer as timer


class Agent(object):
    def __init__(self, param):

        self.sess = tf.Session()
        self.__Param = param
        self.init_datasets()  # 初始化数据管理器
        self.model = Model(self.sess, self.__Param)  # 建立模型
        self.logger = utils.get_logger(param["Log_dir"])

    def run(self):
        if self.__Param["mode"] == "train_segmentation":
            self.train_segmentation()
        elif self.__Param["mode"] == "train_decision":
            self.train_decision()
        elif self.__Param["mode"] == "testing":
            self.test()
        elif self.__Param["mode"] == "savePb":
            self.savePb()
        elif self.__Param["mode"] == "visualization":
            self.visualization()
        else:
            print("got a unexpected mode ,please set the mode  'training', 'testing' or 'savePb' ")

    def init_datasets(self):
        if self.__Param["mode"] != "savePb":
            self.image_list_train, self.mask_list_train = self.listData_train(self.__Param["data_dir"]) 
            self.image_list_valid, self.mask_list_valid = self.listData_val(self.__Param["data_dir"])
            # self.image_list_test, self.mask_list_test = self.listData_test(self.__Param["data_dir"])
            
            self.DataManager_train = DataManager(self.image_list_train, self.mask_list_train, self.__Param)
            self.DataManager_valid = DataManager(self.image_list_valid, self.mask_list_valid, self.__Param, shuffle=False)
            # self.DataManager_test = DataManager(self.image_list_test, self.mask_list_test, self.__Param, shuffle=False)

    def train_segmentation(self):

        with self.sess.as_default():
            self.logger.info('start training segmentation net')

            print('Start training for {} epoches, {} steps per epoch'.format(self.__Param["epochs_num"],
                                                                             self.DataManager_train.number_batch))
            best_loss = 10000
            for i in range(self.model.step, self.__Param["epochs_num"] + self.model.step):
                print('Epoch {}:'.format(i))
                with tqdm(total=self.DataManager_train.number_batch) as pbar:
                    # epoch start
                    iter_loss = 0.0
                    num_step = 0.0
                    accuracy = 0.0
                    for batch in range(self.DataManager_train.number_batch):
                        # run_options = tf.RunOptions()
                        # run_metadata = tf.RunMetadata()
                        # batch start

                        # print(self.sess.run(
                        #     self.sess.graph.get_tensor_by_name('segmentation/MixnetBlock_0_bn1/moving_mean:0')))
                        # print(self.sess.run(
                        #     self.sess.graph.get_tensor_by_name('segmentation/MixnetBlock_0_bn1/moving_variance:0')))

                        img_batch, mask_batch, label_batch, _ = self.sess.run(self.DataManager_train.next_batch)

                        _, loss_value_batch = self.sess.run([self.model.optimize_segment,
                                                             self.model.segmentation_loss],
                                                             # self.model.merged],
                                                            feed_dict={self.model.Image: img_batch,
                                                                       self.model.mask: mask_batch,
                                                                       self.model.label: label_batch,
                                                                       self.model.is_training_seg: TRAIN_MODE_IN_TRAIN,
                                                                       self.model.is_training_dec: False})
                                                                       # options=run_options,
                                                                       # run_metadata=run_metadata)

                        # self.model.train_writer.add_run_metadata(run_metadata, 'step%03d' % batch)
                        # iter_loss = (iter_loss*(num_step)+ loss_value_batch)/(num_step+1)
                        iter_loss += loss_value_batch
                        num_step = num_step+1
                        pbar.update(1)
                        # self.model.train_writer.add_summary(summary, batch)
                pbar.close()
                iter_loss /= num_step

                self.logger.info('epoch:[{}] ,train_mode, loss: {}, accuracy: {}'
                                 .format(self.model.step, iter_loss, accuracy))
                # 验证
                self.model.step += 1
                # if i % self.__Param["valid_frequency"] == 0 and i>0:
                val_loss = self.valid_segmentation()
                print('train_loss:{},   val_loss:{}'
                      .format(iter_loss, val_loss))

                # 保存模型
                if i % self.__Param["save_frequency"] == 0 or i == self.__Param["epochs_num"] + self.model.step - 1:
                    # if val_loss < best_loss:
                        # best_loss = val_loss
                        # print('reduce loss to {}, saving model at epoch:{}'.format(val_loss, i))
                    self.model.save()
            # self.model.train_writer.close()

    def train_decision(self):

        with self.sess.as_default():
            self.logger.info('start training decision net')

            print('Start training for {} epoches, {} steps per epoch'.format(self.__Param["epochs_num"],
                                                                             self.DataManager_train.number_batch))
            best_loss = 10000
            for i in range(self.model.step, self.__Param["epochs_num"] + self.model.step):
                print('Epoch {}:'.format(i))
                with tqdm(total=self.DataManager_train.number_batch) as pbar:
                    # epoch start
                    iter_loss = 0.0
                    num_step = 0.0
                    accuracy = 0.0
                    for batch in range(self.DataManager_train.number_batch):
                        # batch start
                        img_batch, mask_batch, label_batch, _ = self.sess.run(self.DataManager_train.next_batch)

                        loss_value_batch = 0

                        _, loss_value_batch, decision_out = self.sess.run([self.model.optimize_decision,
                                                                           self.model.decision_loss,
                                                                           self.model.decison_out],
                                                                           feed_dict={self.model.Image: img_batch,
                                                                                      self.model.mask: mask_batch,
                                                                                      self.model.label: label_batch,
                                                                                      self.model.is_training_seg: False,
                                                                                      self.model.is_training_dec: TRAIN_MODE_IN_TRAIN})
                        # self.model.keep_prob: 0.9}
                        if decision_out[0][0] >= 0.5 and label_batch[0][0] == 1:
                            step_accuracy = 1
                        elif decision_out[0][0] < 0.5 and label_batch[0][0] == 0:
                            step_accuracy = 1
                        else:
                            step_accuracy = 0
                        # iter_loss = (iter_loss*(num_step)+ loss_value_batch)/(num_step+1)
                        iter_loss += loss_value_batch
                        num_step = num_step + 1
                        accuracy = accuracy + step_accuracy
                        pbar.update(1)
                pbar.clear()
                pbar.close()

                # 保存PB
                # frozen_graph = self.model.freeze_session()
                # from tensorflow.python.framework import graph_io
                # graph_io.write_graph(frozen_graph, 'pbMode', 'pb_model_name.pb', as_text=False)

                accuracy = accuracy / num_step
                iter_loss /= num_step

                self.logger.info('epoch:[{}] ,train_mode, loss: {}, accuracy: {}'
                                 .format(self.model.step, iter_loss, accuracy))
                # 验证
                self.model.step += 1
                # if i % self.__Param["valid_frequency"] == 0 and i>0:
                val_loss, val_acc = self.validation(self.DataManager_valid)
                print('train_loss:{},   train_acc:{},   val_loss:{},    val_acc:{}'
                      .format(iter_loss, accuracy, val_loss, val_acc))

                # 保存模型
                if i % self.__Param["save_frequency"] == 0 or i == self.__Param["epochs_num"] + self.model.step - 1:
                    # if val_loss < best_loss:
                    #     best_loss = val_loss
                    #     print('reduce loss to {}, saving model at epoch:{}'.format(val_loss, i))
                    self.model.save()

    def valid_segmentation(self):

        with self.sess.as_default():
            self.logger.info('start validing segmentation')
            print('start validing segmentation')
            DataManager = self.DataManager_valid
            total_loss = 0.0
            num_step = 0.0
            accuracy = 0.0   
            
            for batch in range(DataManager.number_batch):
                img_batch, mask_batch, label_batch, _ = self.sess.run(DataManager.next_batch)

                total_loss_value_batch = self.sess.run(self.model.segmentation_loss,
                                                         feed_dict={self.model.Image: img_batch,
                                                                    self.model.mask: mask_batch,
                                                                    self.model.label: label_batch,
                                                                    self.model.is_training_seg: TRAIN_MODE_IN_VALID,
                                                                    self.model.is_training_dec: TRAIN_MODE_IN_VALID})
            # self.visualization(img_batch, label_pixel_batch,mask_batch, file_name_batch,save_dir=visualization_dir)
                
                # total_loss = (total_loss*(num_step)+ total_loss_value_batch)/(num_step+1)
                num_step = num_step+1
                total_loss += total_loss_value_batch
            total_loss /= num_step
            self.logger.info(" validation loss = {}".format(total_loss))
            return total_loss
            # self.logger.info("the visualization saved in {}".format(visualization_dir))

    def validation(self, Dataset):

        with self.sess.as_default():
            DataManager = Dataset
            total_loss = 0.0
            num_step = 0.0
            step_accuracy = 0
            accuracy = 0.0

            for batch in range(DataManager.number_batch):
                img_batch, mask_batch, label_batch, _ = self.sess.run(DataManager.next_batch)

                total_loss_value_batch, decision_out = self.sess.run([self.model.decision_loss,
                                                                      self.model.decison_out],
                                                                     feed_dict={self.model.Image: img_batch,
                                                                                self.model.mask: mask_batch,
                                                                                self.model.label: label_batch,
                                                                                self.model.is_training_seg: TRAIN_MODE_IN_VALID,
                                                                                self.model.is_training_dec: TRAIN_MODE_IN_VALID})
                if decision_out[0][0] >= 0.5 and label_batch[0][0] == 1:
                    step_accuracy = 1
                elif decision_out[0][0] < 0.5 and label_batch[0][0] == 0:
                    step_accuracy = 1
                else:
                    step_accuracy = 0
                accuracy = accuracy + step_accuracy

                # total_loss = (total_loss*(num_step)+ total_loss_value_batch)/(num_step+1)
                num_step = num_step + 1
                total_loss += total_loss_value_batch
            total_loss /= num_step
            accuracy /= num_step
            self.logger.info(" validation loss = {}".format(total_loss))
            return total_loss, accuracy
            # self.logger.info("the visualization saved in {}".format(visualization_dir))

    def test(self):
        with self.sess.as_default():
            self.logger.info('start testing')
            print('start testing')
            # train_loss, train_acc = self.validation(self.DataManager_train)
            # print('train_loss={},   train_accuracy={}'.format(train_loss, train_acc))

            val_loss, val_acc = self.validation(self.DataManager_valid)
            print('val_loss={},   val_accuracy={}'.format(val_loss, val_acc))

            test_loss, test_acc = self.validation(self.DataManager_test)
            print('test_loss={},   test_accuracy={}'.format(test_loss, test_acc))

    def visualization(self, save_dir="./visualization"):
        # anew a floder to save visualization
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        DataManager = self.DataManager_valid
        for batch in range(DataManager.number_batch):

            img_batch, mask_in, _, path = self.sess.run(DataManager.next_batch)
            mask_out = self.sess.run(self.model.mask_out, feed_dict={self.model.Image: img_batch})

            start = timer()
            _ = self.sess.run(self.model.decison_out, feed_dict={self.model.Image: img_batch})
            end = timer()
            print(end - start)
            start = timer()
            _ = self.sess.run(self.model.feature_list, feed_dict={self.model.Image: img_batch})
            end = timer()
            print(end - start)

            for i in range(self.__Param["batch_size_inference"]):
                filename = str(path[i]).split('/')[-1].split('\\')[0].split("'")[0]
                print(filename)
                if IMAGE_MODE == 0:
                    image = np.array(img_batch[i]).squeeze()
                else:
                    image = np.mean(img_batch[i], axis=2)
                mask = np.array(mask_out[i]).squeeze(2)*255
                mask_in = np.array(mask_in[i]).squeeze()*255
                image = image*255
                # label_pixel = np.array(label_pixel_batch[i]).squeeze(2)*255
                img_visual = utils.concatImage([image, mask_in, mask])
                visualization_path = os.path.join(save_dir, filename)
                img_visual.save(visualization_path)
    
    def savePb(self):
        self.model.save_PbModel()

    def listData_train(self, data_dir, test_ratio=TEST_RATIO):

        image_dirs = [x[2] for x in os.walk(data_dir+'train_image/')][0]
        images_train = []
        masks_train = []

        for i in range(len(image_dirs)):
            image_dir = image_dirs[i]
            # 训练数据
            image_path = data_dir + 'train_image/' + image_dir
            mask_path = data_dir + 'train_mask/' + image_dir
            images_train.append(image_path)
            masks_train.append(mask_path)
        return images_train, masks_train

    def listData_val(self, data_dir, test_ratio=TEST_RATIO):

        image_dirs = [x[2] for x in os.walk(data_dir+'val_image/')][0]
        images_val = []
        masks_val = []

        for i in range(len(image_dirs)):
            image_dir = image_dirs[i]
            # 训练数据
            image_path = data_dir + 'val_image/' + image_dir
            mask_path = data_dir + 'val_mask/' + image_dir
            images_val.append(image_path)
            masks_val.append(mask_path)
        return images_val, masks_val

    def listData_test(self, data_dir, test_ratio=TEST_RATIO):

        image_dirs = [x[2] for x in os.walk(data_dir+'test_image/')][0]
        images_val = []
        masks_val = []

        for i in range(len(image_dirs)):
            image_dir = image_dirs[i]
            # 训练数据
            image_path = data_dir + 'test_image/' + image_dir
            mask_path = data_dir + 'test_mask/' + image_dir
            images_val.append(image_path)
            masks_val.append(mask_path)
        return images_val, masks_val



