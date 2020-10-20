from tensorflow.python.platform import gfile
import tensorflow as tf
import os
import cv2
from config import *
from timeit import default_timer as timer
from iouEval import iouEval
from matplotlib import pyplot as plt

pb_file_path = './pbMode/frozen_inference_graph_fuse.pb'
train_dir = 'E:/CODES/FAST-SCNN/DATA/1pzt/'
val_dir = 'E:/TensorRT-7.0.0.11.Windows10.x86_64.cuda-10.0.cudnn7.6/TensorRT-7.0.0.11/data/mnist/images/train_image/'
test_dir = 'E:/CODES/FAST-SCNN/DATA/1pzt/test_image/'

def listData(data_dir):

	image_dirs = [x[2] for x in os.walk(data_dir)][0]
	images_train = []

	for i in range(len(image_dirs)):
		image_dir = image_dirs[i]
		image_path = data_dir + image_dir
		images_train.append(image_path)

	return images_train


class DataManager(object):

	def __init__(self, imageList, mask_list):

		self.image_list = imageList
		self.mask_list = mask_list
		self.data_size = len(imageList)
		self.next_batch = self.get_next()
		self.number_batch = int(np.floor(len(self.image_list)))

	def get_next(self):
		dataset = tf.data.Dataset.from_generator(self.generator, (tf.float32, tf.float32, tf.float32, tf.string))
		dataset = dataset.batch(1)
		iterator = dataset.make_one_shot_iterator()
		out_batch = iterator.get_next()
		return out_batch

	def generator(self):

		for index in range(len(self.image_list)):
			image_path = self.image_list[index]
			mask_path = self.mask_list[index]
			if image_path.split('/')[-1].split('_')[0] == 'n':
				label = np.array([0.0])
			else:
				label = np.array([1.0])
			image = self.read_data(image_path)
			image = image / 255
			# image = np.array(image[:,:, np.newaxis])
			mask = self.read_data(mask_path)
			mask = mask[:,:,0]
			mask = np.array(mask[:, :, np.newaxis])
			mask = mask/255


			yield image, mask, label, image_path

	def read_data(self, image_path):

		img = cv2.imread(image_path, 1)  # /255.#read the gray image
		img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))

		return img


def test(sess, dataset):
	with sess.as_default():

		input_image = sess.graph.get_tensor_by_name('Image:0')
		decision_out = sess.graph.get_tensor_by_name('decision_out:0')
		mask_out = sess.graph.get_tensor_by_name('mask_out:0')
		DataManager = dataset
		num_step = 0.0
		accuracy = 0.0
		false_account = 0
		iouGen = iouEval()
		for batch in range(DataManager.number_batch):
			img_batch, mask_batch, label_batch, _ = sess.run(DataManager.next_batch)
			start = timer()
			b, decision = sess.run([mask_out, decision_out], feed_dict={input_image: img_batch})

			iouGen.addBatch(mask_batch, b)
			print(decision)
			end = timer()
			if decision > 0.5:
				false_account += 1
			# print(end-start)
			if decision[0][0] >= 0.5 and label_batch[0][0] == 1:
				step_accuracy = 1
			elif decision[0][0] < 0.5 and label_batch[0][0] == 0:
				step_accuracy = 1
			else:
				step_accuracy = 0
			accuracy = accuracy + step_accuracy
			num_step = num_step + 1

			cv2.imwrite('visualization/{}.png'.format(str(batch)), np.squeeze(b, axis=(0, -1))*255)
		iou = iouGen.getIoU()
		accuracy /= num_step
		print("accuracy: {}		iou:{}".format(accuracy, iou))

		return accuracy


image_list = listData(train_dir+'val_image/')
mask_list = listData(train_dir+'val_mask/')
# val_list = listData(val_dir)
# test_list = listData(test_dir)

DataManager_train = DataManager(image_list, mask_list)

# DataManager_val = DataManager(val_list)
# DataManager_test = DataManager(test_list)

sess = tf.Session()
with gfile.FastGFile(pb_file_path, 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	sess.graph.as_default()
	tf.import_graph_def(graph_def, name='')
sess.run(tf.global_variables_initializer())
mask_out = sess.graph.get_tensor_by_name('mask_out:0')
input_image = sess.graph.get_tensor_by_name('Image:0')
image = cv2.imread('C:/Users/yinha/source/repos/PAT_AOI/PAT_AOI/bin/x64/Debug/p_l_296.png', 0)
image = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
image = image/255
image = np.array(image[np.newaxis,:, :, np.newaxis])

image = np.concatenate([image,image], axis=0)
mask = sess.run([mask_out], feed_dict={input_image:image})
mask = np.squeeze(mask, axis=(0,-1))*255
plt.imshow(mask, cmap='gray')
plt.show()
cv2.waitKey()

# train_acc = test(sess, DataManager_train)
# val_acc = test(sess, DataManager_val)
# test_acc = test(sess, DataManager_test)
#
# print('train_accuracy = {},   val_accuracy = {},   test_accuracy = {}'.format(train_acc, val_acc, test_acc))



