from tensorflow.python.platform import gfile
import tensorflow as tf
import os
import cv2
from config import *
from timeit import default_timer as timer

pb_file_path = './pbMode/tensorrt.pb'
train_dir = 'F:/TensorRT-7.0.0.11.Windows10.x86_64.cuda-10.0.cudnn7.6/TensorRT-7.0.0.11/data/mnist/images/valid_image/'
val_dir = 'F:/TensorRT-7.0.0.11.Windows10.x86_64.cuda-10.0.cudnn7.6/TensorRT-7.0.0.11/data/mnist/images/train_image/'
test_dir = 'F:/CODES/FAST-SCNN/DATA/1pzt/test_image/'

def listData(data_dir):

	image_dirs = [x[2] for x in os.walk(data_dir)][0]
	images_train = []

	for i in range(len(image_dirs)):
		image_dir = image_dirs[i]
		image_path = data_dir + image_dir
		images_train.append(image_path)

	return images_train


class DataManager(object):

	def __init__(self, imageList):

		self.image_list = imageList
		self.data_size = len(imageList)
		self.next_batch = self.get_next()
		self.number_batch = int(np.floor(len(self.image_list)))

	def get_next(self):
		dataset = tf.data.Dataset.from_generator(self.generator, (tf.float32, tf.float32, tf.string))
		dataset = dataset.batch(1)
		iterator = dataset.make_one_shot_iterator()
		out_batch = iterator.get_next()
		return out_batch

	def generator(self):

		for index in range(len(self.image_list)):
			image_path = self.image_list[index]
			if image_path.split('/')[-1].split('_')[0] == 'n':
				label = np.array([0.0])
			else:
				label = np.array([1.0])
			image = self.read_data(image_path)
			image = image / 255

			yield image, label, image_path

	def read_data(self, image_path):

		img = cv2.imread(image_path, 0)  # /255.#read the gray image
		img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
		img = np.array(img[np.newaxis,:,:])

		return img


def test(sess, dataset):
	with sess.as_default():

		input_image = sess.graph.get_tensor_by_name('Image:0')
		decision_out = sess.graph.get_tensor_by_name('decision_out:0')
		DataManager = dataset
		num_step = 0.0
		accuracy = 0.0
		false_account = 0
		for batch in range(DataManager.number_batch):
			img_batch, label_batch, _ = sess.run(DataManager.next_batch)
			start = timer()
			decision = sess.run(decision_out, feed_dict={input_image: img_batch})
			# print(decision)
			end = timer()
			if decision > 0.5:
				false_account += 1
			print(end-start)
			if decision[0][0] >= 0.5 and label_batch[0][0] == 1:
				step_accuracy = 1
			elif decision[0][0] < 0.5 and label_batch[0][0] == 0:
				step_accuracy = 1
			else:
				step_accuracy = 0
			accuracy = accuracy + step_accuracy
			num_step = num_step + 1
		accuracy /= num_step
		print("accuracy: {}".format(accuracy))

		return accuracy


train_list = listData(train_dir)
val_list = listData(val_dir)
test_list = listData(test_dir)

DataManager_train = DataManager(train_list)
DataManager_val = DataManager(val_list)
DataManager_test = DataManager(test_list)

sess = tf.Session()
with gfile.FastGFile(pb_file_path, 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	sess.graph.as_default()
	tf.import_graph_def(graph_def, name='')
sess.run(tf.global_variables_initializer())

train_acc = test(sess, DataManager_train)
# val_acc = test(sess, DataManager_val)
# test_acc = test(sess, DataManager_test)
#
# print('train_accuracy = {},   val_accuracy = {},   test_accuracy = {}'.format(train_acc, val_acc, test_acc))



