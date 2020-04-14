import tensorflow as tf
import os

def save_PbModel(session, __pb_model_path):
	input_name = "Image"
	output_name = "decision_out"
	output_node_names = [input_name, output_name]
	output_graph_def = tf.graph_util.convert_variables_to_constants(session,
																	session.graph_def,
																	output_node_names)
	output_graph_def = tf.graph_util.remove_training_nodes(output_graph_def, protected_nodes=None)

	if not os.path.exists(__pb_model_path):
		os.makedirs(__pb_model_path)
	pbpath = os.path.join(__pb_model_path, 'tensorrt.pb')
	with tf.gfile.GFile(pbpath, mode='wb') as f:
		f.write(output_graph_def.SerializeToString())


sess = tf.Session()
with sess.as_default():
	x = tf.keras.layers.Input((3, 928, 320), name='Image')
	x = tf.keras.layers.Conv2D(128, (3, 3), (2, 2), padding='same', data_format='channels_first')(x)
	x = tf.layers.batch_normalization(x, axis=1)
	x = tf.nn.relu(x)
	x = tf.keras.layers.UpSampling2D((2, 2), data_format='channels_first')(x)
	x1 = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(x)
	x2 = tf.keras.layers.GlobalMaxPool2D(data_format='channels_first')(x)
	x = tf.concat([x1, x2], axis=1)
	x = tf.keras.layers.Dense(1)(x)

	x = tf.nn.sigmoid(x, name='decision_out')
sess.run(tf.global_variables_initializer())


pb_Model_dir = "pbMode"

save_PbModel(sess, pb_Model_dir)



