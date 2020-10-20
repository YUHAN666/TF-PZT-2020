"""
Resize images in given dir
"""

import pathlib
import cv2
from config import IMAGE_SIZE

dir = 'F:/TensorRT-7.0.0.11.Windows10.x86_64.cuda-10.0.cudnn7.6/TensorRT-7.0.0.11/data/mnist/images/val_image/'
path = pathlib.Path(dir).glob('*')
path = [str(i) for i in path]

for i in range(len(path)):
	image = cv2.imread(path[i], 0)
	image = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
	# cv2.imwrite(dir+str(i)+'.png', image)
	cv2.imwrite(path[i], image)