# 一些宏定义
import numpy as np

# IMAGE_SIZE = [640, 320]  # [H,W]
IMAGE_SIZE = [1024, 256]  # glue
# IMAGE_SIZE = [256, 1280]    # side
# IMAGE_SIZE = [256, 1248]    # crop
# IMAGE_SIZE = [576, 320]     #     P08
# IMAGE_SIZE = [928, 320]     # PZT
# IMAGE_SIZE = [928, 320]  # [H,W]
BIN_SIZE = [1, 2, 4, 7]
ACTIVATION = 'relu'  #swish mish or relu

ATTENTION = 'se'  # se or cbam
DATA_FORMAT = 'channels_last'
DROP_OUT = False


TRAIN_MODE_IN_TRAIN = True
TRAIN_MODE_IN_VALID = False
TRAIN_MODE_IN_TEST = False


TEST_RATIO = 0.25
IMAGE_MODE = 0  # 0: mono, 1:color

CLASS_NUM = 1



label_dict = {"1": np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
              "2": np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
              "3": np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
              "4": np.array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
              "5": np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
              "6": np.array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
              "7": np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]),
              "8": np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]),
              "B": np.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]),
              "C": np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]),
              "D": np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
              "E": np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]),
              "F": np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]),
              "G": np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]),
              "empty": np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])}

label_list = ['1', '2', '3', '4', '5', '6', '7', '8', 'B', 'C', 'D', 'E', 'F', 'G', 'empty']
