import os
import argparse
from agent import Agent
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 默认参数(Notes : some params are disable )
DefaultParam = {
    "mode": "testing",  # 模式  {"training","testing" }
    "epochs_num": 20,
    "batch_size": 4,
    "batch_size_inference": 2,
    "learn_rate": 0.0003,
    "momentum": 0.9,                 # BN momentum
    "data_dir": "E:/CODES/FAST-SCNN/DATA/surplus7/",  # 数据路径
    # "data_dir": "D:/DL/data_set/test/",
    "checkPoint_dir": "checkpoint",  # 模型保存路径
    "Log_dir": "Log",  # 日志打印路径
    "valid_ratio": 0,  # 数据集中用来验证的比例  (disable)
    "valid_frequency": 1,  # 每几个周期验证一次  (disable)
    "save_frequency": 1,  # 几个周期保存一次模型
    "max_to_keep": 10,  # 最多保存几个模型
    "pb_Mode_dir": "pbMode",
    "b_restore": True,  # 导入参数
    "b_saveNG": True,  # 测试时是否保存错误的样本  (disable)
    "tensorboard_logdir": "./tensorboard",
    "backbone": "ghostnet",
    "neck": "bifpn"
}


def parse_arguments():
    """
        Parse the command line arguments of the program.a
    """

    parser = argparse.ArgumentParser(description='Train or test the CRNN model.')

    parser.add_argument(
        "--train_segmentation",
        action="store_true",
        help="Define if we wanna to train the segmentation part",
        default=True
    )
    parser.add_argument(
        "--train_decision",
        action="store_true",
        help="Define if we wanna to train the decision part",
        default=False
    )

    parser.add_argument(
        "--pb",
        action="store_true",
        help="Define if we wanna to get the pbmodel",
        default=False
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Define if we wanna test the model",
        default=False
    )
    parser.add_argument(
        "--visualization",
        action="store_true",
        help="Define if we wanna visualize the segmentation output",
        default=False
    )
    parser.add_argument(
        "--anew",
        action="store_true",
        help="Define if we try to start from scratch  instead of  loading a checkpoint file from the save folder",
        default=False
    )
    parser.add_argument(
        "-vr",
        "--valid_ratio",
        type=float,
        nargs="?",
        help="How the data will be split between training and testing",
        default=DefaultParam["valid_ratio"]
    )
    parser.add_argument(
        "-ckpt",
        "--checkPoint_dir",
        type=str,
        nargs="?",
        help="The path where the pretrained model can be found or where the model will be saved",
        default=DefaultParam["checkPoint_dir"]
    )
    parser.add_argument(
        "-dd",
        "--data_dir",
        type=str,
        nargs="?",
        help="The path to the file containing the examples (training samples)",
        default=DefaultParam["data_dir"]
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        nargs="?",
        help="Size of a batch",
        default=DefaultParam["batch_size"]
    )
    parser.add_argument(
        "-en",
        "--epochs_num",
        type=int,
        nargs="?",
        help="How many iteration in training",
        default=DefaultParam["epochs_num"]
    )

    return parser.parse_args()


def main():

    # 导入默认参数
    param = DefaultParam
    # 从命令行更新参数
    args = parse_arguments()
    # if not args.train and not args.test and not args.pb:
    #     print("If we are not training, and not testing, what is the point?")
    if args.train_segmentation:
        param["mode"] = "train_segmentation"
    if args.train_decision:
        param["mode"] = "train_decision"
    if args.test:
        param["mode"] = "testing"
    if args.pb:
        param["mode"] = "savePb"
    if args.visualization:
        param["mode"] = 'visualization'
    if args.anew:
        param["b_restore"] = False
    param["data_dir"] = args.data_dir
    param["valid_ratio"] = args.valid_ratio
    param["batch_size"] = args.batch_size
    param["epochs_num"] = args.epochs_num
    param["checkPoint_dir"] = args.checkPoint_dir

    agent = Agent(param)
    agent.run()


if __name__ == '__main__':
    main()
