from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file(file_name="F:/ckpt/ckp-20",
                                      tensor_name = None, # 如果为None,则默认为ckpt里的所有变量
                                      all_tensors = False, # bool 是否打印所有的tensor，这里打印出的是tensor的值，一般不推荐这里设置为False
                                      all_tensor_names = False) # bool 是否打印所有的tensor的name
