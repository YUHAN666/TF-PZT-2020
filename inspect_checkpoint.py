from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file(file_name="F:/ckpt/ckp-20",
                                      tensor_name = None, # ���ΪNone,��Ĭ��Ϊckpt������б���
                                      all_tensors = False, # bool �Ƿ��ӡ���е�tensor�������ӡ������tensor��ֵ��һ�㲻�Ƽ���������ΪFalse
                                      all_tensor_names = False) # bool �Ƿ��ӡ���е�tensor��name
