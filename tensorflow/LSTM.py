import string
import tensorflow as tf
import zipfile
import numpy as np

class ModelConfig(object):
    def __init__(self):
        self.num_unrollings    = 10    # 每条数据的字符串长度
        self.batch_size        = 64    # 每一批数据的个数
        self.vocabulary_size   = len(string.ascii_lowercase) + 1
        self.summary_frequency = 100
        self.num_steps         = 7001
        self.num_nodes         = 64

config = ModelConfig()

class LoadData(object):
    def __init__(self,valid_size=1000):
        self.text           = self._read_data()
        self.valid_text     = self.text[:valid_size]
        self.train_text     = self.text[valid_size:]

    def _read_data(self,filename ):





class LSTM_Cell(object):

    def __init__(self,train_data,train_label,num_nodes=64):
        with tf.variable_scope("input",initializer=tf.truncated_normal_initializer(-0.1,0.1)) as input_layer:
            self.ix, self.im ,self.ib = self._generate_w_b()





