# author:sheng.Gw
# -*- coding: utf-8 -*-
# @Date :  2018/9/3

import tensorflow as tf
from tensorflow.python.ops import array_ops


print(array_ops.concat([[1,2,3],[4,5,6]],0))  # 必须是2维的


