#!/usr/bin/python
# -*- coding: utf-8 -*-
# set random seed for consistent predict result
from numpy.random import seed
seed(2018)
from tensorflow import set_random_seed
set_random_seed(2018)
import tensorflow as tf

class TRCNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 100      # 词向量维度
    seq_length = 100       # 序列长度
    num_classes = 3        # 类别数
    vocab_size = 20000       # 词汇表达小

    num_layers= 2           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元
    rnn = 'gru'             # lstm 或 gru

    dropout_keep_prob = 0.7 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    num_filters = 256        # 卷积核数目
    kernel_size = 3         # 卷积核尺寸

    batch_size = 20        # 每批训练大小 128
    num_epochs = 3        # 总迭代轮次

    print_per_batch = 4    # 每多少轮输出一次结果100
    save_per_batch = 10      # 每多少轮存入tensorboard


class TextRCNN(object):
    """文本分类，RNN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.rcnn()

    def rcnn(self):
        """rnn模型"""

        def lstm_cell():   # lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout(): # 为每一个rnn核后面加一个dropout层
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("rnn"):
            # 多层rnn网络
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)

            with tf.name_scope("cnn"):
                # CNN layer
                conv = tf.layers.conv1d(_outputs, self.config.num_filters, self.config.kernel_size, name='conv')
                print("conv'shape",conv)
                # global max pooling layer
                gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
                print("gmp'shape",gmp)


            with tf.name_scope("score"):
                # 全连接层，后面接dropout以及relu激活
                fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
                fc = tf.contrib.layers.dropout(fc, self.keep_prob)
                fc = tf.nn.relu(fc)
                print("fc'shape", fc)

                # 分类器
                self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
                self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
                self.predict_label = tf.nn.top_k(tf.nn.softmax(self.logits), k=1, sorted=True, name=None)

            with tf.name_scope("optimize"):
                # 损失函数，交叉熵
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
                self.loss = tf.reduce_mean(cross_entropy)
                # 优化器
                self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

            with tf.name_scope("accuracy"):
                # 准确率
                correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
