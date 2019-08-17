#!/usr/bin/python
# -*- coding: utf-8 -*-
# set random seed for consistent predict result
from numpy.random import seed
seed(2018)
from tensorflow import set_random_seed
set_random_seed(2018)


import tensorflow as tf
from tcn import TemporalConvNet


class TCNConfig(object):
    """CNN配置参数"""

    embedding_dim = 100      # 词向量维度
    seq_length = 64        # 序列长度
    num_classes = 3        # 类别数
    num_filters = 256        # 卷积核数目
    kernel_size = 2         # 卷积核尺寸
    vocab_size = 20000       # 词汇表达小

    hidden_dim = 200        # 全连接层神经元

    dropout_keep_prob = 0.7 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 20         # 每批训练大小 128
    num_epochs = 3       # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard
    levels = 3


class TextTCN(object):
    """文本分类，TCN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        num_channels = [self.config.hidden_dim] * self.config.levels
        self.tcn = TemporalConvNet(num_channels, stride=1, kernel_size=self.config.kernel_size, dropout=self.config.dropout_keep_prob)
        self.tcn_init()

    def tcn_init(self):
        """TCN模型"""
        # 词向量映射
        #with tf.device('/cpu:0'):
        with tf.name_scope("cnn_embedding"):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            print("embedding'shape",embedding)
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
            print("embedding_inputs'shape",embedding_inputs)

        with tf.name_scope("cnn"):

            output = self.tcn(embedding_inputs)
            print("output'shape", output)
            gmp = tf.reduce_max(output, reduction_indices=[1], name='gmp')
            print("gmp'shape",gmp)


        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            #fc = tf.layers.dense(gmp, 3*self.config.hidden_dim, name='fc1')
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            print("fc'shape",fc)

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
