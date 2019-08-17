#!/usr/bin/python
# -*- coding: utf-8 -*-


# set random seed for consistent predict result
from numpy.random import seed
seed(2018)
from tensorflow import set_random_seed
set_random_seed(2018)

import tensorflow as tf

class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 100      # 词向量维度
    seq_length = 100        # 序列长度
    num_classes = 3        # 类别数
    num_filters = 256        # 卷积核数目
    kernel_size = 3         # 卷积核尺寸
    filter_sizes = [3,4,5]
    vocab_size = 20000       # 词汇表达小

    hidden_dim = 100        # 全连接层神经元

    dropout_keep_prob = 0.7 # dropout保留比例
    learning_rate = 1e-3    # 学习率
    l2_reg_lambda = 0.0

    batch_size = 20         # 每批训练大小 128
    num_epochs = 3        # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        #with tf.device('/cpu:0'):
        with tf.name_scope("cnn_embedding"):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            print("embedding'shape",embedding)
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
            print("embedding_inputs'shape",embedding_inputs.shape)
            self.embedding_inputs_expanded = tf.expand_dims(embedding_inputs, -1)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)


        #refrence https://github.com/jiegzhan/multi-class-text-classification-cnn/blob/master/text_cnn.py


        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.config.embedding_dim, 1, self.config.hidden_dim]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.config.hidden_dim]), name="b")
                conv = tf.nn.conv2d(
                    self.embedding_inputs_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.config.seq_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.config.hidden_dim * len(self.config.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.config.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.config.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
            self.predict_label = tf.nn.top_k(tf.nn.softmax(self.logits), k=1, sorted=True, name=None)

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.config.l2_reg_lambda * l2_loss
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)


        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
