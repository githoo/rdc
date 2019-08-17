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
    seq_length = 64        # 序列长度
    num_classes = 3        # 类别数
    num_filters = 256        # 卷积核数目
    kernel_size = 2         # 卷积核尺寸
    vocab_size = 20000       # 词汇表达小

    hidden_dim = 200        # 全连接层神经元

    dropout_keep_prob = 0.7 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 20        # 每批训练大小 128
    num_epochs = 3      # 总迭代轮次

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
            print("embedding_inputs'shape",embedding_inputs)
        #
        # with tf.name_scope("cnn"):
        #     # CNN layer
        #     conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
        #     print("conv'shape",conv)
        #     # global max pooling layer
        #     gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
        #     print("gmp'shape",gmp)


        with tf.name_scope("cnn"):
            # CNN layer
            conv1 = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv1')
            print("conv1'shape",conv1)
            #conv1 = tf.layers.batch_normalization(inputs=conv1, momentum=0.997, epsilon=1e-5,
            #                                       center=True, scale=True, training=True)
            gmp1 = tf.reduce_max(conv1, reduction_indices=[1], name='gmp1')
            print("gmp1'shape", gmp1)
            conv2 = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size+1, name='conv2')
            print("conv2'shape", conv2)
            gmp2 = tf.reduce_max(conv2, reduction_indices=[1], name='gmp2')
            print("gmp2'shape", gmp2)
            conv3 = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size+2, name='conv3')
            print("conv3'shape", conv3)
            gmp3 = tf.reduce_max(conv3, reduction_indices=[1], name='gmp3')
            print("gmp3'shape", gmp3)
            #global max pooling layer

            conv4 = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size + 3,
                                     name='conv4')
            gmp4 = tf.reduce_max(conv4, reduction_indices=[1], name='gmp4')

            conv5 = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size + 4,
                                     name='conv5')
            gmp5 = tf.reduce_max(conv5, reduction_indices=[1], name='gmp5')

            conv6 = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size + 5,
                                     name='conv6')
            gmp6 = tf.reduce_max(conv6, reduction_indices=[1], name='gmp6')

            conv7 = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size + 6,
                                     name='conv7')
            gmp7 = tf.reduce_max(conv7, reduction_indices=[1], name='gmp7')

            gmp_tmp = tf.add(gmp1,gmp2)
            gmp_tmp = tf.add(gmp_tmp, gmp3)
            gmp_tmp = tf.add(gmp_tmp, gmp4)
            gmp_tmp = tf.add(gmp_tmp, gmp5)
            gmp_tmp = tf.add(gmp_tmp, gmp6)
            gmp = tf.add(gmp_tmp, gmp7)



            #print(gmp)
            gmp = tf.div(gmp,7)


            #gmp = tf.concat([gmp1,gmp2,gmp3],0)
            #print("=======")
            #print(gmp)
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
