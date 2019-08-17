#!/usr/bin/python
# -*- coding: utf-8 -*-

# set random seed for consistent predict result
from numpy.random import seed
seed(2018)
from tensorflow import set_random_seed
set_random_seed(2018)


import tensorflow as tf

class vdcnnConfig(object):
    """配置参数"""

    embedding_dim = 100      # 词向量维度
    seq_length = 100        # 序列长度
    num_classes = 3        # 类别数 3008
    #num_filters = 256        # 卷积核数目
    #kernel_size = 3         # 卷积核尺寸
    vocab_size = 20000       # 词汇表达小

    #hidden_dim = 100        # 全连接层神经元

    dropout_keep_prob = 0.7 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 20         # 每批训练大小 128
    num_epochs = 3        # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard

    depth = 9
    use_he_uniform = True
    optional_shortcut = False
    downsampling_type = 'maxpool'
    num_quantized_chars = 69
    num_batches_per_epoch = 1000


import math

# weights initializers
he_normal = tf.keras.initializers.he_normal()
regularizer = tf.contrib.layers.l2_regularizer(1e-4)

def Convolutional_Block(inputs, shortcut, num_filters, name, is_training):
    print("-"*20)
    print("Convolutional Block", str(num_filters), name)
    print("-"*20)
    with tf.variable_scope("conv_block_" + str(num_filters) + "_" + name):
        for i in range(2):
            with tf.variable_scope("conv1d_%s" % str(i)):
                filter_shape = [3, inputs.get_shape()[2], num_filters]
                W = tf.get_variable(name='W', shape=filter_shape,
                    initializer=he_normal,
                    regularizer=regularizer)
                inputs = tf.nn.conv1d(inputs, W, stride=1, padding="SAME")
                inputs = tf.layers.batch_normalization(inputs=inputs, momentum=0.997, epsilon=1e-5,
                                                center=True, scale=True, training=is_training)
                inputs = tf.nn.relu(inputs)
                print("Conv1D:", inputs.get_shape())
    print("-"*20)
    if shortcut is not None:
        print("-"*5)
        print("Optional Shortcut:", shortcut.get_shape())
        print("-"*5)
        return inputs + shortcut
    return inputs

# Three types of downsampling methods described by paper
def downsampling(inputs, downsampling_type, name, optional_shortcut=False, shortcut=None):
    # k-maxpooling
    if downsampling_type=='k-maxpool':
        k = math.ceil(int(inputs.get_shape()[1]) / 2)
        pool = tf.nn.top_k(tf.transpose(inputs, [0,2,1]), k=k, name=name, sorted=False)[0]
        pool = tf.transpose(pool, [0,2,1])
    # Linear
    elif downsampling_type=='linear':
        pool = tf.layers.conv1d(inputs=inputs, filters=inputs.get_shape()[2], kernel_size=3,
                            strides=2, padding='same', use_bias=False)
    # Maxpooling
    else:
        pool = tf.layers.max_pooling1d(inputs=inputs, pool_size=3, strides=2, padding='same', name=name)
    if optional_shortcut:
        shortcut = tf.layers.conv1d(inputs=shortcut, filters=shortcut.get_shape()[2], kernel_size=1,
                            strides=2, padding='same', use_bias=False)
        print("-"*5)
        print("Optional Shortcut:", shortcut.get_shape())
        print("-"*5)
        pool += shortcut
    pool = fixed_padding(inputs=pool)
    return tf.layers.conv1d(inputs=pool, filters=pool.get_shape()[2]*2, kernel_size=1,
                            strides=1, padding='valid', use_bias=False)

def fixed_padding(inputs, kernel_size=3):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [0, 0]])
    return padded_inputs


class VDCNN(object):
    """文本分类，CNN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.is_training = tf.placeholder(tf.bool)

        self.vdcnn()

    def vdcnn(self):
        """VDCNN模型"""
        # 词向量映射


        # Depth to No. Layers
        if self.config.depth == 9:
            num_layers = [2,2,2,2]
        elif self.config.depth == 17:
            num_layers = [4,4,4,4]
        elif self.config.depth == 29:
            num_layers = [10,10,4,4]
        elif self.config.depth == 49:
            num_layers = [16,16,10,6]
        else:
            raise ValueError('depth=%g is a not a valid setting!' % depth)



        # Embedding Lookup 16
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if self.config.use_he_uniform:
                self.embedding_W = tf.get_variable(name='lookup_W', shape=[self.config.vocab_size, self.config.embedding_dim], initializer=tf.keras.initializers.he_uniform())
            else:
                self.embedding_W = tf.Variable(tf.random_uniform([self.config.num_quantized_chars, self.config.embedding_dim], -1.0, 1.0),name="embedding_W")
            self.embedded_characters = tf.nn.embedding_lookup(self.embedding_W, self.input_x)
            print("-"*20)
            print("Embedded Lookup:", self.embedded_characters.get_shape())
            print("-"*20)

        self.layers = []

        # Temp(First) Conv Layer
        with tf.variable_scope("temp_conv") as scope:
            filter_shape = [3, self.config.embedding_dim, 64]
            W = tf.get_variable(name='W_1', shape=filter_shape,
                initializer=he_normal,
                regularizer=regularizer)
            inputs = tf.nn.conv1d(self.embedded_characters, W, stride=1, padding="SAME")
            #inputs = tf.nn.relu(inputs)
        print("Temp Conv", inputs.get_shape())
        self.layers.append(inputs)

        # Conv Block 64
        for i in range(num_layers[0]):
            if i < num_layers[0] - 1 and self.config.optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=64, is_training=self.is_training, name=str(i+1))
            self.layers.append(conv_block)
        pool1 = downsampling(self.layers[-1], downsampling_type=self.config.downsampling_type, name='pool1', optional_shortcut=self.config.optional_shortcut, shortcut=self.layers[-2])
        self.layers.append(pool1)
        print("Pooling:", pool1.get_shape())

        # Conv Block 128
        for i in range(num_layers[1]):
            if i < num_layers[1] - 1 and self.config.optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=128, is_training=self.is_training, name=str(i+1))
            self.layers.append(conv_block)
        pool2 = downsampling(self.layers[-1], downsampling_type=self.config.downsampling_type, name='pool2', optional_shortcut=self.config.optional_shortcut, shortcut=self.layers[-2])
        self.layers.append(pool2)
        print("Pooling:", pool2.get_shape())

        # Conv Block 256
        for i in range(num_layers[2]):
            if i < num_layers[2] - 1 and self.config.optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=256, is_training=self.is_training, name=str(i+1))
            self.layers.append(conv_block)
        pool3 = downsampling(self.layers[-1], downsampling_type=self.config.downsampling_type, name='pool3', optional_shortcut=self.config.optional_shortcut, shortcut=self.layers[-2])
        self.layers.append(pool3)
        print("Pooling:", pool3.get_shape())

        # Conv Block 512
        for i in range(num_layers[3]):
            if i < num_layers[3] - 1 and self.config.optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=512, is_training=self.is_training, name=str(i+1))
            self.layers.append(conv_block)

        # Extract 8 most features as mentioned in paper
        self.k_pooled = tf.nn.top_k(tf.transpose(self.layers[-1], [0,2,1]), k=8, name='k_pool', sorted=False)[0]
        print("8-maxpooling:", self.k_pooled.get_shape())
        self.flatten = tf.reshape(self.k_pooled, (-1, 512*8))

        # fc1
        with tf.variable_scope('fc1'):
            w = tf.get_variable('w', [self.flatten.get_shape()[1], 2048], initializer=he_normal,
                regularizer=regularizer)
            b = tf.get_variable('b', [2048], initializer=tf.constant_initializer(1.0))
            out = tf.matmul(self.flatten, w) + b
            self.fc1 = tf.nn.relu(out)

        # fc2
        with tf.variable_scope('fc2'):
            w = tf.get_variable('w', [self.fc1.get_shape()[1], 2048], initializer=he_normal,
                regularizer=regularizer)
            b = tf.get_variable('b', [2048], initializer=tf.constant_initializer(1.0))
            out = tf.matmul(self.fc1, w) + b
            self.fc2 = tf.nn.relu(out)

        # fc3
        with tf.variable_scope('fc3'):
            w = tf.get_variable('w', [self.fc2.get_shape()[1], self.config.num_classes], initializer=he_normal,
                regularizer=regularizer)
            b = tf.get_variable('b', [self.config.num_classes], initializer=tf.constant_initializer(1.0))
            self.fc3 = tf.matmul(self.fc2, w) + b

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            self.y_pred_cls = tf.argmax(self.fc3, 1, name="predictions")
            self.predict_label = tf.nn.top_k(tf.nn.softmax(self.fc3), k=1, sorted=True, name=None)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc3, labels=self.input_y)
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = tf.reduce_mean(losses) + sum(regularization_losses)
            # 优化器
            #self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        # Optimizer and LR Decay
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step = tf.Variable(0, name="global_step", trainable=False)
            learning_rate = tf.train.exponential_decay(self.config.learning_rate, global_step,
                                                       self.config.num_epochs * self.config.num_batches_per_epoch, 0.95, staircase=True)
            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 7.0)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.y_pred_cls, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")



