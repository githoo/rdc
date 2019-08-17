# -*- coding:utf-8 -*-
# set random seed for consistent predict result
from numpy.random import seed
seed(2018)
from tensorflow import set_random_seed
set_random_seed(2018)


import tensorflow as tf
import numpy as np


class TABRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 128  # 词向量维度
    seq_length = 128  # 序列长度
    num_classes = 3  # 类别数
    vocab_size = 500000  # 词汇表达小

    num_layers = 2  # 隐藏层层数
    hidden_dim = 128  # 隐藏层神经元
    rnn = 'gru'  # lstm 或 gru

    attn_size = 256     # attention layer dimension
    grad_clip = 5       # gradient clipping threshold

    dropout_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 20  # 每批训练大小
    num_epochs = 3  # 总迭代轮次

    print_per_batch = 4  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextABRNN(object):
    """
	用于文本分类的双向RNN
	"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.abrnn()

    def abrnn(self):


        # 定义前向RNN Cell
        with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
            print(tf.get_variable_scope().name)
            lstm_fw_cell_list = [tf.contrib.rnn.LSTMCell(self.config.hidden_dim) for _ in range(self.config.num_layers)]
            lstm_fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list),
                                                           output_keep_prob=self.config.dropout_keep_prob)

        # 定义反向RNN Cell
        with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
            print(tf.get_variable_scope().name)
            lstm_bw_cell_list = [tf.contrib.rnn.LSTMCell(self.config.hidden_dim) for _ in range(self.config.num_layers)]
            lstm_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list),
                                                           output_keep_prob=self.config.dropout_keep_prob)

        with tf.device('/cpu:0'):
            embedding = tf.Variable(
                tf.truncated_normal([self.config.vocab_size, self.config.embedding_dim], stddev=0.1), name='embedding')
            inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        # self.input_x shape: (batch_size , sequence_length)
        # inputs shape : (batch_size , sequence_length , rnn_size)

        # bidirection rnn 的inputs shape 要求是(sequence_length, batch_size, rnn_size)
        # 因此这里需要对inputs做一些变换
        # 经过transpose的转换已经将shape变为(sequence_length, batch_size, rnn_size)
        # 只是双向rnn接受的输入必须是一个list,因此还需要后续两个步骤的变换
        inputs = tf.transpose(inputs, [1, 0, 2])
        # 转换成(batch_size * sequence_length, rnn_size)
        inputs = tf.reshape(inputs, [-1, self.config.hidden_dim])
        # 转换成list,里面的每个元素是(batch_size, rnn_size)
        inputs = tf.split(inputs, self.config.seq_length, 0)

        with tf.name_scope('bi_rnn'), tf.variable_scope('bi_rnn'):
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, inputs,
                                                                    dtype=tf.float32)

        # 定义attention layer
        attention_size = self.config.attn_size

        with tf.name_scope('attention'), tf.variable_scope('attention'):
            attention_w = tf.Variable(tf.truncated_normal([2 * self.config.hidden_dim, attention_size], stddev=0.1),
                                      name='attention_w')
            attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
            u_list = []
            for t in range(self.config.seq_length):
                u_t = tf.tanh(tf.matmul(outputs[t], attention_w) + attention_b)
                u_list.append(u_t)
            u_w = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw')
            attn_z = []
            for t in range(self.config.seq_length):
                z_t = tf.matmul(u_list[t], u_w)
                attn_z.append(z_t)
            # transform to batch_size * sequence_length
            attn_zconcat = tf.concat(attn_z, axis=1)
            self.alpha = tf.nn.softmax(attn_zconcat)
            # transform to sequence_length * batch_size * 1 , same rank as outputs
            alpha_trans = tf.reshape(tf.transpose(self.alpha, [1, 0]), [self.config.seq_length, -1, 1])
            self.final_output = tf.reduce_sum(outputs * alpha_trans, 0)

        # print self.final_output.shape
        print(self.final_output.shape)
        # outputs shape: (sequence_length, batch_size, 2*rnn_size)
        fc_w = tf.Variable(tf.truncated_normal([2 * self.config.hidden_dim, self.config.num_classes], stddev=0.1),
                           name='fc_w')
        fc_b = tf.Variable(tf.zeros([self.config.num_classes]), name='fc_b')

        # self.final_output = outputs[-1]

        # 用于分类任务, outputs取最终一个时刻的输出
        self.logits = tf.matmul(self.final_output, fc_w) + fc_b
        self.prob = tf.nn.softmax(self.logits)
        self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        self.loss = tf.losses.softmax_cross_entropy(self.input_y, self.logits)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.config.grad_clip)

        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        # 准确率
        correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #self.accuracy = tf.reduce_mean(
          #  tf.cast(tf.equal(tf.argmax(self.input_y, axis=1), tf.argmax(self.y_pred_cls, axis=1)), tf.float32))

    def inference(self, sess, labels, inputs):

        prob = sess.run(self.prob, feed_dict={self.input_x: inputs, self.keep_prob: 1.0})
        ret = np.argmax(prob, 1)
        ret = [labels[i] for i in ret]
        return ret



