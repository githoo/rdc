#!/usr/bin/python
# -*- coding: utf-8 -*-

# set random seed for consistent predict result
from numpy.random import seed
seed(2018)
from tensorflow import set_random_seed
set_random_seed(2018)



import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn




class TAblstmConfig(object):
    """配置参数"""
    learning_rate = 1e-3  # 学习率
    dropout_keep_prob = 0.5 # dropout保留比例
    batch_size = 20         # 每批训练大小 256

    embedding_dim = 300  # 词向量维度
    seq_length = 25  # 序列长度
    num_classes = 3  # 类别数 3008
    hidden_dim = 512  # 隐藏层神经元
    epsilon = 7.5

    vocab_size = 20000  # 词汇表达小
    voc_freq=[]

    num_epochs = 3  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    topk = 3 #预测前topk


def attention(inputs, attention_size, time_major=False, return_alphas=False):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article

    """

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas

def _scale_l2(x, norm_length):
    alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
    l2_norm = alpha * tf.sqrt(
        tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
    x_unit = x / l2_norm
    return norm_length * x_unit


def add_perturbation(embedded, loss, epsilon):
    """Adds gradient to embedding and recomputes classification loss."""
    grad, = tf.gradients(
        loss,
        embedded,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    grad = tf.stop_gradient(grad)
    perturb = _scale_l2(grad, epsilon)
    return embedded + perturb


def normalize(emb, weights):
    print("Weights: ", weights)
    mean = tf.reduce_sum(weights * emb, 0, keep_dims=True)
    var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.), 0, keep_dims=True)
    stddev = tf.sqrt(1e-6 + var)
    return (emb - mean) / stddev


class TextAblstm(object):
    """文本分类，ablstm模型"""
    def __init__(self, config):
        self.config = config
        self.vocab_freqs = tf.constant(self.config.voc_freq, dtype=tf.float32, shape=(self.config.vocab_size, 1))
        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length])
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes])
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)

        self.ablstm()

    def cal_loss_logit(self,batch_embedded, keep_prob, W, W_fc, b_fc, batch_y, reuse=True, scope="loss"):
        with tf.variable_scope(scope, reuse=reuse) as scope:
            rnn_outputs, _ = bi_rnn(BasicLSTMCell(self.config.hidden_dim), BasicLSTMCell(self.config.hidden_dim),
                                    inputs=batch_embedded, dtype=tf.float32)
            # Attention
            ATTENTION_SIZE = 50
            attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)
            drop = tf.nn.dropout(attention_output, keep_prob)
            # Fully connected layer
            y_hat = tf.nn.xw_plus_b(drop, W_fc, b_fc)
            y_hat = tf.squeeze(y_hat)
        return y_hat, tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=batch_y))




    def ablstm(self):
        """模型"""

        with tf.name_scope("input_embedding"):
            embeddings_var = tf.Variable(
                tf.random_uniform([self.config.vocab_size, self.config.embedding_dim], -1.0, 1.0))
            weights = self.vocab_freqs / tf.reduce_sum(self.vocab_freqs)
            embedding_norm = normalize(embeddings_var, weights)
            print("embedding_norm'shape", embedding_norm)
            batch_embedded = tf.nn.embedding_lookup(embedding_norm, self.input_x)
            print("batch_embedded'shape", batch_embedded)


        with tf.name_scope("loss"):
            W = tf.Variable(tf.random_normal([self.config.hidden_dim], stddev=0.1))
            #attention *2
            W_fc = tf.Variable(tf.truncated_normal([self.config.hidden_dim*2, self.config.num_classes], stddev=0.1))

            b_fc = tf.Variable(tf.constant(0., shape=[self.config.num_classes]))

            logits, cl_loss = self.cal_loss_logit(batch_embedded, self.keep_prob,W, W_fc, b_fc, self.input_y, reuse=False)
            embedding_perturbated = add_perturbation(batch_embedded, cl_loss,self.config.epsilon)
            ad_logits, ad_loss = self.cal_loss_logit(embedding_perturbated, self.keep_prob,
                                                     W,W_fc, b_fc, self.input_y,reuse=True)

            self.loss = cl_loss + ad_loss

            self.optim  = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

            self.y_pred_cls = tf.argmax(tf.nn.softmax(logits), 1)  # 预测类别
            self.predict_label = tf.nn.top_k(tf.nn.softmax(logits), k=self.config.topk, sorted=True, name=None)



        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))














