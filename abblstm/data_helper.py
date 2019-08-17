#!/usr/bin/python
# -*- coding: utf-8 -*-

# set random seed for consistent predict result
from numpy.random import seed
seed(2018)
from tensorflow import set_random_seed
set_random_seed(2018)



import numpy as np



import pandas as pd
import tensorflow as tf

names = ["class", "title", "content"]
cat_to_id = []


def load_data_predict(file_name, sample_ratio=1):
    '''load data from .csv file'''
    pre_data = pd.read_csv(file_name, sep='\t', header=-1)
    x = pd.Series(pre_data[0])
    print("pre_data len:",len(x))
    return x


def data_preprocessing_predict(train, predict, max_len):
    '''transform to one-hot idx vector by VocabularyProcess'''
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_len,min_frequency=1)
    x_transform_train = vocab_processor.fit_transform(train)
    x_transform_predict = vocab_processor.transform(predict)
    x_predict_list = list(x_transform_predict)
    x_predict = np.array(x_predict_list)

    return x_predict



def to_one_hot(y, n_class):
    return np.eye(n_class)[y.astype(int)]

def get_cate_dic(class_path):
    with open(class_path) as f:
        classes = []
        for line in f:
            classes.append(line.strip())
    f.close()
    cat_to_id = dict(zip(classes, range(len(classes))))
    id_to_cat = dict(zip(range(len(classes)), classes))
    return cat_to_id,id_to_cat,classes

def load_data(file_name, sample_ratio= 1, n_class=15, names=names,cat_to_id=cat_to_id):
    '''load data from .csv file'''
    csv_file = pd.read_csv(file_name, names=names,sep='\t')
    # 8 models
    csv_file['new_class'] = csv_file['class'].apply(lambda x: cat_to_id[str(x)])
    shuffle_csv = csv_file.sample(frac=sample_ratio)
    x = pd.Series(shuffle_csv["content"])
    y = pd.Series(shuffle_csv["new_class"])
    y = to_one_hot(y, n_class)
    print(y.shape)
    return x, y


def data_preprocessing(train, test, max_len,all_data):
    '''transform to one-hot idx vector by VocabularyProcess'''
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_len,min_frequency=1)
    train_vec = vocab_processor.fit_transform(all_data)
    x_transform_train = vocab_processor.transform(train)
    x_transform_test = vocab_processor.transform(test)
    vocab = vocab_processor.vocabulary_
    vocab_size = len(vocab)
    x_train_list = list(x_transform_train)
    x_test_list = list(x_transform_test)
    x_train = np.array(x_train_list)
    x_test = np.array(x_test_list)

    return x_train, x_test, vocab, vocab_size


def split_dataset(x_test, y_test, dev_ratio):
    '''split test dataset to test and dev set with ratio '''
    test_size = len(x_test)
    print(test_size)
    dev_size = (int)(test_size * dev_ratio)
    print(dev_size)
    x_dev = x_test[:dev_size]
    x_test = x_test[dev_size:]
    y_dev = y_test[:dev_size]
    y_test = y_test[dev_size:]
    return x_test, x_dev, y_test, y_dev, dev_size, test_size - dev_size


def fill_feed_dict(data_X, data_Y, batch_size):
    '''Generator to yield batches'''
    # Shuffle data first.
    perm = np.random.permutation(data_X.shape[0])
    data_X = data_X[perm]
    data_Y = data_Y[perm]
    for idx in range(data_X.shape[0] // batch_size):
        x_batch = data_X[batch_size * idx: batch_size * (idx + 1)]
        y_batch = data_Y[batch_size * idx: batch_size * (idx + 1)]
        yield x_batch, y_batch


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


if __name__=="__main__":
    x_predict = load_data_predict("../rdc/predict.csv")
