#!/usr/bin/python
# -*- coding: utf-8 -*-
# set random seed for consistent predict result
from numpy.random import seed
seed(2018)
from tensorflow import set_random_seed
set_random_seed(2018)


from collections import Counter
import tensorflow.contrib.keras as kr
import numpy as np
import os
import re


def open_file(filename, mode='r'):
    """
    Commonly used file reader, change this to switch between python2 and python3.
    mode: 'r' or 'w' for read or write
    """
    #return open(filename, mode, encoding='utf-8', errors='ignore')
    return open(filename, mode)

def read_file(filename,is_char_flag,predict_flag=0):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                if predict_flag == 1:
                    content = line.strip()
                else:
                    label, content = line.strip().split('\t')
                    labels.append(label)
                if(is_char_flag == 1):
                    contents.append(list(content))
                else:
                    #contents.append(clean_str(content).split())
                    contents.append(format_str(content))

            except:
                pass
    return contents, labels



def format_str(string):
    split_re_str = u'[\u4e00-\u9fa5]|[，．？：；！,.?:;!]+|[A-Za-z]{1,}|[\'\-]+|[0-9]+\.[0-9]+|\d+'
    TOKENIZER_RE = re.compile(split_re_str)
    text = TOKENIZER_RE.findall(string.lower())
    return text


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets

    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\_\-]", " ", string)
    #string = re.sub(r"[^A-Za-z0-9(),!?\'\`\_\-\u4e00-\u9fa5]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def build_vocab(train_dir, vocab_dir, vocab_size=5000,is_char=0):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir,is_char)

    all_data = []
    for content in data_train:
        all_data.extend(list(content))

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)

    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')

def read_vocab(vocab_dir):
    """读取词汇表"""
    words = open_file(vocab_dir).read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id

def read_category(label_dir):
    """读取分类目录，固定"""
    categories = []
    with open_file(label_dir) as f:
        for line in f:
            category = line.strip()
            categories.append(category)
    #print('categories',categories)
    cat_to_id = dict(zip(categories, range(len(categories))))
    id_to_cat = dict(zip(range(len(categories)),categories))
    return categories, cat_to_id,id_to_cat

def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)

def to_words_word(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in clean_str(content).split())

def process_file(filename, word_to_id, cat_to_id, max_length=600,is_char=1,predict_flag=0):
    """将文件转换为id表示"""
    contents, labels = read_file(filename,is_char,predict_flag)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        if predict_flag == 0 :
            label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    print("x_pad'shape",x_pad.shape)
    if predict_flag == 0  :
        y_pad = kr.utils.to_categorical(label_id)  # 将标签转换为one-hot表示
        print("y_pad'shape", y_pad.shape)
    else:
        y_pad = None
    return x_pad, y_pad




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

    # constant define
    base_dir = '../rdc_data'
    train_dir = os.path.join(base_dir, 'train.csv')
    test_dir = os.path.join(base_dir, 'test.csv')
    val_dir = os.path.join(base_dir, 'test.csv')
    vocab_dir = os.path.join(base_dir, 'cps.vocab.txt')
    label_dir = os.path.join(base_dir, 'classes.txt')
    predict_in_dir = os.path.join(base_dir, 'predict.csv')
    predict_out_dir = os.path.join(base_dir, 'cnn2_predict_out_result_prob.txt')

    build_vocab(train_dir, vocab_dir, 50000,0)

    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, 50000,0)

    categories, cat_to_id, id_to_cat = read_category(label_dir)
    words, word_to_id = read_vocab(vocab_dir)

    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, 100, 0,0)
    x_predict, _ = process_file(predict_in_dir, word_to_id, cat_to_id, 100,0,1)
