#!/usr/bin/python
# -*- coding: utf-8 -*-

# set random seed for consistent predict result
from numpy.random import seed
seed(2018)
from tensorflow import set_random_seed
set_random_seed(2018)


import csv
import numpy as np
import random
import re
from collections import Counter



class data_helper():
	def __init__(self, sequence_max_length=1024,train_file_path='train.csv',):
		self.alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{} '
		self.char_dict = {}
		self.sequence_max_length = sequence_max_length
		self.words, self.word_to_id = self.build_vocab(train_file_path, 20000)
		for i,c in enumerate(self.alphabet):
			self.char_dict[c] = i+1
	def format_str(self,string):
		split_re_str = u'[\u4e00-\u9fa5]|[，．？：；！,.?:;!]+|[A-Za-z]{1,}|[\'\-]+|[0-9]+\.[0-9]+|\d+'
		TOKENIZER_RE = re.compile(split_re_str)
		text = TOKENIZER_RE.findall(string.lower())
		return text


	def build_vocab(self,train_file_path, vocab_size=20000):
		"""根据训练集构建词汇表，存储"""
		"""
			Load CSV file, generate  process text data as Paper did.
		"""
		all_data = []
		labels = []
		with open(train_file_path) as f:
			reader = csv.DictReader(f, delimiter='\t', fieldnames=['class'], restkey='fields')
			for row in reader:
				text = row['fields'][-1].lower()
				content = self.format_str(text)
				all_data.extend(list(content))
		f.close()
		counter = Counter(all_data)
		count_pairs = counter.most_common(vocab_size - 1)
		words, _ = list(zip(*count_pairs))
		# 添加一个 <PAD> 来将所有文本pad为同一长度
		words = ['<PAD>'] + list(words)
		word_to_id = dict(zip(words, range(len(words))))
		return  words,word_to_id

	def word2vec(self,text):
		data = np.zeros(self.sequence_max_length)
		content = self.format_str(text)
		for i in range(0, len(content)):
			if i >= self.sequence_max_length:
				return data
			elif content[i] in self.words:
				data[i] = self.word_to_id[content[i]]
			else:
				# unknown character set to be <PAD>
				data[i] = self.word_to_id['<PAD>']
		return data

	def char2vec(self, text):
		data = np.zeros(self.sequence_max_length)
		for i in range(0, len(text)):
			if i >= self.sequence_max_length:
				return data
			elif text[i] in self.char_dict:
				data[i] = self.char_dict[text[i]]
			else:
				# unknown character set to be 68
				data[i] = 68
		return data

	def load_csv_file(self, filename, num_classes,cate_to_id,is_char=0):
		"""
		Load CSV file, generate one-hot labels and process text data as Paper did.
		"""
		all_data = []
		labels = []
		with open(filename) as f:
			reader = csv.DictReader(f, delimiter='\t',fieldnames=['class'], restkey='fields')
			for row in reader:
				# One-hot
				one_hot = np.zeros(num_classes)
				#one_hot[int(row['class']) - 1] = 1
				#one_hot[int(cate_to_id[row['class']]) - 1] = 1
				one_hot[int(cate_to_id[row['class']])] = 1
				labels.append(one_hot)
				# Char2vec
				data = np.ones(self.sequence_max_length)*68
				text = row['fields'][-1].lower()
				if is_char == 1:
				    # Char2vec
				    all_data.append(self.char2vec(text))
				else:
				    # Word2vec
				    all_data.append(self.word2vec(text))

				#all_data.append(self.char2vec(text))
		f.close()
		return np.array(all_data), np.array(labels)

	def load_dataset(self, dataset_path,is_char):
		# Read Classes Info
		with open(dataset_path+"classes.txt") as f:
			classes = []
			for line in f:
				classes.append(line.strip())
		f.close()
		num_classes = len(classes)
		cat_to_id = dict(zip(classes, range(len(classes))))
		id_to_cat = dict(zip(range(len(classes)), classes))
		# Read CSV Info
		train_data, train_label = self.load_csv_file(dataset_path+'train.csv', num_classes,cat_to_id,is_char)
		test_data, test_label = self.load_csv_file(dataset_path+'test.csv', num_classes,cat_to_id,is_char)
		return train_data, train_label, test_data, test_label,cat_to_id,id_to_cat,classes

	def load_dataset_predict(self, dataset_path,is_char):
		all_data = []
		with open(dataset_path+'predict.csv') as f:
			reader = csv.DictReader(f, delimiter='\t', fieldnames=['content'])
			for row in reader:
				# Char2vec
				data = np.ones(self.sequence_max_length) * 68
				text = row['content'].lower()
				if is_char == 1:
					# Char2vec
					all_data.append(self.char2vec(text))
				else:
					# Word2vec
					all_data.append(self.word2vec(text))
		f.close()
		return np.array(all_data)

		return train_data

	def split_dataset(self,x_test, y_test, dev_ratio):
		'''split test dataset to test and dev set with ratio '''
		test_size = len(x_test)
		print(test_size)
		dev_size = (int)(test_size * dev_ratio)
		print(dev_size)
		x_dev = x_test[:dev_size]
		x_test = x_test[dev_size:]
		y_dev = y_test[:dev_size]
		y_test = y_test[dev_size:]
		return x_test, x_dev, y_test, y_dev

	def batch_iter(self,x, y, batch_size=64):
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

