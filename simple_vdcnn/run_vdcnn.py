#!/usr/bin/python
# -*- coding: utf-8 -*-

# set random seed for consistent predict result
from numpy.random import seed
seed(2018)
from tensorflow import set_random_seed
set_random_seed(2018)

import os
from vdcnn_model import *
from sklearn import metrics
from data_helper import *

import time
from datetime import timedelta


# constant define
base_dir = '../rdc_data/'
train_dir = os.path.join(base_dir, 'train.csv')
predict_out_dir = os.path.join(base_dir, 'vdcnn_predict_out_result_prob.txt')

save_dir = 'checkpoints'
save_path = os.path.join(save_dir, 'best_validation')   # 最佳验证结果保存路径
tensorboard_dir = 'tensorboard'

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def feed_data(x_batch, y_batch, keep_prob,is_training):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob,
        model.is_training: is_training
    }
    return feed_dict

def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = data_helper.batch_iter(x_, y_, config.batch_size)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0,False)
        loss, acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len

def train(x_train,y_train,x_val,y_val):
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖

    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0              # 总批次
    best_acc_val = 0.0           # 最佳验证集准确率
    last_improved = 0            # 记录上一次提升批次
    require_improvement = 1000   # 如果超过1000轮未提升，提前结束训练
    sum_total = 0
    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)        
        sum_total = sum_total + total_batch
        total_batch = 0 
        last_improved = 0
        batch_train = data_helper.batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob,True)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.accuracy], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)   # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},'\
                    + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        #if flag:  # 同上
        #    break
    time_dif = get_time_dif(start_time)
    print("sum_total=%s,time=%s " % (sum_total,time_dif))


def test(x_test,y_test):
    print("Loading test data...")
    start_time = time.time()

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = config.batch_size
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32) # 保存预测结果
    for i in range(num_batch):   # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0,
            model.is_training: False
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

def predict(x_predict):
    print("Loading predict data...")
    start_time = time.time()    


    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Predicting...')
    y_pred_cls_result = []
    
    batch_size = config.batch_size
    data_len = len(x_predict)
    num_batch = int((data_len - 1) / batch_size) + 1

    for i in range(num_batch):   # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_predict[start_id:end_id],
            model.keep_prob: 1.0,
            model.is_training: False
        }
        predict_result = session.run(model.predict_label, feed_dict=feed_dict)
        #predict result process    
        labels = predict_result[1]
        values = predict_result[0]
        kvs = zip(labels,values)    
        for kv in kvs:                   
            kv_str = ' '.join([id_to_cat[int(kv[0][0])],str(kv[1][0])])
            #print("kv_str",kv_str)
            y_pred_cls_result.append(kv_str)
    #return y_pred_cls_result
    with open(predict_out_dir,'w') as fw:       
        for label in y_pred_cls_result:
            #print("label",label)
            fw.write(label.replace(' ','\t')+'\n')
    print("predict finished")       
 

if __name__ == '__main__':
    # if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test','predict']:
    #     raise ValueError("""usage: python run_abblstm.py [train / test / predict]""")

    print('Configuring VDCNN model...')
    is_char = 0
    config = vdcnnConfig()
    # if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
    #     build_vocab(train_dir, vocab_dir, config.vocab_size,is_char)
    # build_vocab(train_dir, vocab_dir, config.vocab_size, is_char)

    # Data Preparation
    # Load data
    print("Loading data...")
    data_helper = data_helper(sequence_max_length=config.seq_length,train_file_path=train_dir)
    x_train, y_train, test_data, test_label,cat_to_id,id_to_cat,categories = data_helper.load_dataset(base_dir, 0)
    print("train size: ", len(x_train))
    x_test, x_val, y_test, y_val = data_helper.split_dataset(test_data, test_label,0.1)
    print("test size: ", len(x_test))
    print("Validation size: ", len(x_val))
    num_batches_per_epoch = int((len(x_train) - 1) / config.batch_size) + 1
    print("num_batches_per_epoch size: ", num_batches_per_epoch)
    x_predict = data_helper.load_dataset_predict(base_dir, 0)
    print("predict size: ", len(x_predict))
    print("Loading data succees...")

    config.num_classes = len(categories)
    config.num_batches_per_epoch = num_batches_per_epoch
    model = VDCNN(config)

    train(x_train,y_train,x_val,y_val)
    test(x_test,y_test)
    predict(x_predict)

    # if sys.argv[1] == 'train':
    #     train()
    # elif sys.argv[1] == 'predict':
    #     predict(x_predict)
    # else:
    #     test()

