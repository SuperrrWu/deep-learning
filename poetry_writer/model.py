# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  #不显示一些提示警告信息

def rnn_model(model, input_data, output_data, vocab_size, rnn_size=128, num_layers=2, batch_size=64,
              learning_rate=0.01):
    """
    construct rnn seq2seq model.
    :param model: model class 模型种类
    :param input_data: input data placeholder 输入
    :param output_data: output data placeholder 输出
    :param vocab_size: 词长度
    :param rnn_size: 一个RNN单元的大小
    :param num_layers: RNN层数,神经元
    :param batch_size: 步长
    :param learning_rate: 学习速率
    :return:
    """
    end_points = {}

    def rnn_cell():
        if model == 'rnn':
            cell_fun = tf.contrib.rnn.BasicRNNCell
        elif model == 'gru':
            cell_fun = tf.contrib.rnn.GRUCell
        elif model == 'lstm':
            # 基础模型
            cell_fun = tf.contrib.rnn.BasicLSTMCell
        # 指定rnn_size大小，H,C，控制值和输出值，是否当做元组返回，默认为True
        cell = cell_fun(rnn_size, state_is_tuple=True)
        return cell
    # 基本单元
    cell = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(num_layers)], state_is_tuple=True)
    # cell = tf.contrib.rnn.MultiRNNCell([rnn_cell()] * num_layers, state_is_tuple=True)


    if output_data is not None:
        # 初始化
        initial_state = cell.zero_state(batch_size, tf.float32)
    else:
        initial_state = cell.zero_state(1, tf.float32)
    # 指定用cpu运行
    with tf.device("/cpu:0"):
        # 输入的向量转化为128维向量，所以先构建隐层，指定值为+1 到-1区间
        embedding = tf.get_variable('embedding', initializer=tf.random_uniform(
            [vocab_size + 1, rnn_size], -1.0, 1.0))
        # embedding = tf.Variable(tf.random_uniform([vocab_size + 1,rnn_size],-1.0,1.0))
        # 输入的不是所有的词，是所有词的一部分，寻找属于哪个词
        inputs = tf.nn.embedding_lookup(embedding, input_data)

    # [batch_size, ?, rnn_size] = [64, ?, 128]
    # 输出，和最后一次的输出
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
    # 转换为128列的输出，output相当于中间一个128维隐层结果，向量
    output = tf.reshape(outputs, [-1, rnn_size])
    # 128维的权重，和词汇量
    weights = tf.Variable(tf.truncated_normal([rnn_size, vocab_size + 1]))
    # 偏置
    bias = tf.Variable(tf.zeros(shape=[vocab_size + 1]))
    # bias加到前面没一行，预测值
    logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)
    # [?, vocab_size+1]

    if output_data is not None:
        # output_data must be one-hot encode  指定深度为，词汇量深度+1，真实值
        labels = tf.one_hot(tf.reshape(output_data, [-1]), depth=vocab_size + 1)
        # should be [?, vocab_size+1]
        # 计算损失,真实值和预测值
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        # loss shape should be [?, vocab_size+1]
        # 对损失求平均
        total_loss = tf.reduce_mean(loss)
        # 训练，优化损失
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        end_points['initial_state'] = initial_state
        end_points['output'] = output
        end_points['train_op'] = train_op
        end_points['total_loss'] = total_loss
        end_points['loss'] = loss
        end_points['last_state'] = last_state
    else:
        # 预测值
        prediction = tf.nn.softmax(logits)

        end_points['initial_state'] = initial_state
        end_points['last_state'] = last_state
        end_points['prediction'] = prediction

    return end_points