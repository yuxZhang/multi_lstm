# !/usr/bin/python
# coding=utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import numpy as np
reload(sys)
sys.setdefaultencoding('utf-8')


import tensorflow as tf


def _read_word2id(word2id_file, vec_file):
    vec_dict = dict()
    word2id = dict()
    id2word = dict()
    with open(vec_file) as fi:
        for line in fi:
            arr = line.decode('utf-8').strip().split(' ')
            word = arr[0]
            vec = np.array(arr[1:], dtype=np.float32)
            vec_dict[word] = vec
    vec_list = []
    count = 0
    with open(word2id_file) as fi:
        for line in fi:
            arr = line.decode('utf-8').strip().split(' ')
            word = arr[0]
            if word in vec_dict:
                vec_list.append(vec_dict[word])
                word2id[word] = count
                id2word[count] = word
                count = count + 1

    print (count)
    print (len(vec_list))



def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().decode('utf-8').replace("\n", " ").split()


def _build_vocab(filename, min_word_freq=0):
    data = _read_words(filename)

    word_to_id = dict()
    id_to_word = dict()
    count = 0
    counter = sorted(collections.Counter(data).items(), key=lambda x:x[1], reverse=True)
    for i in range(len(counter)):
        x = counter[i]
        if x[1] >= min_word_freq:
            word_to_id[x[0]] = count
            id_to_word[count] = x[0]
            count = count + 1
        else:
            break
    word_to_id['<unknow>'] = count
    id_to_word[count] = '<unknow>'

    return word_to_id, id_to_word


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    data_to_id = []
    for word in data:
        if word in word_to_id:
            data_to_id.append(word_to_id[word])
        else:
            data_to_id.append(word_to_id["<unknow>"])
    return data_to_id


def name_file_raw_data(data_path, word_to_id, min_word_freq=0):
    train_path = os.path.join(data_path, "inputs")
    test_path = os.path.join(data_path, "tests")
    #word_to_id, id_to_word = _build_vocab(train_path, 0)
    train_data = _file_to_word_ids(train_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, test_data


def name_producer(raw_data, batch_size, num_steps, name=None):
    with tf.name_scope(name, "NameProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0: batch_size * batch_len],
                          [batch_size, batch_len])
        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(epoch_size,
                                       message='epoch_size == 0, decrease batch_size or num_steps')
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name='epoch_size')

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],
                             [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                             [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y
