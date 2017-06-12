# coding:utf-8
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import tensorflow as tf


import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", "./files",
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", "./tmp/model0612.2",
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("min_word_freq", 10, "threshold of low frequency coefficient")
FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class NameInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.name_producer(
            data, batch_size, num_steps, name=name)



class MultiRnnModel(object):
    def __init__(self, is_training, config, input_, ebd):
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.


        def lstm_cell():
            # With the latest TensorFlow source code (as of Mar 27, 2017),
            # the BasicLSTMCell will need a reuse parameter which is unfortunately not
            # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
            # an argument check here:
            if 'reuse' in inspect.getargspec(
                    tf.contrib.rnn.BasicLSTMCell.__init__).args:
                return tf.contrib.rnn.BasicLSTMCell(
                    size, forget_bias=0.0, state_is_tuple=True,
                    reuse=tf.get_variable_scope().reuse)
            else:
                return tf.contrib.rnn.BasicLSTMCell(
                    size, forget_bias=0.0, state_is_tuple=True)

        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)

        # 上面这一段选择 究竟用什么 cell 一种是reuse 一种普通 一种是有dropout策略的

        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        # 多层Cell

        self._initial_state = cell.zero_state(batch_size, data_type())
        self.embedding = tf.constant(ebd, tf.float32)

        with tf.device("/cpu:0"):
            a = tf.Variable([0 for i in range(batch_size*num_steps)], dtype=tf.int64)
            self._input_tmp = tf.reshape(a, [batch_size, num_steps])
            self._input_tmp = input_.input_data

            #embedding = tf.get_variable(
            #    "embedding", [vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(self.embedding, self._input_tmp)
            # shape of input_.input_data : batch_size * num_steps
            # shape of inputs: batch_size * num_steps * hidden_size

        # 如果是训练 inputs也要dropout
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
		
 

        # Simplified version of models/tutorials/rnn/rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        # outputs, state = tf.contrib.rnn.static_rnn(
        #     cell, inputs, initial_state=self._initial_state)
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                # 如果不是第一步，就要重用变量
                # 如果是第一步，就是创建
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
        # shape of outputs: batch_size * num_steps * hidden_size
        # shape of output: (batch_size * num_steps) * hidden_size
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b
        # shape of logits: (batch_size * num_steps) * vocab_size
        # shape of tf.reshape(input_.targets, [-1]): (batch_size * num_steps)
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(input_.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=data_type())])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def input_tmp(self):
        return self._input_tmp


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 3
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 30
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 15000


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


def run_epoch(session, model, eval_op=None, verbose=False, id_to_word=None ,output_fi=None):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op
    else:
        fetches["input_tmp"] = model.input_tmp

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if eval_op is None and output_fi is not None:
            input_name = vals["input_tmp"]
            output = ' '.join([id_to_word[x] for x in input_name[0]]) + '\t' + str(cost) + '\n'
            output_fi.write(output)

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --name file data_path")
    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 5


    ebd = np.array(np.zeros([config.vocab_size, config.hidden_size]))
    word_to_id = dict()
    id_to_word = dict()
    count = 0
    with open('./files/vecs') as fi:
        for line in fi:
            arr = line.decode('utf-8').strip().split(' ')
            ebd[count] = np.array(arr[1:], np.float32)
            word_to_id[arr[0]] = count
            id_to_word[count] = arr[0]
            count = count + 1

    raw_data = reader.name_file_raw_data(FLAGS.data_path, word_to_id, FLAGS.min_word_freq)
    # train_data, valid_data, test_data, _ = raw_data
    train_data, test_data = raw_data


    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = NameInput(config=config, data=train_data, name="TrainInput")

            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = MultiRnnModel(is_training=True, config=config, input_=train_input, ebd=ebd)

            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        # with tf.name_scope("Valid"):
        #     valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
        #     with tf.variable_scope("Model", reuse=True, initializer=initializer):
        #         mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
        #     tf.summary.scalar("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            test_input = NameInput(config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = MultiRnnModel(is_training=False, config=eval_config,
                                 input_=test_input, ebd=ebd)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                # 更新learnning rate
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                             verbose=True)
                # print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                # valid_perplexity = run_epoch(session, mvalid)
                # print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
            with open('jieguo', 'w') as fi:
                test_perplexity = run_epoch(session, mtest, id_to_word=id_to_word, output_fi=fi)
            #print("Test Perplexity: %.3f" % test_perplexity)

            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
    tf.app.run()
