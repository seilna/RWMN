import tensorflow as tf
import numpy as np
import sys
import time
import os
import h5py
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
from IPython import embed
from tensorflow import flags
FLAGS = flags.FLAGS
from compact_bilinear_pooling import compact_bilinear_pooling_layer as cbp
#from tensorflow.contrib.framework import get_variables_by_name as gvn
class ltm_video_sub_model(object):
    def __init__(self, flags, inputs):
        self.init_lr = flags.init_lr
        self.batch_size = flags.batch_size
        self.num_hop = flags.num_hop
        self.num_answer = flags.num_answer
        self.dim_text = flags.dim_text
        self.dim_memory = flags.dim_memory
        self.dim_mcb_output = flags.dim_mcb_output
        self.pretrain = flags.pretrain
        self.initializer = layers.xavier_initializer()
        self.initializer_conv = layers.xavier_initializer_conv2d()
        self.reg = flags.reg
        self.sharp = flags.sharp
        self.data_source = flags.data_source

        self.write = {'kernel': [], 'stride': [], 'channel': []}
        self.read = {'kernel': [], 'stride': [], 'channel': []}

        self.write_mode, self.read_mode = True, True
        if len(flags.write) == 0: self.write_mode = False
        if len(flags.read) == 0: self.read_mode = False
        if len(flags.write) == 0 and len(flags.read) == 0:
            self.baseline = True
        else: self.baseline = False

        if self.write_mode == True:
            for conv in flags.write.split('/'):
                k,s,c = conv.split('-')
                self.write['kernel'].append(int(k))
                self.write['stride'].append(int(s))
                self.write['channel'].append(int(c))

        if self.read_mode == True:
            for conv in flags.read.split('/'):
                k,s,c = conv.split('-')
                self.read['kernel'].append(int(k))
                self.read['stride'].append(int(s))
                self.read['channel'].append(int(c))

        with tf.variable_scope('inputs'):   
            self.rgb = layers.unit_norm(inputs=inputs['rgb'], dim=1, epsilon=1e-12)
            self.sub = layers.unit_norm(inputs=inputs['sub'], dim=1, epsilon=1e-12)
            self.query = layers.unit_norm(inputs=inputs['query'], dim=1, epsilon=1e-12)
            self.answer = layers.unit_norm(inputs=inputs['answer'], dim=1, epsilon=1e-12)
            self.answer_index = inputs['cor_idx']
            self.model_inputs = self.sub
            #self.model_inputs = tf.concat([self.rgb, self.sub], axis=1)

        with tf.variable_scope('memory_write'):
            self.query_w = tf.get_variable(
                    name='query_w', 
                    shape=[self.dim_text, self.dim_memory], 
                    initializer=self.initializer)

            self.query_b = tf.get_variable(
                    name='query_b', 
                    shape=[self.dim_memory], 
                    initializer=self.initializer)
            
    def fc(self, inputs, num_outputs, name):
        with tf.variable_scope(name) as scope:
            return layers.fully_connected(
                    inputs=inputs, num_outputs=num_outputs,
                    weights_initializer=self.initializer,
                    biases_initializer=self.initializer)

    def layer_norm(self, inputs):
        return layers.layer_norm(
                inputs=inputs, center=True, scale=True,
                activation_fn=tf.nn.relu, reuse=False, trainable=True)
                

    def write_network(self, memory):
        print 'Write-CNN', '-'*70
        print self.write
        num_layer = len(self.write['kernel'])
        for i in range(num_layer):
            with tf.variable_scope('write-CNN-%d' % i):
                memory = layers.convolution2d(
                        inputs=memory, num_outputs=self.write['channel'][i],
                        kernel_size=[self.write['kernel'][i], self.dim_memory],
                        stride=[self.write['stride'][i], 1],
                        weights_initializer=self.initializer_conv,
                        biases_initializer=self.initializer,
                        activation_fn=tf.nn.relu)

        memory = tf.reshape(memory, [-1, self.dim_memory])
        return memory

    def read_network(self, memory, query):
        # memory.shape = (m, dim_memory)
        # query.shape = (batch, dim_memory)
        print 'Read-CNN', '-'*70
        print self.read
        memory = tf.reshape(memory, shape=[1, -1, self.dim_memory, 1])

        num_layer = len(self.read['kernel'])
        for i in range(num_layer):
            with tf.variable_scope('Read-CNN-%d' % i):
                memory = layers.convolution2d(
                        inputs=memory, num_outputs=self.read['channel'][i],
                        kernel_size=[self.read['kernel'][i], self.dim_memory],
                        stride=[self.read['stride'][i], 1],
                        weights_initializer=self.initializer_conv,
                        biases_initializer=self.initializer,
                        activation_fn=tf.nn.relu)

        # output_memory = (l, dim_memory)
        # query_shape = (batch, dim_memory)
        output_memory = tf.reshape(memory, shape=[-1, self.dim_memory])
        output_memory = layers.unit_norm(inputs=output_memory, dim=1, epsilon=1e-12)
        
        # att.shape = (batch, n')
        att = tf.matmul(query, output_memory, transpose_b=True)
        att = tf.nn.softmax(self.sharp * att)
        self.att = att
        # output.shape = (batch, dim_memory)
        output = tf.matmul(att, output_memory)
        
        # output.shape = (batch, dim_memory)
        return output

    def build_model(self):
        with tf.variable_scope('query'): 
            # u.shape = (batch, dim_memory)
            self.u = tf.matmul(self.query, self.query_w) + self.query_b
            self.u = tf.reshape(self.u, shape=[-1, self.dim_memory])
            self.u = layers.unit_norm(inputs=self.u, dim=1, epsilon=1e-12)

        with tf.variable_scope('answer'):
            # g.shape = (batch, 5, dim_memory)
            self.g = tf.matmul(tf.reshape(self.answer,shape=[-1, self.dim_text]),
                        self.query_w) + self.query_b
            self.g = tf.reshape(self.g, shape=[-1, self.num_answer, self.dim_memory])
            self.g = layers.unit_norm(inputs=self.g, dim=2, epsilon=1e-12)

        with tf.variable_scope('memory'):
            #video = tf.reshape(self.model_inputs[:, :-300], shape=[-1, 1, 1, FLAGS.dim_rgb])
            #text = tf.reshape(self.model_inputs[:, -300:], shape=[-1, 1, 1, 300])

            #self.memory = cbp(video, text, FLAGS.dim_mcb, sum_pool=True)
            self.memory = self.fc(inputs=self.model_inputs, num_outputs=self.dim_memory, name='E')
            self.memory = layers.unit_norm(inputs=self.memory, dim=1, epsilon=1e-12)
            # (N,H,W,C) style memory

            for _ in range(self.num_hop): 
                if self.write_mode == True:
                    self.memory = self.write_network(tf.reshape(self.memory, shape=[1, -1, self.dim_memory, 1]))

                if self.read_mode == True:
                    self.o = self.read_network(self.memory, self.u)

                elif self.read_mode == False:
                    #----------baseline-----------
                    self.att = tf.matmul(self.u, self.memory, transpose_b=True)
                    self.att = tf.nn.softmax(self.sharp * self.att)
                    self.o = tf.matmul(self.att, self.memory)
                    #-----------------------------

            self.o = layers.unit_norm(self.o, dim=1, epsilon=1e-12)
            self.u = self.o + self.u
            self.u = layers.unit_norm(self.u, dim=1, epsilon=1e-12)
            self.u = tf.reshape(self.u, shape=[-1, self.dim_memory, 1])

        # a.shape = (batch, 1, 5)
        self.a = tf.reshape(
            tf.matmul(self.u, self.g, transpose_a=True, transpose_b=True),
            shape=[-1, self.num_answer])

        self.prob = tf.nn.softmax(self.a)
        self.answer_prediction = tf.argmax(self.prob, dimension=1)
        correct_prediction = tf.equal(self.answer_prediction, self.answer_index)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.correct_examples = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        self.reg_loss = 0.0
        for var in tf.trainable_variables():
            self.reg_loss += tf.nn.l2_loss(var)
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.prob, 
                labels=self.answer_index) + self.reg * self.reg_loss

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
            self.init_lr,
            global_step,
            FLAGS.learning_rate_decay_examples,
            FLAGS.learning_rate_decay_rate,
            staircase=True)

        optimizer = tf.train.AdagradOptimizer(learning_rate)
        #optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.loss)

