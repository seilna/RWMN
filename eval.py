from __future__ import print_function
import sys
sys.path.append('./model')
sys.path.append('./config')
sys.path.append('./model/tf-cbp')
from ltm import ltm_model
from memnet_configs import config
from glob import glob
from utils import build_summary_ops, build_checkpoint_ops, save_checkpoint
from time import time
import tensorflow as tf
import numpy as np
import os
from os.path import join as opj
from numpy.random import randn as nrr
from custom_input_ops import BatchQueue, LTM_Queue
from tensorflow import flags
FLAGS = flags.FLAGS
def full_eval(step, _inputs, model, sess, STORY_FILE, QA_VAL, num_val_examples):
    if FLAGS.video_features:
        full_eval_queue = Clip_Queue(
            story_filename=STORY_FILE,
            qa_filelist=QA_VAL,
            capacity=30,
            batch_size=1,
            num_threads=1)
    else:
        full_eval_queue = LTM_Queue(
            story_filename=STORY_FILE,
            qa_filelist=QA_VAL,
            capacity=30,
            batch_size=1,
            num_threads=1)
    full_eval_queue.start_threads(sequential=True)

    qid_dict = {}
    correct_examples = 0.0
    validation_examples = 0

    while True:
        queue_inputs = full_eval_queue.get_inputs()
        feed = {}
        if FLAGS.video_features == True:
            keys = ['query', 'answer', 'rgb', 'sub', 'cor_idx']
        else:
            keys = ['query', 'answer', 'story', 'cor_idx']

        for key in keys:
            try: feed[_inputs[key]] = queue_inputs[key + '_rep']
            except: feed[_inputs[key]] = queue_inputs[key]
        correct_examples += sess.run(model.correct_examples, feed)
        validation_examples += len(queue_inputs["qid"])
        if validation_examples == num_val_examples: break

    accuracy = correct_examples * 100 / num_val_examples
    assert validation_examples == num_val_examples
    print ('')
    print ('| Step %07d | Acc %.3f |' % (step, accuracy))
    return accuracy

def main():
    flags = config()
    STORY_FILE = opj(os.getcwd(), 'data', flags.data_source, 'story.h5')
    QA_TRAIN = glob(opj(os.getcwd(), 'data', flags.data_source, 'train', '*.h5'))
    QA_VAL = glob(opj(os.getcwd(), 'data', flags.data_source, 'val', '*.h5'))

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth=True
    with tf.Session(config=gpu_config) as sess:
        _inputs = {
                'query': tf.placeholder(dtype=tf.float32, shape=[None, flags.dim_text]),
                'answer': tf.placeholder(dtype=tf.float32, shape=[None, 5, flags.dim_text]),
                'story': tf.placeholder(dtype=tf.float32, shape=[None, flags.dim_mcb]),
                'cor_idx': tf.placeholder(dtype=tf.int64, shape=[None]),
                }

        model = ltm_model(flags=flags, inputs=_inputs)
        model.build_model()
        saved_op = {}
        for var in tf.trainable_variables():
            print (var.name)
            saved_op[var.name] = var

        saver = tf.train.Saver(saved_op)
        saver.restore(sess, flags.checkpoint_file)

        num_val_examples = {
                'video_sub':886,
                'video_sub_aug':886,
                'sub':1958,
                'dvs':282,
                'script':976}

        acc = full_eval(0, _inputs, model, sess, STORY_FILE, QA_VAL, num_val_examples[flags.data_source])
        print (acc)
        '''
        val_queue = LTM_Queue(
                story_filename=STORY_FILE, qa_filelist=QA_VAL, capacity=30, batch_size=1, num_threads=1)

        val_queue.start_threads(sequential=True)
        for var in tf.trainable_variables():
            print (var.name)

        qid_dict = {}
        total_acc = 0.0
        for step in xrange(int(1e+8)):
            ts = time()
            queue_inputs = val_queue.get_inputs()
            feed = {}
            for key, val in _inputs.iteritems():
                try: feed[_inputs[key]] = queue_inputs[key + '_rep']
                except: feed[_inputs[key]] = queue_inputs[key]
            acc = sess.run(model.acc, feed)
            total_acc += acc
            print ('{0} | {1}'.format(queue_inputs['qid'], acc), end='\r')
            assert queue_inputs['qid'][0] not in qid_dict.keys()
            qid_dict[queue_inputs['qid'][0]] = True
            if len(qid_dict) == flags.num_val_examples:
                break
        total_acc = total_acc * 100 / flags.num_val_examples
        print ('Total acc %f' % total_acc)
        '''

if __name__ == '__main__':
    main()
