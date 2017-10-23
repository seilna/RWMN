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
from os.path import join as opj
import tensorflow as tf
import numpy as np
import os
from numpy.random import randn as nrr
from custom_input_ops import BatchQueue, LTM_Queue
from eval import full_eval

flags = config()
STORY_FILE = opj(os.getcwd(), 'data', flags.data_source, 'story.h5')
QA_VAL = glob(opj(os.getcwd(), 'data', flags.data_source, 'val', '*.h5'))
def main():
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth=True
    with tf.Session(config=gpu_config) as sess:
        _inputs = {
                'query': tf.placeholder(dtype=tf.float32, shape=[None, flags.dim_text]),
                'answer': tf.placeholder(dtype=tf.float32, shape=[None, 5, flags.dim_text]),
                'story': tf.placeholder(dtype=tf.float32, shape=[None, flags.dim_mcb]),
                'cor_idx': tf.placeholder(dtype=tf.int64, shape=[None])
                }

        model = ltm_model(flags=flags, inputs=_inputs)
        model.build_model()
        saved_op = {}
        for var in tf.trainable_variables():
            print (var.name)
            saved_op[var.name] = var

        saver = tf.train.Saver(saved_op)
        saver.restore(sess, flags.checkpoint_file)

        val_queue = LTM_Queue(
                story_filename=STORY_FILE, qa_filelist=QA_VAL, capacity=30, batch_size=1, num_threads=1)

        val_queue.start_threads(sequential=True)
        for var in tf.trainable_variables():
            print (var.name)
        
        num_val_examples = {
                'video_sub':886,
                'sub':1958,
                'dvs':282,
                'script':976,
                'plot':1958
                }
                
        qid_dict = {}
        acc = 0.0
        while True:
            ts = time()
            queue_inputs = val_queue.get_inputs()
            feed = {}
            for key, val in _inputs.iteritems():
                try: feed[_inputs[key]] = queue_inputs[key + '_rep']
                except: feed[_inputs[key]] = queue_inputs[key]

            qid = queue_inputs['qid'][0]
            pred = sess.run(model.answer_prediction, feed)[0]
            correct = int(sess.run(model.acc, feed))
            acc += correct
            print (pred, correct)
            result = str(qid) + ' ' + str(pred) + ' %d\n' % correct
            
            assert qid not in qid_dict.keys()
            qid_dict[qid] = str(pred)
            if len(qid_dict) == num_val_examples[flags.data_source]:
                acc /= float(num_val_examples[flags.data_source])
                print (acc)
                    
                break

        f = open('./' + flags.data_source + '_result_val.txt', 'w')
        keys = qid_dict.keys()
        def sort_fun(x):
            return int(x.split(':')[-1])
        keys.sort(key=sort_fun)
        for key in keys:
            result = str(key) + ' ' + str(qid_dict[key]) + '\n'
            f.write(result)
        f.close()
            
if __name__ == '__main__':
    main()
