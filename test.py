from __future__ import print_function
import sys
sys.path.append('./model')
sys.path.append('./config')
sys.path.append('./model/tf-cbp')
from ltm import ltm_model
from ltm_video import ltm_video_model
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
from tensorflow import flags

FLAGS = flags.FLAGS
flags = config()

if FLAGS.video_features == True:
    STORY_FILE = opj(os.getcwd(), 'data', flags.data_source, 'story.hkl')
else:
    STORY_FILE = opj(os.getcwd(), 'data', flags.data_source, 'story.h5')
QA_TEST = glob(opj(os.getcwd(), 'data', flags.data_source, 'test', '*.h5'))
QA_VAL = glob(opj(os.getcwd(), 'data', flags.data_source, 'val', '*.h5'))
def main():
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth=True
    with tf.Session(config=gpu_config) as sess:
        _inputs = {
                'query': tf.placeholder(dtype=tf.float32, shape=[None, flags.dim_text]),
                'answer': tf.placeholder(dtype=tf.float32, shape=[None, 5, flags.dim_text]),
                'story': tf.placeholder(dtype=tf.float32, shape=[None, flags.dim_mcb]),
                'cor_idx': tf.placeholder(dtype=tf.int64, shape=[None]),
                'rgb': tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.dim_rgb]),
                'sub': tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.dim_sub])
                }
        if FLAGS.video_features == True:
            _inputs.pop("story", None)
        else:
            _inputs.pop("rgb", None)
            _inputs.pop("sub", None)

        if FLAGS.video_features == True:
            model = ltm_video_model(flags=flags, inputs=_inputs)
        else:    
            model = ltm_model(flags=flags, inputs=_inputs)
        model.build_model()
        saved_op = {}
        for var in tf.trainable_variables():
            print (var.name)
            saved_op[var.name] = var

        saver = tf.train.Saver(saved_op)
        saver.restore(sess, flags.checkpoint_file)

        test_queue = LTM_Queue(
                story_filename=STORY_FILE, qa_filelist=QA_TEST, capacity=30, batch_size=1, num_threads=1)

        test_queue.start_threads(sequential=True)
        for var in tf.trainable_variables():
            print (var.name)

        num_test_examples ={
                'video_sub':1258,
                'video_sub_aug':1258,
                "VideoSubInception":1258,
                "VideoSubResnet":1258,
                'sub':3138,
                'script':1598,
                'plot':3138,
                'dvs':570}
                
        qid_dict = {}
        while True:
            ts = time()
            queue_inputs = test_queue.get_inputs()
            feed = {}
            for key, val in _inputs.iteritems():
                if key=='cor_idx': continue
                try: feed[_inputs[key]] = queue_inputs[key + '_rep']
                except: feed[_inputs[key]] = queue_inputs[key]

            batch_predictions = sess.run(model.answer_prediction, feed)

            for i in range(len(queue_inputs["qid"])):
                qid = queue_inputs['qid'][i]
                pred = batch_predictions[i]
                print (pred)
                result = str(qid) + ' ' + str(pred) + '\n'
                
                assert qid not in qid_dict.keys()
                qid_dict[qid] = str(pred)
            if len(qid_dict) == num_test_examples[flags.data_source]:
                break
        f = open('./' + flags.data_source + '_result.txt', 'w')
        keys = qid_dict.keys()
        def sort_fun(x):
            return int(x.split(':')[-1])
        keys.sort(key=sort_fun)
        for key in keys:
            print (key, qid_dict[key])
            result = str(key) + ' ' + str(qid_dict[key]) + '\n'
            f.write(result)
        f.close()
            
if __name__ == '__main__':
    main()
