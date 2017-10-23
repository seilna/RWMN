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
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import threading
from os.path import join as opj
from IPython import embed
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
QA_TRAIN = glob(opj(os.getcwd(), 'data', flags.data_source, 'train', '*.h5'))
QA_VAL = glob(opj(os.getcwd(), 'data', flags.data_source, 'val', '*.h5'))

SUMMARY_STEP = FLAGS.summary_step
CHECKPOINT_STEP = FLAGS.checkpoint_step
VAL_STEP = FLAGS.validation_step
FULL_EVAL_STEP = FLAGS.full_validation_step

def main():
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth=True
    if flags.data_source=='video_sub': flags.dim_mcb=4096
    with tf.Session(config=gpu_config) as sess:
        _inputs = {
                'query': tf.placeholder(dtype=tf.float32, shape=[None, flags.dim_text], name='query_input'),
                'answer': tf.placeholder(dtype=tf.float32, shape=[None, 5, flags.dim_text], name='answer_input'),
                'story': tf.placeholder(dtype=tf.float32, shape=[None, flags.dim_mcb], name='story_input'),
                'cor_idx': tf.placeholder(dtype=tf.int64, shape=[None], name='cor_idx_input'),
                'rgb': tf.placeholder(dtype=tf.float32, shape=[None, flags.dim_rgb]),
                'sub': tf.placeholder(dtype=tf.float32, shape=[None, flags.dim_sub]) 
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

        model_vars = tf.contrib.framework.get_model_variables()
        tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)
        saved_op = {}
        for var in tf.trainable_variables():
            print (var.name)
            saved_op[var.name] = var

        train_queue = LTM_Queue(
                story_filename=STORY_FILE, qa_filelist=QA_TRAIN, capacity=30, batch_size=32, num_threads=20)

        val_queue = LTM_Queue(
                story_filename=STORY_FILE, qa_filelist=QA_VAL, capacity=30, batch_size=32, num_threads=10)

        train_queue.start_threads()
        val_queue.start_threads()
    
        merge_op, train_writer, val_writer = build_summary_ops(model, flags, sess)
        checkpoint_op, checkpoint_dir = build_checkpoint_ops(flags)
        tf.global_variables_initializer().run()

        best_accuracy = 0.0
        for step in xrange(FLAGS.max_steps):
            ts = time()
            queue_inputs = train_queue.get_inputs()
            feed = {}
            for key, val in _inputs.iteritems():
                try: feed[_inputs[key]] = queue_inputs[key + '_rep']
                except: feed[_inputs[key]] = queue_inputs[key]
            loss, _ = sess.run([model.loss, model.train_op], feed_dict=feed)
            loss_mean = np.mean(loss)

            print ('| Step %07d | Loss %.3f | Time spent %.3f |' % (step, loss_mean, time()-ts), end='\r')
            if step == 0: continue
            if step % SUMMARY_STEP == 0:
                summary = sess.run(merge_op, feed_dict=feed)
                train_writer.add_summary(summary, step)

            if step % VAL_STEP == 0:
                queue_inputs = val_queue.get_inputs()
                feed = {}
                for key, val in _inputs.iteritems():
                    try:feed[_inputs[key]] = queue_inputs[key + '_rep']
                    except: feed[_inputs[key]] = queue_inputs[key]
                loss = sess.run(model.loss, feed_dict=feed)
                loss_mean = np.mean(loss)

                summary = sess.run(merge_op, feed_dict=feed)
                val_writer.add_summary(summary, step)

            if step % FULL_EVAL_STEP == 0 and step >= 900:
                num_val_examples = {
                        'VideoSubInception': 886,
                        'VideoSubResnet': 886,
                        'video_sub':886,
                        'sub':1958,
                        "sub_part":1958,
                        'dvs':282,
                        'dvs_pe':282,
                        'script':976,
                        'plot':1958,
                        'sub_fasttext':1958
                        }
                accuracy = full_eval(step, _inputs, model, sess, STORY_FILE, QA_VAL, num_val_examples[FLAGS.data_source])
                best_accuracy = max([best_accuracy, accuracy])
                if accuracy >= FLAGS.save_threshold:
                    print ("Saving checkpoint| Step {} | Accuracy : {} ".format(step, accuracy))
                    save_checkpoint(sess, checkpoint_dir, checkpoint_op, step)
    print ("Best Accuracy : {}".format(best_accuracy))

if __name__ == '__main__':
    main()
