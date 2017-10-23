# ============================================
# This file contains tensorflow ops for
# 1. Summary writing
# 2. Checkpoint saving
# ============================================

import tensorflow as tf
import os
import shutil
from tensorflow import flags
FLAGS = flags.FLAGS
def variable_summary(var, name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar(name, mean)
    tf.summary.histogram(name, var)


def build_summary_ops(model, flags, sess):
    summary_dir = "./logs/" + FLAGS.name

    '''
    if os.path.exists(summary_dir):
        shutil.rmtree(summary_dir, ignore_errors=True)
    '''
        
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
        os.mkdir(os.path.join(summary_dir, 'train'))
        os.mkdir(os.path.join(summary_dir, 'val'))

    summary_var_dict = {
            'Loss': model.loss, 'Accuracy': model.acc, 'Att': model.att, 'Sharpening': model.sharp
            }
    for name, var in summary_var_dict.iteritems():
        variable_summary(var, name)

    for var in tf.trainable_variables():
        variable_summary(var, var.name)

    merge_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'train') ,sess.graph)
    val_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'val'), sess.graph)

    return merge_op, train_writer, val_writer

def build_checkpoint_ops(flags):
    #checkpoint_dir = os.path.join(flags.checkpoint_dir, flags.summary_name)
    checkpoint_dir = "./logs/" + FLAGS.name

    '''
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    '''
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    saved_op = {}
    for var in tf.trainable_variables():
        saved_op[var.name] = var
    return tf.train.Saver(var_list=saved_op, max_to_keep=1000), checkpoint_dir

def save_checkpoint(sess, checkpoint_dir, saver_op, step):
    checkpoint_name  = os.path.join(checkpoint_dir, 'step')
    path = saver_op.save(sess, checkpoint_name, global_step=step)

