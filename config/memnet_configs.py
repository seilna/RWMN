import tensorflow as tf
import os
from tensorflow import flags

def config():
    flags = tf.app.flags
    flags.DEFINE_bool("sub_with_video_features", False, ".")
    flags.DEFINE_float("save_threshold", 100.0, "checkpoint save threshold")
    flags.DEFINE_integer("max_steps", 2500, "max steps to train")
    flags.DEFINE_integer("learning_rate_decay_examples", 200, "step to decay learning rate")
    flags.DEFINE_float("learning_rate_decay_rate", 0.95, "learning rate decay rate")
    flags.DEFINE_string("name", "NoNameModel", "name of training model.")
    flags.DEFINE_bool('video_features', False, "Video+Sub or Text mode.")
    flags.DEFINE_integer('dim_rgb', 2048, 'Inception: 1536, ResNet: 2048')
    flags.DEFINE_integer('dim_sub', 300, 'word2vec embedding dimension')
    flags.DEFINE_float('init_lr', 0.003, 'learning rate initializer.')
    flags.DEFINE_float('init_accumulator_value', 0.1, 'learning rate initializer.')

    flags.DEFINE_float('bn_decay', 0.99, 'Batchnorm decay rate')
    flags.DEFINE_integer('batch_size', 32, 'mini-batch size.')
    flags.DEFINE_integer('num_hop', 1, 'number of hops.')
    flags.DEFINE_integer('dim_text', 300, 'text feature dimension.')
    flags.DEFINE_integer('dim_mcb', 300, 'mcb feature dimension.')
    flags.DEFINE_integer('dim_memory', 300, 'memory dimension.')
    flags.DEFINE_integer('num_answer', 5, 'number of answer candidates.')
    flags.DEFINE_string('data_source', 'sub', '[video_sub, sub, dvs, script, plot]')

    flags.DEFINE_integer('summary_step', 30, 'summery iteration step.')
    flags.DEFINE_integer('validation_step', 30, 'validation iteration step.')
    flags.DEFINE_integer('checkpoint_step', 1000, 'checkpoint iteration step.')
    flags.DEFINE_integer('full_validation_step', 100, 'full validation iteration step')
    flags.DEFINE_string('summary_dir', os.path.join(os.getcwd(), 'logs'), 'path of summary data.')
    flags.DEFINE_string('checkpoint_dir', os.path.join(os.getcwd(), 'logs'), 'path of checkpoint data.')
    flags.DEFINE_string('summary_name', 'default', 'name of the summary')
    flags.DEFINE_string('model', 'ltm', '[stm, ltm]')
    flags.DEFINE_float('dropout', 1.0, 'keeping probability with dropout.')
    flags.DEFINE_string('checkpoint_file', 'None', 'absolute path of checkpoint file')
    flags.DEFINE_integer('num_val_examples', 886, 'The number of validation QA examples.')
    flags.DEFINE_integer('num_test_examples', 1258, 'The number of test QA examples.')
    flags.DEFINE_integer('dim_context', 300, 'Dimension of context.')
    flags.DEFINE_integer('num_memory_channel', 3, 'Number of memory channels.')
    flags.DEFINE_integer('kernel_height', 7, 'conv kernel height.')
    flags.DEFINE_integer('dim_mcb_output', 600, 'Dimension of MCB output.')
    flags.DEFINE_integer('num_params_level', 0, '# of params level.')

    flags.DEFINE_integer('early_stop', 4, 'Early stop threshold.')
    flags.DEFINE_integer('wconv_h', 40, 'conv filter height of Write Network')
    flags.DEFINE_integer('wconv_w', 300, 'conv filter width of Write Network')
    flags.DEFINE_integer('wstride_h', 30, 'stride height of Write Network')
    flags.DEFINE_integer('wstride_w', 1, 'stride width of Write Network')

    flags.DEFINE_integer('rconv_h', 3, 'conv filter height of Write Network')
    flags.DEFINE_integer('rconv_w', 300, 'conv filter width of Write Network')
    flags.DEFINE_integer('rstride_h', 1, 'stride height of Write Network')
    flags.DEFINE_integer('rstride_w', 1, 'stride width of Write Network')

    flags.DEFINE_string('pretrain', 'default', 'pretrained checkpoint file.')

    flags.DEFINE_string('write', '', '10-5-1/10-5-1  --> conv_h, stride_h, channel')
    flags.DEFINE_string('read', '', '5-3-3/5-3-3 --> conv_h, stride_h, channel')
    flags.DEFINE_float('reg', 0.0, 'Regularization strength')
    flags.DEFINE_float('checkpoint_threshold', 0.0, 'Checkpoint threshold')
    flags.DEFINE_float('init_acc', '0.0', 'init acc')
    flags.DEFINE_float('sharp', 1.0, 'sharpening parameter.')

    return flags.FLAGS

