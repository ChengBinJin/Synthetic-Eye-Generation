# --------------------------------------------------------------------------
# Tensorflow Implementation of Synthetic Eye Generation Challenge
# Iris Identification
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import os
import cv2
import logging
import numpy as np
from datetime import datetime
import tensorflow as tf

import utils as utils
from dataset import Dataset
from resnet import ResNet18
from ii_solver import Solver


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_integer('mode', 1, '0 for whole image, 1 for iris part, default: 1')
tf.flags.DEFINE_integer('batch_size', 256, 'batch size for one iteration, default: 256')
tf.flags.DEFINE_float('resize_factor', 0.5, 'resize the original input image, default: 0.5')
tf.flags.DEFINE_string('dataset', 'OpenEDS', 'dataset name, default: OpenEDS')
tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'initial learning rate for optimizer, default: 0.001')
tf.flags.DEFINE_float('weight_decay', 1e-4, 'weight decay for model to handle overfitting, default: 0.0001')
tf.flags.DEFINE_integer('epoch', 100, 'number of iters, default: 100')
tf.flags.DEFINE_integer('print_freq', 10, 'print frequence for loss information, default: 10')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model that you wish to continue training '
                                           '(e.g. 20190806-234308), default: None')


def print_main_parameters(logger, flags, is_train=False):
    if is_train:
        logger.info('gpu_index: \t\t\t{}'.format(flags.gpu_index))
        logger.info('mode: \t\t\t{}'.format(flags.mode))
        logger.info('batch_size: \t\t\t{}'.format(flags.batch_size))
        logger.info('resize_factor: \t\t{}'.format(flags.resize_factor))
        logger.info('dataset: \t\t\t{}'.format(flags.dataset))
        logger.info('is_train: \t\t\t{}'.format(flags.is_train))
        logger.info('learning_rate: \t\t{}'.format(flags.learning_rate))
        logger.info('weight_decay: \t\t{}'.format(flags.weight_decay))
        logger.info('epoch: \t\t\t{}'.format(flags.epoch))
        logger.info('print_freq: \t\t\t{}'.format(flags.print_freq))
        logger.info('load_model: \t\t\t{}'.format(flags.load_model))
    else:
        print('-- gpu_index: \t\t\t{}'.format(flags.gpu_index))
        print('-- mode: \t\t\t{}'.format(flags.mode))
        print('-- batch_size: \t\t\t{}'.format(flags.batch_size))
        print('-- resize_factor: \t\t{}'.format(flags.resize_factor))
        print('-- dataset: \t\t\t{}'.format(flags.dataset))
        print('-- is_train: \t\t\t{}'.format(flags.is_train))
        print('-- learning_rate: \t\t{}'.format(flags.learning_rate))
        print('-- weight_decay: \t\t{}'.format(flags.weight_decay))
        print('-- epoch: \t\t\t{}'.format(flags.epoch))
        print('-- print_freq: \t\t\t{}'.format(flags.print_freq))
        print('-- load_model: \t\t\t{}'.format(flags.load_model))


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_index

    # Initialize model and log folders
    if FLAGS.load_model is None:
        cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        cur_time = FLAGS.load_model

    model_dir, log_dir = utils.make_folders_simple(cur_time=cur_time,
                                                   subfolder='identification')

    # Logger
    logger = logging.getLogger(__name__)  # logger
    logger.setLevel(logging.INFO)
    utils.init_logger(logger=logger, log_dir=log_dir, is_train=FLAGS.is_train, name='main')
    print_main_parameters(logger, flags=FLAGS, is_train=FLAGS.is_train)

    # Initialize dataset
    data = Dataset(is_train=FLAGS.is_train, log_dir=log_dir, mode=FLAGS.mode, is_debug=False)

    # Initialize model
    model = ResNet18(input_img_shape=data.input_img_shape,
                     num_classes=data.num_identities,
                     lr=FLAGS.learning_rate,
                     weight_decay=FLAGS.weight_decay,
                     total_iters=int(np.ceil(FLAGS.epoch * data.num_train_imgs / FLAGS.batch_size)),
                     is_train=FLAGS.is_train,
                     log_dir=log_dir)
    # Initialize solver
    solver = Solver(model, data, batch_size=FLAGS.batch_size)

    # Initialize saver
    saver = tf.compat.v1.train.Saver(max_to_keep=1)

    if FLAGS.is_train is True:
        train(solver, saver, logger, model_dir, log_dir)
    else:
        test(solver, saver, model_dir, log_dir)


def train(solver, saver, logger ,model_dir, log_dir):
    best_acc = 0.
    iter_time = 0
    total_iters = int(np.ceil(FLAGS.epoch * solver.data.num_train_imgs / FLAGS.batch_size))
    eval_iters = int(np.ceil(solver.data.num_train_imgs / FLAGS.batch_size))

    if FLAGS.load_model is not None:
        flag, iter_time, best_acc = load_model(
            saver=saver, solver=solver, model_dir=model_dir, logger=logger, is_train=True)

        if flag is True:
            logger.info(' [!] Load Success! Iter: {}'.format(iter_time))
            logger.info('Best Acc.: {:.3f}'.format(best_acc))
        else:
            exit(' [!] Failed to restore model {}'.format(FLAGS.load_model))

    # Tensorboard writer
    tb_writer = tf.compat.v1.summary.FileWriter(logdir=log_dir, graph=solver.sess.graph_def)

    while iter_time < total_iters:
        total_loss, data_loss, reg_term, batch_acc, summary = solver.train()

        # Write to tensorboard
        tb_writer.add_summary(summary, iter_time)
        tb_writer.flush()

        # Print loss information
        if iter_time % FLAGS.print_freq == 0:
            msg = "[{0:5} / {1:5}] Total loss: {2:.3f}, Data loss: {3:.3f}, Reg. term: {4:.3f}, Batch acc.: {5:.2f}%"
            print(msg.format(iter_time, total_iters, total_loss, data_loss, reg_term, batch_acc))

        # # Evaluate models using validation dataset
        if (iter_time % eval_iters == 0) or (iter_time + 1 == total_iters):
            acc = solver.eval(tb_writer, iter_time)

            if best_acc < acc:
                best_acc = acc
                solver.set_best_acc(best_acc)
                save_model(saver, solver, logger, model_dir, iter_time, best_acc)

            print('Acc.:      {:.3f}%   - Best Acc.:      {:.3f}%'.format(acc, best_acc))

        iter_time += 1

    # Test on train, val, and test dataset using the best model
    flag, iter_time, best_acc = load_model(saver, solver, model_dir, is_train=False)

    if flag is True:
        print(' [!] Load Success! Iter: {}'.format(iter_time))
        print('Best Acc.: {:.3f}'.format(best_acc))
    else:
        exit(' [!] Failed to restore model {}'.format(FLAGS.load_model))

    accuracy = solver.test()
    print("Train accuracy:  {:.2f}%".format(accuracy[0]))
    print("Val accuracy:    {:.2f}%".format(accuracy[1]))
    print("Test accuracy:   {:.2f}%".format(accuracy[2]))


def test(solver, saver, model_dir, log_dir):
    flag, iter_time, best_acc = load_model(saver, solver, model_dir, is_train=False)

    if flag is True:
        print(' [!] Load Success! Iter: {}'.format(iter_time))
        print('Best Acc.: {:.3f}'.format(best_acc))
    else:
        exit(' [!] Failed to restore model {}'.format(FLAGS.load_model))

    solver.test_top_k(log_dir)


def save_model(saver, solver, logger, model_dir, iter_time, best_acc):
    saver.save(solver.sess, os.path.join(model_dir, 'model'), global_step=iter_time)
    logger.info('[*] Model saved! Iter: {}, Best Acc. {:.3f}'.format(iter_time, best_acc))


def load_model(saver, solver, model_dir, logger=None, is_train=False):
    if is_train:
        logger.info(' [*] Reading checkpoint...')
    else:
        print(' [*] Reading checkpoint...')

    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(solver.sess, os.path.join(model_dir, ckpt_name))

        meta_graph_path = ckpt.model_checkpoint_path + '.meta'
        iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

        # Get metmetrics from the model checkpoints
        best_acc = solver.get_best_acc()

        return True, iter_time + 1, best_acc
    else:
        return False, None, None


if __name__ == '__main__':
    tf.compat.v1.app.run()