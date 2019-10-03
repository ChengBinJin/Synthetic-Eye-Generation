# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Synthetic Eye Generation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import os
import logging
import tensorflow as tf
from datetime import datetime

import utils as utils
from eg_solver import Solver, Evaluator


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_integer('gen_mode', 1, 'generation mode selection from [1|2|3|4], default: 1')
tf.flags.DEFINE_integer('batch_size', 1, 'batch size for one iteration, default: 1')
tf.flags.DEFINE_float('resize_factor', 0.5, 'resize original input image, default: 0.5')
tf.flags.DEFINE_string('dataset', 'OpenEDS', 'dataset name, default: OpenEDS')
tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate for optimizer, default: 0.0002')
tf.flags.DEFINE_integer('iters', 200000, 'number of iterations, default: 200000')
tf.flags.DEFINE_integer('print_freq', 10, 'print frequency for loss information, default: 50')
tf.flags.DEFINE_float('lambda_1', 100., 'hyper-paramter for the conditional L1 loss, default: 100.')
tf.flags.DEFINE_integer('sample_freq', 50, 'sample frequence for checking qualitative evaluation, default: 500')
tf.flags.DEFINE_integer('sample_batch', 4, 'number of sampling images for check generator quality, default: 4')
tf.flags.DEFINE_integer('eval_freq', 2000, 'save frequency for model, default: 2000')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model that you wish to continue training '
                                           '(e.g. 20191003-103205), default: None')
tf.flags.DEFINE_string('load_iden_model', '20190921-111742',
                       'folder of saved iden_model that you wish to continue training '
                       '(e.g. 20190921-111742), default: None')


def print_main_parameters(logger, flags, is_train=False):
    if is_train:
        logger.info('gpu_index: \t\t\t{}'.format(flags.gpu_index))
        logger.info('gen_mode: \t\t\t{}'.format(flags.gen_mode))
        logger.info('batch_size: \t\t\t{}'.format(flags.batch_size))
        logger.info('resize_factor: \t\t{}'.format(flags.resize_factor))
        logger.info('dataset: \t\t\t{}'.format(flags.dataset))
        logger.info('is_train: \t\t\t{}'.format(flags.is_train))
        logger.info('learning_rate: \t\t{}'.format(flags.learning_rate))
        logger.info('iters: \t\t\t{}'.format(flags.iters))
        logger.info('print_freq: \t\t\t{}'.format(flags.print_freq))
        logger.info('lambda_1: \t\t\t{}'.format(flags.lambda_1))
        logger.info('sample_freq: \t\t{}'.format(flags.sample_freq))
        logger.info('sample_batch: \t\t{}'.format(flags.sample_batch))
        logger.info('eval_freq: \t\t\t{}'.format(flags.eval_freq))
        logger.info('load_model: \t\t\t{}'.format(flags.load_model))
        logger.info('load_iden_model: \t\t{}'.format(flags.load_iden_model))
    else:
        print('-- gpu_index: \t\t\t{}'.format(flags.gpu_index))
        print('-- gen_mode: \t\t\t{}'.format(flags.gen_mode))
        print('-- batch_size: \t\t\t{}'.format(flags.batch_size))
        print('-- resize_factor: \t\t{}'.format(flags.resize_factor))
        print('-- dataset: \t\t\t{}'.format(flags.dataset))
        print('-- is_train: \t\t\t{}'.format(flags.is_train))
        print('-- learning_rate: \t\t{}'.format(flags.learning_rate))
        print('-- iters: \t\t\t{}'.format(flags.iters))
        print('-- print_freq: \t\t\t{}'.format(flags.print_freq))
        print('-- lambda_1: \t\t\t{}'.format(flags.lambda_1))
        print('-- sample_freq: \t\t{}'.format(flags.sample_freq))
        print('-- sample_batch: \t\t{}'.format(flags.sample_batch))
        print('-- eval_freq: \t\t\t{}'.format(flags.eval_freq))
        print('-- load_model: \t\t\t{}'.format(flags.load_model))
        print('-- load_iden_model: \t\t{}'.format(flags.load_iden_model))


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_index

    # Initialize model and log folders:
    if FLAGS.load_model is None:
        cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        cur_time = FLAGS.load_model

    model_dir, log_dir, sample_dir, _, test_dir = utils.make_folders(is_train=FLAGS.is_train,
                                                                     cur_time=cur_time,
                                                                     subfolder='generation')

    # Logger
    logger = logging.getLogger(__name__)  # logger
    logger.setLevel(logging.INFO)
    utils.init_logger(logger=logger, log_dir=log_dir, is_train=FLAGS.is_train, name='main')
    print_main_parameters(logger, flags=FLAGS, is_train=FLAGS.is_train)

    # Initialize solver
    solver = Solver(flags=FLAGS, log_dir=log_dir)

    # Intialize evaluator
    evaluator = Evaluator(flags=FLAGS, model_dir=FLAGS.load_iden_model, log_dir=log_dir)

    if FLAGS.is_train is True:
        train(solver, evaluator, logger, model_dir, log_dir, sample_dir)
    else:
        test(solver, evaluator, model_dir, test_dir)


def train(solver, evaluator, logger, model_dir, log_dir, sample_dir):
    best_acc = 0.
    iter_time = 0

    if FLAGS.load_model is not None:
        flag, iter_time, best_acc = solver.load_model(logger=logger, model_dir=model_dir, is_train=True)

        if flag is True:
            logger.info(' [!] Load Success! Iter: {}'.format(iter_time))
            logger.info('Best Acc.: {:.3f}%'.format(best_acc))
        else:
            exit(' [!] Failed to restore model {}'.format(FLAGS.load_gan_model))

    # Tensorboard writer
    tb_writer = tf.compat.v1.summary.FileWriter(logdir=log_dir, graph=solver.sess.graph_def)

    while iter_time < FLAGS.iters:
        gen_loss, adv_loss, cond_loss, dis_loss, summary = solver.train()

        # Write to tensorboard
        tb_writer.add_summary(summary, iter_time)
        tb_writer.flush()

        # Print loss information
        if iter_time % FLAGS.print_freq == 0:
            msg = "[{0:6} / {1:6}] Dis_loss: {2:.3f} Gen_loss: {3:.3f}, Adv_loss: {4:.3f}, Cond_loss:{5:.3f}"
            print(msg.format(iter_time, FLAGS.iters, dis_loss, gen_loss, adv_loss, cond_loss))

        # Sampling generated imgs
        if iter_time % FLAGS.sample_freq == 0:
            solver.img_sample(iter_time, sample_dir, FLAGS.sample_batch)

        # Evaluating
        if (iter_time % FLAGS.eval_freq == 0) or (iter_time + 1 == FLAGS.iters):
            segs, outputs, clses = solver.generate_val_imgs()
            acc = evaluator.eval_val(tb_writer, iter_time, segs, outputs, clses)

            if best_acc < acc:
                best_acc = acc
                solver.set_best_acc(best_acc)
                solver.save_model(logger, model_dir, iter_time, best_acc)

            print('Acc.:      {:.3f}%   - Best Acc.:      {:.3f}%'.format(acc, best_acc))


        iter_time += 1


def test(solver, evaluator, model_dir, test_dir):
    if FLAGS.load_model is not None:
        flag, iter_time, best_acc = solver.load_model(logger=None, model_dir=model_dir, is_train=False)

        if flag is True:
            print(' [!] Load Success! Iter: {}'.format(iter_time))
            print('Best Acc.: {:.3f}%'.format(best_acc))
        else:
            exit(' [!] Failed to restore model {}'.format(FLAGS.load_gan_model))

    segs, outputs, clses, imgs = solver.generate_test_imgs()
    acc = evaluator.eval_test(segs, outputs, clses)

    for i in range(segs.shape[0]):
        utils.save_imgs(img_stores=[segs[i:i+1], outputs[i:i+1], imgs[i:i+1]], save_dir=test_dir,
                        img_name=os.path.basename(solver.data.test_paths[i]), is_vertical=False)

    print('Test acc: {:.3f}%'.format(acc))




if __name__ == '__main__':
    tf.compat.v1.app.run()
