# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Synthetic Eye Generation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import os
import tensorflow as tf

import utils as utils
import eg_dataset as eg_dataset
import ii_dataset as ii_dataset
from pix2pix import Pix2pix
from resnet import ResNet18


class Solver(object):
    def __init__(self, flags, log_dir=None):
        # Initialize dataset
        self.data = eg_dataset.Dataset(name='generation', resize_factor=flags.resize_factor,
                                       is_train=flags.is_train, log_dir=log_dir,  is_debug=False)
        self.flags = flags
        self.batch_size = self.flags.batch_size
        self.log_dir = log_dir

        self._init_session()
        self._init_variables()

    def _init_session(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Initialize gan model
            self.model = Pix2pix(input_img_shape=self.data.input_img_shape,
                                 gen_mode=self.flags.gen_mode,
                                 lr=self.flags.learning_rate,
                                 total_iters=self.flags.iters,
                                 is_train=self.flags.is_train,
                                 log_dir=self.log_dir,
                                 lambda_1=self.flags.lambda_1,
                                 num_class=self.data.num_seg_class)

        self.sess = tf.compat.v1.Session(graph=self.graph)

    def _init_variables(self):
        with self.sess.as_default():
            with self.graph.as_default():
                self.sess.run(tf.compat.v1.global_variables_initializer())

                # Initialize saver
                self.saver = tf.compat.v1.train.Saver(max_to_keep=1)

    def train(self):
        def feed_run():
            imgs, _, segs = self.data.train_random_batch(batch_size=self.batch_size)
            feed = {self.model.img_tfph: imgs,
                    self.model.mask_tfph: segs,
                    self.model.rate_tfph: 0.5}
            return feed

        self.sess.run(self.model.dis_optim, feed_dict=feed_run())
        self.sess.run(self.model.gen_optim, feed_dict=feed_run())

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, g_loss, g_adv_loss, g_cond_loss, d_loss, summary = self.sess.run(
            [self.model.gen_optim, self.model.gen_loss, self.model.gen_adv_loss, self.model.cond_loss,
             self.model.dis_loss, self.model.summary_op], feed_dict=feed_run())

        return g_loss, g_adv_loss, g_cond_loss, d_loss, summary

    def img_sample(self, iter_time, save_dir, batch_size=4):
        imgs, _, segs = self.data.train_random_batch(batch_size=batch_size)
        feed = {self.model.mask_tfph: segs,
                self.model.rate_tfph: 0.}  # rate: 1 - keep_prob

        samples = self.sess.run(self.model.g_sample, feed_dict=feed)
        utils.save_imgs(img_stores=[segs, samples, imgs], iter_time=iter_time, save_dir=save_dir, is_vertical=True)

    def load_model(self, logger, model_dir, is_train):
        with self.sess.as_default():
            with self.graph.as_default():
                print('model_dir: {}'.format(model_dir))

                if is_train:
                    logger.info(' [*] Reading checkpoint...')
                else:
                    print(' [*] Reading checkpoint...')

                ckpt = tf.train.get_checkpoint_state(model_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                    self.saver.restore(self.sess, os.path.join(model_dir, ckpt_name))

                    meta_graph_path = ckpt.model_checkpoint_path + '.meta'
                    iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

                    if is_train:
                        logger.info(' [!] Load Iter: {}'.format(iter_time))
                    else:
                        print(' [!] Load Iter: {}'.format(iter_time))

                    return True, iter_time + 1
                else:
                    return False, None


# Evaluator
class Evaluator(object):
    def __init__(self, flags, batch_size=128, model_dir=None, log_dir=None):
        # Initialize dataset
        self.data = ii_dataset.Dataset(name='identification', mode=1, is_train=flags.is_train,
                                       log_dir=log_dir, is_debug=False)

        self.batch_size = batch_size
        self.model_dir = model_dir

        self._init_session()
        self._init_variables()

    def _init_session(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.model = ResNet18(input_img_shape=self.data.input_img_shape,
                                  num_classes=self.data.num_identities,
                                  is_train=False)

        self.sess = tf.compat.v1.Session(graph=self.graph)

    def _init_variables(self):
        with self.sess.as_default():
            with self.graph.as_default():
                self.sess.run(tf.compat.v1.global_variables_initializer())

                if self.model_dir is not None:
                    flag, iter_time = self.load_model(os.path.join('../model/identification', self.model_dir))

                    if flag is True:
                        print(' [!] Loadd IdenModel Success! Iter: {}'.format(iter_time))
                    else:
                        exit(' [!] Failed to restore IdenModel {}'.format(self.load_model))

    def eval_val(self, tb_writer, iter_time):
        print(' [*] Evaluate on the validation dataset...')

        # Initialize/reset the running varaibels.
        self.sess.run(self.model.running_vars_initializer)

        for i, index in enumerate(range(0, self.data.num_val_imgs, self.batch_size)):
            print('[{}/{}] processing...'.format(i + 1, (self.data.num_val_imgs // self.batch_size) + 1))

            img_vals, cls_vals = self.data.direct_batch(batch_size=self.batch_size, index=index, stage='val')

            feed = {
                self.model.img_tfph: img_vals,
                self.model.gt_tfph: cls_vals,
                self.model.train_mode: False
            }

            self.sess.run(self.model.accuracy_metric_update, feed_dict=feed)

        # Calculate the accuracy
        accuracy, metric_summary_op = self.sess.run([self.model.accuracy_metric, self.model.metric_summary_op])

        # Write to tensorabrd
        tb_writer.add_summary(metric_summary_op, iter_time)
        tb_writer.flush()

        return accuracy * 100.

    def eval_test(self):
        print(' [*] Evaluate on the test dataset...')

        # Initialize/reset the running variables
        self.sess.run(self.model.running_vars_initializer)

        for j, index in enumerate(range(0, self.data.num_test_imgs, self.batch_size)):
            print('[{}/{}] processing...'.format(j + 1, (self.data.num_test_imgs // self.batch_size) + 1))

            img_test, cls_test = self.data.direct_batch(batch_size=self.batch_size, index=index, stage='test')

            feed = {
                self.model.img_tfph: img_test,
                self.model.gt_tfph: cls_test,
                self.model.train_mode: False
            }

            self.sess.run(self.model.accuracy_metric_update, feed_dict=feed)

        # Calculate the accurac
        accuracy = self.sess.run(self.model.accuracy_metric) * 100.

        return accuracy

    def load_model(self, model_dir):
        print(' [*] Reading identification checkpoint...')

        saver = tf.compat.v1.train.Saver(max_to_keep=1)
        ckpt = tf.train.get_checkpoint_state(model_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(model_dir, ckpt_name))

            meta_graph_path = ckpt.model_checkpoint_path + '.meta'
            iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

            return True, iter_time
        else:
            return False, None

