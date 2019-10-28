# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Synthetic Eye Generation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import os
import numpy as np
import tensorflow as tf
import utils as utils


class Solver(object):
    def __init__(self, data, gen_model, flags, session, log_dir=None):
        self.data = data
        self.model = gen_model
        self.flags = flags
        self.batch_size = self.flags.batch_size
        self.sess = session
        self.log_dir = log_dir

        # Initialize saver
        self.saver = tf.compat.v1.train.Saver(max_to_keep=1)
        self._init_gen_variables()

    def _init_gen_variables(self):
        var_list = [var for var in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='pix2pix')]
        self.sess.run(tf.compat.v1.variables_initializer(var_list=var_list))

    def train(self):
        imgs, clses, segs, irises, coordinates = self.data.train_random_batch_include_iris(batch_size=self.batch_size)

        feed = {self.model.img_tfph: imgs,
                self.model.mask_tfph: segs,
                self.model.coord_tfph: coordinates,  # [x, y, h, w]
                self.model.cls_tfph: clses,
                self.model.rate_tfph: 0.5,
                self.model.iden_model.img_tfph: irises,
                self.model.iden_model.train_mode: False}

        self.sess.run(self.model.dis_optim, feed_dict=feed)
        self.sess.run(self.model.gen_optim, feed_dict=feed)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, g_loss, g_adv_loss, g_cond_loss, d_loss, summary = self.sess.run(
            [self.model.gen_optim, self.model.gen_loss, self.model.gen_adv_loss, self.model.cond_loss,
             self.model.dis_loss, self.model.summary_op], feed_dict=feed)

        return g_loss, g_adv_loss, g_cond_loss, d_loss, summary


    # def generate_val_imgs(self, batch_size=10):
    #     print(' [*] Generate validation imgs...')
    #
    #     segs_all = np.zeros((self.data.num_val_imgs, *self.data.input_img_shape), dtype=np.float32)
    #     samples_all = np.zeros((self.data.num_val_imgs, *self.data.output_img_shape), dtype=np.float32)
    #     clses_all = np.zeros((self.data.num_val_imgs, 1), dtype=np.uint8)
    #
    #     for i, index in enumerate(range(0, self.data.num_val_imgs, batch_size)):
    #         # print('[{}/{}] processing...'.format(i + 1, (self.data.num_val_imgs // batch_size)))
    #
    #         img_vals, cls_vals, seg_vals = self.data.direct_batch(batch_size=batch_size, index=index, stage='val')
    #
    #         feed = {
    #             self.model.mask_tfph: seg_vals,
    #             self.model.rate_tfph: 0.5,
    #         }
    #
    #         samples = self.sess.run(self.model.g_sample, feed_dict=feed)
    #
    #         segs_all[index:index + img_vals.shape[0], :, :, :] = seg_vals
    #         samples_all[index:index + img_vals.shape[0], :, :, :] = samples
    #         clses_all[index:index + img_vals.shape[0], :] = cls_vals
    #
    #     return segs_all, samples_all, clses_all


    def generate_test_imgs(self, batch_size=20):
        print(' [*] Generate test imgs...')

        imgs_all = np.zeros((self.data.num_test_imgs, *self.data.output_img_shape), dtype=np.float32)
        segs_all = np.zeros((self.data.num_test_imgs, *self.data.input_img_shape), dtype=np.float32)
        samples_all = np.zeros((self.data.num_test_imgs, *self.data.output_img_shape), dtype=np.float32)
        clses_all = np.zeros((self.data.num_test_imgs, 1), dtype=np.uint8)

        for i, index in enumerate(range(0, self.data.num_test_imgs, batch_size)):
            print('[{}/{}] generating process...'.format(i + 1, (self.data.num_test_imgs // batch_size)))

            img_tests, cls_tests, seg_tests, iris_tests, coord_tests = self.data.direct_batch_include_iris(
                batch_size=batch_size, index=index, stage='test')

            feed = {self.model.mask_tfph: seg_tests,
                    self.model.rate_tfph: 0.5,              # rate: 1 - keep_prob
                    self.model.iden_model.img_tfph: iris_tests,
                    self.model.iden_model.train_mode: False}

            samples = self.sess.run(self.model.g_sample, feed_dict=feed)

            imgs_all[index:index + img_tests.shape[0], :, :, :] = img_tests
            segs_all[index:index + img_tests.shape[0], :, :, :] = seg_tests
            samples_all[index:index + img_tests.shape[0], :, :, :] = samples
            clses_all[index:index + img_tests.shape[0], :] = cls_tests

        return segs_all, samples_all, clses_all, imgs_all


    def eval_identification(self, batch_size=20):
        print(' [*] Evaluate identification...')

        # Initialize/reset the running varaibels.
        self.sess.run(self.model.iden_model.running_vars_initializer)

        for i, index in enumerate(range(0, self.data.num_test_imgs, batch_size)):
            print('[{}/{}] identification process...'.format(i + 1, (self.data.num_test_imgs // batch_size)))

            img_tests, cls_tests, seg_tests, iris_tests, coord_tests = self.data.direct_batch_include_iris(
                batch_size=batch_size, index=index, stage='test')

            feed = {
                self.model.iden_model.img_tfph: iris_tests,
                self.model.iden_model.gt_tfph: cls_tests,
                self.model.iden_model.train_mode: False
            }

            self.sess.run(self.model.iden_model.accuracy_metric_update, feed_dict=feed)

        # Calculate the accuracy
        accuracy = self.sess.run(self.model.iden_model.accuracy_metric)
        print('Iden accuracy: {:.2%}, the accuracy will be smaller than original one due to resizing input image'.format(accuracy))


    def img_sample(self, iter_time, save_dir, batch_size=4):
        imgs, clses, segs, irises, coordinates = self.data.train_random_batch_include_iris(batch_size=batch_size)

        feed = {self.model.mask_tfph: segs,
                self.model.rate_tfph: 0.5,                  # rate: 1 - keep_prob
                self.model.iden_model.img_tfph: irises,
                self.model.iden_model.train_mode: False}

        samples = self.sess.run(self.model.g_sample, feed_dict=feed)
        utils.save_imgs(img_stores=[segs, samples, imgs], iter_time=iter_time, save_dir=save_dir, is_vertical=True)

    # def set_best_acc(self, best_acc):
    #     self.sess.run(self.model.assign_best_acc, feed_dict={self.model.best_acc_tfph: best_acc})

    # def get_best_acc(self):
    #     return self.sess.run(self.model.best_acc)


    def save_model(self, logger, model_dir, iter_time):
        self.saver.save(self.sess, os.path.join(model_dir, 'model'), global_step=iter_time)
        logger.info('[*] Model saved! Iter: {}'.format(iter_time))


    def load_model(self, logger, model_dir, is_train=False):
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

            return True, iter_time + 1
        else:
            return False, None, None


# # Evaluator
# class Evaluator(object):
#     def __init__(self, flags, batch_size=128, model_dir=None, log_dir=None):
#         # Initialize dataset
#         self.data = ii_dataset.Dataset(name='identification', mode=1, is_train=flags.is_train,
#                                        log_dir=log_dir, is_debug=False)
#
#         self.batch_size = batch_size
#         self.model_dir = model_dir
#
#         self._init_session()
#         self._init_variables()
#
#     def _init_session(self):
#         self.graph = tf.Graph()
#         with self.graph.as_default():
#             self.model = ResNet18(input_img_shape=self.data.input_img_shape,
#                                   num_classes=self.data.num_identities,
#                                   is_train=False)
#
#         self.sess = tf.compat.v1.Session(graph=self.graph)
#
#     def _init_variables(self):
#         with self.sess.as_default():
#             with self.graph.as_default():
#                 self.sess.run(tf.compat.v1.global_variables_initializer())
#
#                 if self.model_dir is not None:
#                     flag, iter_time = self.load_model(os.path.join('../model/identification', self.model_dir))
#
#                     if flag is True:
#                         print(' [!] Loadd IdenModel Success! Iter: {}'.format(iter_time))
#                     else:
#                         exit(' [!] Failed to restore IdenModel {}'.format(self.load_model))
#
#     def eval_val(self, tb_writer, iter_time, segs_all, imgs_all, clses_all, batch_size=10):
#         print(' [*] Evaluate on the validation dataset...')
#
#         # Initialize/reset the running varaibels.
#         self.sess.run(self.model.running_vars_initializer)
#
#         for i, index in enumerate(range(0, self.data.num_val_imgs, batch_size)):
#             # print('[{}/{}] processing...'.format(i + 1, (self.data.num_val_imgs // batch_size)))
#
#             if index + batch_size < self.data.num_val_imgs:
#                 imgs = imgs_all[index:index + batch_size, :, :, :]
#                 segs = segs_all[index:index + batch_size, :, :, :]
#                 batch_clses = clses_all[index:index + batch_size, :]
#             else:
#                 imgs = imgs_all[index:, :, :, :]
#                 segs = segs_all[index:, :, :, :]
#                 batch_clses = clses_all[index:, :]
#
#             batch_imgs = utils.extract_iris(imgs, segs)
#             feed = {
#                 self.model.img_tfph: batch_imgs,
#                 self.model.gt_tfph: batch_clses,
#                 self.model.train_mode: False
#             }
#
#             self.sess.run(self.model.accuracy_metric_update, feed_dict=feed)
#
#         # Calculate the accuracy
#         accuracy, metric_summary_op = self.sess.run([self.model.accuracy_metric, self.model.metric_summary_op])
#
#         # Write to tensorabrd
#         tb_writer.add_summary(metric_summary_op, iter_time)
#         tb_writer.flush()
#
#         return accuracy * 100.
#
#     def eval_test(self, segs_all, imgs_all, clses_all, batch_size=20):
#         print(' [*] Evaluate on the test dataset...')
#
#         # Initialize/reset the running varaibels.
#         self.sess.run(self.model.running_vars_initializer)
#
#         for i, index in enumerate(range(0, self.data.num_test_imgs, batch_size)):
#             print('[{}/{}] identification process...'.format(i + 1, (self.data.num_test_imgs // batch_size)))
#
#             if index + batch_size < self.data.num_test_imgs:
#                 imgs = imgs_all[index:index + batch_size, :, :, :]
#                 segs = segs_all[index:index + batch_size, :, :, :]
#                 batch_clses = clses_all[index:index + batch_size, :]
#             else:
#                 imgs = imgs_all[index:, :, :, :]
#                 segs = segs_all[index:, :, :, :]
#                 batch_clses = clses_all[index:, :]
#
#             batch_imgs = utils.extract_iris(imgs, segs)
#             feed = {
#                 self.model.img_tfph: batch_imgs,
#                 self.model.gt_tfph: batch_clses,
#                 self.model.train_mode: False
#             }
#
#             self.sess.run(self.model.accuracy_metric_update, feed_dict=feed)
#
#         # Calculate the accuracy
#         accuracy, metric_summary_op = self.sess.run([self.model.accuracy_metric, self.model.metric_summary_op])
#
#         return accuracy * 100.
#
#     def test_top_k(self, segs_all, imgs_all, clses_all, log_dir, batch_size=20):
#         preds = np.zeros((self.data.num_test_imgs, self.data.num_identities), np.float32)
#
#         for i, index in enumerate(range(0, self.data.num_test_imgs, batch_size)):
#             print('[{}/{}] processing...'.format(i + 1, (self.data.num_test_imgs // batch_size) + 1))
#
#             if index + batch_size < self.data.num_test_imgs:
#                 imgs = imgs_all[index:index + batch_size, :, :, :]
#                 segs = segs_all[index:index + batch_size, :, :, :]
#                 batch_clses = clses_all[index:index + batch_size, :]
#             else:
#                 imgs = imgs_all[index:, :, :, :]
#                 segs = segs_all[index:, :, :, :]
#                 batch_clses = clses_all[index:, :]
#
#             num_imgs = imgs.shape[0]
#             batch_imgs = utils.extract_iris(imgs, segs)
#
#             feed = {
#                 self.model.img_tfph: batch_imgs,
#                 self.model.gt_tfph: batch_clses,
#                 self.model.train_mode: False
#             }
#
#             preds[index:index + num_imgs, :] = self.sess.run(self.model.preds, feed_dict=feed)
#
#         np.savetxt(os.path.join(log_dir, 'preds.csv'), preds, delimiter=",")
#
#     def load_model(self, model_dir):
#         print(' [*] Reading identification checkpoint...')
#
#         saver = tf.compat.v1.train.Saver(max_to_keep=1)
#         ckpt = tf.train.get_checkpoint_state(model_dir)
#
#         if ckpt and ckpt.model_checkpoint_path:
#             ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
#             saver.restore(self.sess, os.path.join(model_dir, ckpt_name))
#
#             meta_graph_path = ckpt.model_checkpoint_path + '.meta'
#             iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])
#
#             return True, iter_time
#         else:
#             return False, None


