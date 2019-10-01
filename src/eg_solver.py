# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Synthetic Eye Generation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------

import tensorflow as tf


class Solver(object):
    def __init__(self, model, data, batch_size=1):
        self.model = model
        self.data = data
        self.batch_size = batch_size

        self._init_session()
        self._init_variables()

    def _init_session(self):
        self.sess = tf.compat.v1.Session()

    def _init_variables(self):
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def train(self):
        def feed_run():
            imgs, _, segs = self.data.gen_train_random_batch(batch_size=self.batch_size)
            feed = {self.model.img_tfph: imgs,
                    self.model.mask_tfph: segs,
                    self.model.rate_tfph: 0.5}
            return feed

        self.sess.run(self.model.dis_optim, feed_dict=feed_run())
        self.sess.run(self.model.gen_optim, feed_dict=feed_run())

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        # _, g_loss, g_adv_loss, g_cond_loss, d_loss, summary = self.sess.run(
        #     [self.model.gen_optim, self.model.gen_loss, self.model.gen_adv_loss, self.model.cond_loss,
        #      self.model.dis_loss, self.model.summary_op], feed_dict=feed_run())

        _, g_loss, g_adv_loss, g_cond_loss, d_loss = self.sess.run(
            [self.model.gen_optim, self.model.gen_loss, self.model.gen_adv_loss, self.model.cond_loss,
             self.model.dis_loss], feed_dict=feed_run())

        return g_loss, g_adv_loss, g_cond_loss, d_loss

