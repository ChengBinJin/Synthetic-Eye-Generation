# --------------------------------------------------------------------------
# Tensorflow Implementation of Synthetic Eye Generation Challenge
# Iris Identification
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import tensorflow as tf

class Solver(object):
    def __init__(self, model, data, batch_size=2):
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
        img_trains, cls_trains = self.data.train_batch(batch_size=self.batch_size)

        feed = {
            self.model.img_tfph: img_trains,
            self.model.gt_tfph: cls_trains,
            self.model.train_mode: True
        }

        train_op = self.model.train_op
        total_loss_op = self.model.total_loss
        data_loss_op = self.model.data_loss
        reg_term_op = self.model.reg_term
        summary_op = self.model.summary_op

        _, total_loss, data_loss, reg_term, summary = self.sess.run(
            [train_op, total_loss_op, data_loss_op, reg_term_op, summary_op], feed_dict=feed)

        return total_loss, data_loss, reg_term, summary

    def eval(self, tb_writer, iter_time):
        # Initialize/reset the running variables
        self.sess.run(self.model.running_vars_initializer)

        for i in range(0, self.data.num_val_imgs, self.batch_size):
            img_vals, cls_vals = self.data.val_batch(batch_size=self.batch_size, index=i)

            feed = {
                self.model.img_tfph: img_vals,
                self.model.gt_tfph: cls_vals,
                self.model.train_mode: False
            }

            self.sess.run(self.model.accuracy_metric_update, feed_dict=feed)

        # Calculate the accuracy
        accuracy, metric_summary_op = self.sess.run([self.model.accuracy_metric, self.model.metric_summary_op])

        # Write to tensorboard
        tb_writer.add_summary(metric_summary_op, iter_time)
        tb_writer.flush()

        return accuracy * 100.
