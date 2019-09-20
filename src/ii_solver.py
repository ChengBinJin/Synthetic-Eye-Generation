# --------------------------------------------------------------------------
# Tensorflow Implementation of Synthetic Eye Generation Challenge
# Iris Identification
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import numpy as np
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
        img_trains, cls_trains = self.data.train_random_batch(batch_size=self.batch_size)

        feed = {
            self.model.img_tfph: img_trains,
            self.model.gt_tfph: cls_trains,
            self.model.train_mode: True
        }

        train_op = self.model.train_op
        total_loss_op = self.model.total_loss
        data_loss_op = self.model.data_loss
        reg_term_op = self.model.reg_term
        batch_acc_op = self.model.batch_acc
        summary_op = self.model.summary_op

        _, total_loss, data_loss, reg_term, batch_acc, summary = self.sess.run(
            [train_op, total_loss_op, data_loss_op, reg_term_op, batch_acc_op, summary_op], feed_dict=feed)

        return total_loss, data_loss, reg_term, batch_acc, summary

    def eval(self, tb_writer, iter_time):
        print(' [*] Evaluate on the validation dataset...')

        # Initialize/reset the running variables
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

        # Write to tensorboard
        tb_writer.add_summary(metric_summary_op, iter_time)
        tb_writer.flush()

        return accuracy * 100.

    def test(self):
        stage_list = ['train', 'val', 'test']
        num_imgs_list = [self.data.num_train_imgs, self.data.num_val_imgs, self.data.num_test_imgs]
        accuracy = np.zeros(len(stage_list), np.float32)

        for i, (stage, num_imgs) in enumerate(zip(stage_list, num_imgs_list)):
            print(' [*] Evaluate on the {} dataset...'.format(stage))

            # Initialize/reset the running variables
            self.sess.run(self.model.running_vars_initializer)

            for j, index in enumerate(range(0, num_imgs, self.batch_size)):
                print('[{}/{}] processing...'.format(j + 1, (num_imgs // self.batch_size) + 1))

                img_vals, cls_vals = self.data.direct_batch(batch_size=self.batch_size, index=index, stage=stage)

                feed = {
                    self.model.img_tfph: img_vals,
                    self.model.gt_tfph: cls_vals,
                    self.model.train_mode: False
                }

                # self.sess.run(self.model.accuracy_metric_update, feed_dict=feed)
                self.sess.run(self.model.accuracy_metric_update, feed_dict=feed)

            # Calculate the accuracy
            accuracy[i] = self.sess.run(self.model.accuracy_metric) * 100.

        return accuracy

    def set_best_acc(self, best_acc):
        self.sess.run(self.model.assign_best_acc, feed_dict={self.model.best_acc_tfph: best_acc})

    def get_best_acc(self):
        return self.sess.run(self.model.best_acc)
