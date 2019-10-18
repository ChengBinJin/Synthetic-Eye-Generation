# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Synthetic Eye Generation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import os
import logging
import tensorflow as tf

import utils as utils
from resnet import ResNet18
import tensorflow_utils as tf_utils


class Pix2pix(object):
    def __init__(self, input_img_shape=(320, 200, 1), gen_mode=1, iden_model_dir=None, session=None, lr=2e-4, total_iters=2e5,
                 is_train=True, log_dir=None, lambda_1=100., num_class=4, num_identities=122, name='pix2pix'):
        self.input_img_shape = input_img_shape
        self.output_img_shape = (*self.input_img_shape[0:2], 1)
        self.iris_shape = (200, 200, 1)
        self.gen_mode = gen_mode
        self.iden_model_dir = iden_model_dir
        self.sess = session
        self.is_train = is_train
        self.labmda_1 = lambda_1
        self.num_class = num_class
        self.num_identities = num_identities
        self.gen_c = [64, 128, 256, 512, 512, 512, 512, 512,
                      512, 512, 512, 512, 256, 128, 64, self.output_img_shape[2]]
        self.dis_c = [64, 128, 256, 512, 1]

        self.lr = lr
        self.total_steps = total_iters
        self.start_decay_step = int(self.total_steps * 0.5)
        self.decay_steps = self.total_steps - self.start_decay_step
        self.log_dir = log_dir
        self.name = name
        self.tb_lr = None
        self.iris_value = tf.reshape(tf.constant([204.], dtype=tf.float32), shape=(1, 1, 1, 1))

        self.logger = logging.getLogger(__name__)  # logger
        self.logger.setLevel(logging.INFO)
        utils.init_logger(logger=self.logger, log_dir=self.log_dir, is_train=self.is_train, name=self.name)

        # Initialize the identification network
        if self.gen_mode == 4:
            self._init_iden_model()

        self._build_graph()         # main graph
        self._best_metrics_record()
        self._init_tensorboard()    # tensorboard
        tf_utils.show_all_variables(logger=self.logger if self.is_train else None)

    def _init_iden_model(self):
        self.iden_model = ResNet18(input_img_shape=self.iris_shape,
                                   num_classes=self.num_identities,
                                   is_train=False)

        flag, iter_time = self.load_iden_model(os.path.join('../model/identification', self.iden_model_dir))
        if flag is True:
            print(' [!] Loadd IdenModel Success! Iter: {}'.format(iter_time))
        else:
            exit(' [!] Failed to restore IdenModel {}'.format(self.load_iden_model))

    def load_iden_model(self, model_dir):
        print(' [*] Reading {} checkpoint...'.format(model_dir))

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

    def _build_graph(self):
        self.top_scope = tf.get_variable_scope()  # top-level scope

        with tf.compat.v1.variable_scope(self.name):
            self.mask_tfph = tf.compat.v1.placeholder(tf.float32, shape=[None, *self.input_img_shape], name='mask_tfph')
            self.img_tfph = tf.compat.v1.placeholder(tf.float32, shape=[None, *self.output_img_shape], name='img_tfph')
            self.rate_tfph = tf.compat.v1.placeholder(tf.float32, name='keep_prob_ph')
            self.cls_tfph = tf.compat.v1.placeholder(dtype=tf.dtypes.int64, shape=[None, 1])

            # Initialize generator & discriminator
            self.gen_obj = Generator(name='G', gen_c=self.gen_c, norm='instance',
                                     logger=self.logger, _ops=None)
            self.dis_obj = Discriminator(name='D', dis_c=self.dis_c, norm='instance',
                                         logger=self.logger,  _ops=None)

            # Transform img_train and seg_img_train
            input_mask = self.transform_img(self.mask_tfph)
            real_img = self.transform_img(self.img_tfph)

            # Generator
            if self.gen_mode == 4:
                fake_img = self.gen_obj(input_mask, self.rate_tfph, self.iden_model.feat, gen_mode=self.gen_mode)
            else:
                fake_img = self.gen_obj(input_mask, self.rate_tfph)

            # Concatenation
            self.g_sample = self.inv_transform_img(fake_img)
            self.real_pair = tf.concat([input_mask, real_img], axis=3)
            self.fake_pair = tf.concat([input_mask, fake_img], axis=3)

            # Define generator loss
            self.gen_adv_loss = self.generator_loss(self.dis_obj, self.fake_pair)
            self.cond_loss = self.conditional_loss(pred=fake_img, gt=real_img, mask=self.mask_tfph)
            self.gen_loss = self.gen_adv_loss + self.cond_loss

            # Define discriminator loss
            self.dis_loss = self.discriminator_loss(self.dis_obj, self.real_pair, self.fake_pair)

            # Optimizers
            self.gen_optim = self.init_optimizer(loss=self.gen_loss, variables=self.gen_obj.variables, name='Adam_gen')
            self.dis_optim = self.init_optimizer(loss=self.dis_loss, variables=self.dis_obj.variables, name='Adam_dis')

    def _best_metrics_record(self):
        self.best_acc_tfph = tf.compat.v1.placeholder(tf.float32, name='best_acc')

        # Best accuracy variable
        self.best_acc = tf.compat.v1.get_variable(name='best_acc', dtype=tf.float32, initializer=tf.constant(0.),
                                                  trainable=False)
        self.assign_best_acc = tf.compat.v1.assign(self.best_acc, value=self.best_acc_tfph)

    def init_optimizer(self, loss, variables, name='Adam'):
        with tf.compat.v1.variable_scope(name):
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.lr
            end_leanring_rate = 0.
            start_decay_step = self.start_decay_step
            decay_steps = self.decay_steps

            learning_rate = (tf.where(tf.greater_equal(global_step, start_decay_step),
                                      tf.compat.v1.train.polynomial_decay(starter_learning_rate,
                                                                          global_step - start_decay_step,
                                                                          decay_steps, end_leanring_rate, power=1.0),
                                      starter_learning_rate))

            self.tb_lr = tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            learn_step = tf.compat.v1.train.AdamOptimizer(learning_rate, beta1=0.5).\
                minimize(loss, global_step=global_step, var_list=variables)

            return learn_step

    @staticmethod
    def generator_loss(dis_obj, fake_img):
        d_logit_fake = dis_obj(fake_img)
        loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake ,
                                                                           labels=tf.ones_like(d_logit_fake)))
        return loss

    def conditional_loss(self, pred, gt, mask=None):
        loss = tf.constant(0., dtype=tf.float32)

        if self.gen_mode == 2:      # whole-constraint
            cond_loss = tf.math.reduce_mean(tf.math.abs(pred - gt))
            loss = self.labmda_1 * cond_loss
        elif self.gen_mode == 3:    # iris-constraint
            iris_region = self.extract_iris_region(mask)
            cond_loss = tf.math.reduce_mean(tf.math.abs(iris_region * (pred - gt)))
            loss = self.labmda_1 * cond_loss
        elif self.gen_mode == 4:    # cls-constraint
            with tf.compat.v1.variable_scope(self.top_scope):  # resets the current scope!
                cls_preds, _ = self.iden_model.forward_network(input_img=pred, reuse=True)

            loss = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=cls_preds,
                labels=self.convert_one_hot(self.cls_tfph)))

        return loss

    def extract_iris_region(self, seg_mask):
        mask = tf.math.reduce_sum(tf.dtypes.cast(tf.math.equal(seg_mask, self.iris_value), dtype=tf.float32),
                                  axis=-1, keepdims=True)
        return mask

    @staticmethod
    def discriminator_loss(dis_obj, real_img, fake_img):
        d_logit_real = dis_obj(real_img)
        d_logit_fake = dis_obj(fake_img)

        error_real = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
        error_fake = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))

        loss = 0.5 * (error_real + error_fake)
        return loss

    def _init_tensorboard(self):
        self.tb_gen_loss = tf.compat.v1.summary.scalar('loss/G_loss', self.gen_loss)
        self.tb_adv_loss = tf.compat.v1.summary.scalar('loss/adv_loss', self.gen_adv_loss)
        self.tb_cond_loss = tf.compat.v1.summary.scalar('loss/cond_loss', self.cond_loss)
        self.tb_dis_loss = tf.compat.v1.summary.scalar('loss/D_lss', self.dis_loss)
        self.summary_op = tf.compat.v1.summary.merge(
            inputs=[self.tb_gen_loss, self.tb_adv_loss, self.tb_cond_loss, self.tb_dis_loss, self.tb_lr])

    @staticmethod
    def transform_img(img):
        return img / 127.5 - 1.

    @staticmethod
    def inv_transform_img(img):
        img = (img + 1.) * 127.5
        return img

    def convert_one_hot(self, data):
        data = tf.dtypes.cast(data, dtype=tf.uint8)
        data = tf.one_hot(data, depth=self.num_identities, name='one_hot')
        data = tf.reshape(data, shape=[-1, self.num_identities])
        return data


class Generator(object):
    def __init__(self, name=None, gen_c=None, norm='instance', logger=None, _ops=None):
        self.name = name
        self.gen_c = gen_c
        self.norm = norm
        self.logger = logger
        self._ops = _ops
        self.reuse = False

    def __call__(self, x, keep_rate=0.5, iris_feat=None, gen_mode=0):
        with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x, logger=self.logger)

            # E0: (320, 200) -> (160, 100)
            e0_conv2d = tf_utils.conv2d(x, output_dim=self.gen_c[0], initializer='He', logger=self.logger,
                                        name='e0_conv2d')
            e0_lrelu = tf_utils.lrelu(e0_conv2d, logger=self.logger, name='e0_lrelu')

            # E1: (160, 100) -> (80, 50)
            e1_conv2d = tf_utils.conv2d(e0_lrelu, output_dim=self.gen_c[1], initializer='He', logger=self.logger,
                                        name='e1_conv2d')
            e1_batchnorm = tf_utils.norm(e1_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='e1_norm')
            e1_lrelu = tf_utils.lrelu(e1_batchnorm, logger=self.logger, name='e1_lrelu')

            # E2: (80, 50) -> (40, 25)
            e2_conv2d = tf_utils.conv2d(e1_lrelu, output_dim=self.gen_c[2], initializer='He', logger=self.logger,
                                        name='e2_conv2d')
            e2_batchnorm = tf_utils.norm(e2_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='e2_norm')
            e2_lrelu = tf_utils.lrelu(e2_batchnorm, logger=self.logger, name='e2_lrelu')

            # E3: (40, 25) -> (20, 13)
            e3_conv2d = tf_utils.conv2d(e2_lrelu, output_dim=self.gen_c[3], initializer='He', logger=self.logger,
                                        name='e3_conv2d')
            e3_batchnorm = tf_utils.norm(e3_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='e3_norm')
            e3_lrelu = tf_utils.lrelu(e3_batchnorm, logger=self.logger, name='e3_lrelu')

            # E4: (20, 13) -> (10, 7)
            e4_conv2d = tf_utils.conv2d(e3_lrelu, output_dim=self.gen_c[4], initializer='He', logger=self.logger,
                                        name='e4_conv2d')
            e4_batchnorm = tf_utils.norm(e4_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='e4_norm')
            e4_lrelu = tf_utils.lrelu(e4_batchnorm, logger=self.logger, name='e4_lrelu')

            # E5: (10, 7) -> (5, 4)
            e5_conv2d = tf_utils.conv2d(e4_lrelu, output_dim=self.gen_c[5], initializer='He', logger=self.logger,
                                        name='e5_conv2d')
            e5_batchnorm = tf_utils.norm(e5_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='e5_norm')
            e5_lrelu = tf_utils.lrelu(e5_batchnorm, logger=self.logger, name='e5_lrelu')

            # E6: (5, 4) -> (3, 2)
            e6_conv2d = tf_utils.conv2d(e5_lrelu, output_dim=self.gen_c[6], initializer='He', logger=self.logger,
                                        name='e6_conv2d')
            e6_batchnorm = tf_utils.norm(e6_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='e6_norm')
            e6_lrelu = tf_utils.lrelu(e6_batchnorm, logger=self.logger, name='e6_lrelu')

            # E7: (3, 2) -> (2, 1)
            e7_conv2d = tf_utils.conv2d(e6_lrelu, output_dim=self.gen_c[7], initializer='He', logger=self.logger,
                                        name='e7_conv2d')
            e7_batchnorm = tf_utils.norm(e7_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='e7_norm')
            e7_relu = tf_utils.lrelu(e7_batchnorm, logger=self.logger, name='e7_relu')

            # ID preserving feature
            if gen_mode == 4:
                iris_feat = tf.reshape(iris_feat, shape=(-1, 1, 1, iris_feat.get_shape()[-1]), name='iris_feat')
                iris_feat = tf.concat(values=[iris_feat, iris_feat], axis=1, name='iris_feat_concat')
                e7_relu = tf.concat([e7_relu, iris_feat], axis=3, name='id_preserving')
                tf_utils.print_activations(e7_relu)

            # D0: (2, 1) -> (3, 2)
            # Stage1: (2, 1) -> (4, 2)
            d0_deconv = tf_utils.deconv2d(e7_relu, output_dim=self.gen_c[8], initializer='He', logger=self.logger,
                                          name='d0_deconv2d')
            # Stage2: (4, 2) -> (3, 2)
            shapeA = e6_conv2d.get_shape().as_list()[1]
            shapeB = d0_deconv.get_shape().as_list()[1] - e6_conv2d.get_shape().as_list()[1]
            d0_split, _ = tf.split(d0_deconv, [shapeA, shapeB], axis=1, name='d0_split')
            tf_utils.print_activations(d0_split, logger=self.logger)
            # Stage3: Batch norm, concatenation, and relu
            d0_batchnorm = tf_utils.norm(d0_split, _type=self.norm, _ops=self._ops, logger=self.logger, name='d0_norm')
            d0_drop = tf_utils.dropout(d0_batchnorm, keep_prob=keep_rate, logger=self.logger, name='d0_dropout')
            d0_concat = tf.concat([d0_drop, e6_batchnorm], axis=3, name='d0_concat')
            d0_relu = tf_utils.relu(d0_concat, logger=self.logger, name='d0_relu')

            # D1: (3, 2) -> (5, 4)
            # Stage1: (3, 2) -> (6, 4)
            d1_deconv = tf_utils.deconv2d(d0_relu, output_dim=self.gen_c[9], initializer='He', logger=self.logger,
                                          name='d1_deconv2d')
            # Stage2: (6, 4) -> (5, 4)
            shapeA = e5_batchnorm.get_shape().as_list()[1]
            shapeB = d1_deconv.get_shape().as_list()[1] - e5_batchnorm.get_shape().as_list()[1]
            d1_split, _ = tf.split(d1_deconv, [shapeA, shapeB], axis=1, name='d1_split')
            tf_utils.print_activations(d1_split, logger=self.logger)
            # Stage3: Batch norm, concatenation, and relu
            d1_batchnorm = tf_utils.norm(d1_split, _type=self.norm, _ops=self._ops, logger=self.logger, name='d1_norm')
            d1_drop = tf_utils.dropout(d1_batchnorm, keep_prob=keep_rate, logger=self.logger, name='d1_dropout')
            d1_concat = tf.concat([d1_drop, e5_batchnorm], axis=3, name='d1_concat')
            d1_relu = tf_utils.relu(d1_concat, logger=self.logger, name='d1_relu')

            # D2: (5, 4) -> (10, 7)
            # Stage1: (5, 4) -> (10, 8)
            d2_deconv = tf_utils.deconv2d(d1_relu, output_dim=self.gen_c[10], initializer='He', logger=self.logger,
                                          name='d2_deconv2d')
            # Stage2: (10, 8) -> (10, 7)
            shapeA = e4_batchnorm.get_shape().as_list()[2]
            shapeB = d2_deconv.get_shape().as_list()[2] - e4_batchnorm.get_shape().as_list()[2]
            d2_split, _ = tf.split(d2_deconv, [shapeA, shapeB], axis=2, name='d2_split')
            tf_utils.print_activations(d2_split, logger=self.logger)
            # Stage3: Batch norm, concatenation, and relu
            d2_batchnorm = tf_utils.norm(d2_split, _type=self.norm, _ops=self._ops, logger=self.logger, name='d2_norm')
            d2_drop = tf_utils.dropout(d2_batchnorm, keep_prob=keep_rate, logger=self.logger, name='d2_dropout')
            d2_concat = tf.concat([d2_drop, e4_batchnorm], axis=3, name='d2_concat')
            d2_relu = tf_utils.relu(d2_concat, logger=self.logger, name='d2_relu')

            # D3: (10, 7) -> (20, 13)
            # Stage1: (10, 7) -> (20, 14)
            d3_deconv = tf_utils.deconv2d(d2_relu, output_dim=self.gen_c[11], initializer='He', logger=self.logger,
                                          name='d3_deconv2d')
            # Stage2: (20, 14) -> (20, 13)
            shapeA = e3_batchnorm.get_shape().as_list()[2]
            shapeB = d3_deconv.get_shape().as_list()[2] - e3_batchnorm.get_shape().as_list()[2]
            d3_split, _ = tf.split(d3_deconv, [shapeA, shapeB], axis=2, name='d3_split_2')
            tf_utils.print_activations(d3_split, logger=self.logger)
            # Stage3: Batch norm, concatenation, and relu
            d3_batchnorm = tf_utils.norm(d3_split, _type=self.norm, _ops=self._ops, logger=self.logger, name='d3_norm')
            d3_concat = tf.concat([d3_batchnorm, e3_batchnorm], axis=3, name='d3_concat')
            d3_relu = tf_utils.relu(d3_concat, logger=self.logger, name='d3_relu')

            # D4: (20, 13) -> (40, 25)
            # Stage1: (20, 13) -> (40, 26)
            d4_deconv = tf_utils.deconv2d(d3_relu, output_dim=self.gen_c[12], initializer='He', logger=self.logger,
                                          name='d4_deconv2d')
            # Stage2: (40, 26) -> (40, 25)
            shapeA = e2_batchnorm.get_shape().as_list()[2]
            shapeB = d4_deconv.get_shape().as_list()[2] - e2_batchnorm.get_shape().as_list()[2]
            d4_split, _ = tf.split(d4_deconv, [shapeA, shapeB], axis=2, name='d4_split')
            tf_utils.print_activations(d4_split, logger=self.logger)
            # Stage3: Batch norm, concatenation, and relu
            d4_batchnorm = tf_utils.norm(d4_split, _type=self.norm, _ops=self._ops, logger=self.logger, name='d4_norm')
            d4_concat = tf.concat([d4_batchnorm, e2_batchnorm], axis=3, name='d4_concat')
            d4_relu = tf_utils.relu(d4_concat, logger=self.logger, name='d4_relu')

            # D5: (40, 25, 256) -> (80, 50, 128)
            d5_deconv = tf_utils.deconv2d(d4_relu, output_dim=self.gen_c[13], initializer='He', logger=self.logger,
                                          name='d5_deconv2d')
            d5_batchnorm = tf_utils.norm(d5_deconv, _type=self.norm, _ops=self._ops, logger=self.logger, name='d5_norm')
            d5_concat = tf.concat([d5_batchnorm, e1_batchnorm], axis=3, name='d5_concat')
            d5_relu = tf_utils.relu(d5_concat, logger=self.logger, name='d5_relu')

            # D6: (80, 50, 128) -> (160, 100, 64)
            d6_deconv = tf_utils.deconv2d(d5_relu, output_dim=self.gen_c[14], initializer='He', logger=self.logger,
                                          name='d6_deconv2d')
            d6_batchnorm = tf_utils.norm(d6_deconv, _type=self.norm, _ops=self._ops, logger=self.logger, name='d6_norm')
            d6_concat = tf.concat([d6_batchnorm, e0_conv2d], axis=3, name='d6_concat')
            d6_relu = tf_utils.relu(d6_concat, logger=self.logger, name='d6_relu')

            # D7: (160, 100, 64) -> (320, 200, 1)
            d7_deconv = tf_utils.deconv2d(d6_relu, output_dim=self.gen_c[15], initializer='He', logger=self.logger,
                                          name='d7_deconv2d')
            output = tf_utils.tanh(d7_deconv, logger=self.logger, name='output_tanh')

            # Set reuse=True for next call
            self.reuse = True
            self.variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='pix2pix/'+self.name)

        return output


class Discriminator(object):
    def __init__(self, name=None, dis_c=None, norm='instance', logger=None, _ops=None):
        self.name = name
        self.dis_c = dis_c
        self.norm = norm
        self.logger = logger
        self._ops = _ops
        self.reuse = False

    def __call__(self, x):
        with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x, logger=self.logger)

            # H1: (320, 200) -> (160, 100)
            h0_conv2d = tf_utils.conv2d(x, output_dim=self.dis_c[0], initializer='He', logger=self.logger,
                                        name='h0_conv2d')
            h0_lrelu = tf_utils.lrelu(h0_conv2d, logger=self.logger, name='h0_lrelu')

            # H2: (160, 100) -> (80, 50)
            h1_conv2d = tf_utils.conv2d(h0_lrelu, output_dim=self.dis_c[1], initializer='He', logger=self.logger,
                                        name='h1_conv2d')
            h1_norm = tf_utils.norm(h1_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='h1_norm')
            h1_lrelu = tf_utils.lrelu(h1_norm, logger=self.logger, name='h1_lrelu')

            # H3: (80, 50) -> (40, 25)
            h2_conv2d = tf_utils.conv2d(h1_lrelu, output_dim=self.dis_c[2], initializer='He', logger=self.logger,
                                        name='h2_conv2d')
            h2_norm = tf_utils.norm(h2_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='h2_norm')
            h2_lrelu = tf_utils.lrelu(h2_norm, logger=self.logger, name='h2_lrelu')

            # H4: (40, 25) -> (20, 13)
            h3_conv2d = tf_utils.conv2d(h2_lrelu, output_dim=self.dis_c[3], initializer='He', logger=self.logger,
                                        name='h3_conv2d')
            h3_norm = tf_utils.norm(h3_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='h3_norm')
            h3_lrelu = tf_utils.lrelu(h3_norm, logger=self.logger, name='h3_lrelu')

            # H5: (20, 13) -> (20, 13)
            output = tf_utils.conv2d(h3_lrelu, output_dim=self.dis_c[4], k_h=3, k_w=3, d_h=1, d_w=1,
                                     initializer='He', logger=self.logger, name='output_conv2d')

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='pix2pix/'+self.name)

        return output



