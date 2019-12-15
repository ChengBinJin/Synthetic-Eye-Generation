# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Synthetic Eye Generation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import os
import cv2
import numpy as np
import tensorflow as tf

from resnet import ResNet18
from utils import all_files_under, extract_iris

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')


def main(_, num_features=512, num_objects=122, target_examples=10, h=320, w=200, batch_size=512):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_index

    # read real, mask, and fake imgs
    real_imgs, mask_imgs, fake_imgs = read_imgs(num_objects, target_examples, h, w)

    # Initialize model
    model = ResNet18(input_img_shape=(w, w, 1),
                     num_classes=num_objects,
                     is_train=False)

    # Initialize session
    sess = tf.compat.v1.Session()

    # Initilize identification network
    load_model(sess)

    total_examples = target_examples * num_objects
    real_features = np.zeros((total_examples, num_features), dtype=np.float32)
    fake_features = np.zeros((total_examples, num_features), dtype=np.float32)

    for i, index in enumerate(range(0, total_examples, batch_size)):
        print('[{}/{}] Extracting features...'.format(i, (total_examples // batch_size)))

        if index + batch_size < total_examples:
            batch_real_imgs = real_imgs[index:index + batch_size, :, :, :]
            batch_fake_imgs = fake_imgs[index:index + batch_size, :, :, :]
            batch_mask_imgs = mask_imgs[index:index + batch_size, :, :, :]
        else:
            batch_real_imgs = real_imgs[index:, :, :, :]
            batch_fake_imgs = fake_imgs[index:, :, :, :]
            batch_mask_imgs = mask_imgs[index:, :, :, :]

        num_imgs = batch_real_imgs.shape[0]

        # Extract real iris features
        batch_real_iris, _ = extract_iris(batch_real_imgs, batch_mask_imgs)
        real_feed = {
            model.img_tfph: batch_real_iris,
            model.train_mode: False
        }
        real_features[index:index + num_imgs, :] = sess.run(model.feat, feed_dict=real_feed)

        # Extract fake iris features
        batch_fake_iris, _ = extract_iris(batch_fake_imgs, batch_mask_imgs)
        fake_feed = {
            model.img_tfph: batch_fake_iris,
            model.train_mode: False
        }
        fake_features[index:index + num_imgs, :] = sess.run(model.feat, feed_dict=fake_feed)

    np.save('./data/real_features', real_features)
    np.save('./data/fake_features', fake_features)


def load_model(sess, model_dir='../model/identification/20190921-111742'):
    # Initialize saver
    saver = tf.compat.v1.train.Saver(max_to_keep=1)

    print(' [*] Reading checkpoint...')

    ckpt = tf.train.get_checkpoint_state(model_dir)
    iter_time = 0
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(model_dir, ckpt_name))

        meta_graph_path = ckpt.model_checkpoint_path + '.meta'
        iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

        flag = True
    else:
        flag = False

    if flag is True:
        print(' [!] Load Success! Iter: {}'.format(iter_time))
    else:
        exit(' [!] Failed to restore model')




def read_imgs(num_objects, target_examples, h, w, data_folder='../test/generation/20191009-091833', num_examples=20):
    target_img_names = list()
    all_img_names = all_files_under(folder=data_folder, subfolder=None, endswith='.png')

    # Extract 10 examples for each object
    for i, img_name in enumerate(all_img_names):
        if i % num_examples < target_examples:
            target_img_names.append(img_name)

    # Read real, mask, and fake imgs
    real_imgs = np.zeros((target_examples * num_objects, h, w, 1), dtype=np.uint8)
    mask_imgs = np.zeros((target_examples * num_objects, h, w, 3), dtype=np.uint8)
    fake_imgs = np.zeros((target_examples * num_objects, h, w, 1), dtype=np.uint8)

    for i, img_name in enumerate(target_img_names):
        img = cv2.imread(img_name)
        real_img = img[:, :w, 1]
        mask_img = img[:, w:2*w, :]
        fake_img = img[:, -w:, 1]

        real_imgs[i, :, :, :] = np.expand_dims(real_img, axis=-1)
        mask_imgs[i, :, :, :] = mask_img
        fake_imgs[i, :, :, :] = np.expand_dims(fake_img, axis=-1)

    return real_imgs, mask_imgs, fake_imgs


if __name__ == '__main__':
    tf.compat.v1.app.run()