import os
import logging
import cv2
import numpy as np
import utils as utils

class Dataset(object):
    def __init__(self, name='Generation', resize_factor=0.5, img_shape=(640, 400, 1), is_train=False,
                 log_dir=None, is_debug=False):
        self.name = name
        self.resize_factor = resize_factor
        self.num_identities = 122
        self.num_seg_class = 4
        self.img_shape = img_shape
        self.input_img_shape = (int(self.resize_factor * img_shape[0]),
                                int(self.resize_factor * img_shape[1]), 3)

        self.train_folder = '../../Data/OpenEDS/Identification/train'
        self.val_folder = '../../Data/OpenEDS/Identification/val'
        self.test_folder = '../../Data/OpenEDS/Identification/test'
        self._read_img_path()

        if is_train:
            self.logger = logging.getLogger(__name__)  # logger
            self.logger.setLevel(logging.INFO)
            utils.init_logger(logger=self.logger, log_dir=log_dir, is_train=is_train, name='eg_dataset')

            self.logger.info('Dataset name: \t\t{}'.format(self.name))
            self.logger.info('Train folder: \t\t{}'.format(self.train_folder))
            self.logger.info('Val folder: \t\t\t{}'.format(self.val_folder))
            self.logger.info('Test folder: \t\t{}'.format(self.test_folder))
            self.logger.info('Num. train imgs: \t\t{}'.format(self.num_train_imgs))
            self.logger.info('Num. val imgs: \t\t{}'.format(self.num_val_imgs))
            self.logger.info('Num. test imgs: \t\t{}'.format(self.num_test_imgs))
            self.logger.info('Num. identities: \t\t{}'.format(self.num_identities))
            self.logger.info('Num. seg. classes: \t\t{}'.format(self.num_seg_class))
            self.logger.info('Original img shape: \t\t{}'.format(self.img_shape))
            self.logger.info('Input img shape: \t\t{}'.format(self.input_img_shape))
            self.logger.info('Resize_factor: \t\t{}'.format(self.resize_factor))

    def _read_img_path(self):
        self.train_paths = utils.all_files_under(self.train_folder)
        self.val_paths = utils.all_files_under(self.val_folder)
        self.test_paths = utils.all_files_under(self.test_folder)
        self.num_train_imgs = len(self.train_paths)
        self.num_val_imgs = len(self.val_paths)
        self.num_test_imgs = len(self.test_paths)

    def train_random_batch(self, batch_size):
        img_paths = [self.train_paths[idx] for idx in np.random.randint(self.num_train_imgs, size=batch_size)]
        train_imgs, train_labels, train_segs = self.read_data(img_paths)
        return train_imgs, train_labels, train_segs

    def read_data(self, img_paths):
        batch_imgs = np.zeros((len(img_paths), *self.input_img_shape), dtype=np.float32)
        batch_segs = np.zeros((len(img_paths), *self.input_img_shape[0:2], 3), dtype=np.float32)
        batch_labels = np.zeros((len(img_paths), 1), dtype=np.uint8)

        for i, img_path in enumerate(img_paths):
            # Read img and seg
            img_combine = cv2.imread(img_path)
            img = img_combine[:, :self.img_shape[1], 1]
            seg = img_combine[:, self.img_shape[1]:, :]

            # Resize
            img = cv2.resize(img, None, fx=self.resize_factor, fy=self.resize_factor, interpolation=cv2.INTER_LINEAR)
            seg = cv2.resize(seg, None, fx=self.resize_factor, fy=self.resize_factor, interpolation=cv2.INTER_NEAREST)

            batch_imgs[i, :, :, 0] = img
            batch_segs[i, :, :, :] = seg
            batch_labels[i] = self.convert_to_cls(img_path)

        return batch_imgs, batch_labels, batch_segs

    @staticmethod
    def convert_to_cls(img_name):
        user_id = int(img_name[img_name.find('U')+1:img_name.find('.png')])

        if user_id < 199:
            return user_id - 111
        else:
            return user_id - 112
