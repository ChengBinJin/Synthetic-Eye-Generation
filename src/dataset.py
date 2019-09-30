import os
import logging
import cv2
import numpy as np
import utils as utils

class Dataset(object):
    def __init__(self, name='Identification', mode=1, resize_factor=0.5, img_shape=(640, 400, 1), is_train=False,
                 log_dir=None, is_debug=False):
        self.name = name
        self.mode = mode
        self.resize_factor = resize_factor
        self.num_identities = 122
        self.num_seg_class = 4
        self.img_shape = img_shape
        if self.mode == 0:      # e.g. (320, 200, 1)
            self.input_img_shape = (int(self.resize_factor * img_shape[0]),
                                int(self.resize_factor * img_shape[1]), img_shape[2])
        elif self.mode == 1:    # e.g. (200, 200, 1)
            self.input_img_shape = (int(self.resize_factor * img_shape[1]),
                                    int(self.resize_factor * img_shape[1]), img_shape[2])
        else:
            raise NotImplementedError

        self.train_folder = '../../Data/OpenEDS/Identification/train'
        self.val_folder = '../../Data/OpenEDS/Identification/val'
        self.test_folder = '../../Data/OpenEDS/Identification/test'
        self._read_img_path()

        if is_train:
            self.logger = logging.getLogger(__name__)  # logger
            self.logger.setLevel(logging.INFO)
            utils.init_logger(logger=self.logger, log_dir=log_dir, is_train=is_train, name='dataset')

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

        if self.mode == 1 & is_debug:
            self.debug_iris_cropping(num_try=8, save_dir='../debug')

    def _read_img_path(self):
        self.train_paths = utils.all_files_under(self.train_folder)
        self.val_paths = utils.all_files_under(self.val_folder)
        self.test_paths = utils.all_files_under(self.test_folder)
        self.num_train_imgs = len(self.train_paths)
        self.num_val_imgs = len(self.val_paths)
        self.num_test_imgs = len(self.test_paths)

    def debug_iris_cropping(self, num_try, save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # Random select img paths
        img_paths = [self.train_paths[idx] for idx in np.random.randint(self.num_train_imgs, size=num_try)]
        utils.debug_iris_cropping(img_paths, save_dir)

    def train_random_batch(self, batch_size):
        img_paths = [self.train_paths[idx] for idx in np.random.randint(self.num_train_imgs, size=batch_size)]

        if self.mode == 0:
            train_imgs, train_labels, tran_segs = self.read_data(img_paths)
            return train_imgs, train_labels, tran_segs
        elif self.mode == 1:
            train_imgs, train_labels = self.read_iris_data(img_paths, is_augment=True)
            return train_imgs, train_labels
        else:
            raise NotImplementedError


    def direct_batch(self, batch_size, index, stage='train'):
        if stage == 'train':
            num_imgs = self.num_train_imgs
            all_paths = self.train_paths
        elif stage == 'val':
            num_imgs = self.num_val_imgs
            all_paths = self.val_paths
        elif stage == 'test':
            num_imgs = self.num_test_imgs
            all_paths = self.test_paths
        else:
            raise NotImplementedError

        if index + batch_size < num_imgs:
            img_paths = all_paths[index:index + batch_size]
        else:
            img_paths = all_paths[index:]

        if self.mode == 0:
            imgs, labels, segs = self.read_data(img_paths)
            return imgs, labels, segs
        elif self.mode == 1:
            imgs, labels = self.read_iris_data(img_paths, is_augment=False)
            return imgs, labels
        else:
            raise NotImplementedError


    def read_iris_data(self, img_paths, margin=5, is_augment=False):
        batch_imgs = np.zeros((len(img_paths), self.input_img_shape[1], self.input_img_shape[1], 1), dtype=np.float32)
        batch_labels = np.zeros((len(img_paths), 1), dtype=np.uint8)

        for i, img_path in enumerate(img_paths):
            mask = np.zeros((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8)

            # Extract Iris part
            img_combine = cv2.imread(img_path)
            img = img_combine[:, :self.img_shape[1], :]
            seg = img_combine[:, self.img_shape[1]:, :]

            if is_augment is True:
                # Data augmentation: random brightness + random rotation
                img_aug, seg_aug = utils.data_augmentation(img, seg)
                mask[:, :, :][seg_aug[:, :, 1] == 204] = 1
                img = img_aug * mask
            else:
                mask[:, :, :][seg[:, :, 1] == 204] = 1
                img = img * mask

            # Cropping iris part
            x, y, w, h = cv2.boundingRect(mask[:, :, 1])
            new_x = np.maximum(0, x - margin)
            new_y = np.maximum(0, y - margin)
            crop_img = img[new_y:new_y + h + margin, new_x:new_x + w + margin, 1]  # Extract more bigger area

            # Padding to the required size by preserving ratio of height and width
            batch_imgs[i, :, :, 0] = utils.padding(crop_img)
            batch_labels[i] = self.convert_to_cls(img_path)

        return batch_imgs, batch_labels

    def read_data(self, img_paths):
        batch_imgs = np.zeros((len(img_paths), *self.input_img_shape), dtype=np.float32)
        batch_segs = np.zeros((len(img_paths), *self.input_img_shape), dtype=np.float32)
        batch_labels = np.zeros((len(img_paths), 1), dtype=np.uint8)

        for i, img_path in enumerate(img_paths):
            # Read img and seg
            img_combine = cv2.imread(img_path)
            img = img_combine[:, :self.img_shape[1], 1]
            seg = img_combine[:, self.img_shape[1]:, 1]

            # Resize
            img = cv2.resize(img, None, fx=self.resize_factor, fy=self.resize_factor, interpolation=cv2.INTER_LINEAR)
            seg = cv2.resize(seg, None, fx=self.resize_factor, fy=self.resize_factor, interpolation=cv2.INTER_NEAREST)

            batch_imgs[i, :, :, 0] = img
            batch_segs[i, :, :, 0] = seg
            batch_labels[i] = self.convert_to_cls(img_path)

        return batch_imgs, batch_labels, batch_segs


    @staticmethod
    def convert_to_cls(img_name):
        user_id = int(img_name[img_name.find('U')+1:img_name.find('.png')])

        if user_id < 199:
            return user_id - 111
        else:
            return user_id - 112
