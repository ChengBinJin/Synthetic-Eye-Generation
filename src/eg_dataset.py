import os
import logging
import cv2
import numpy as np
import utils as utils
from scipy.ndimage import rotate

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
        self.output_img_shape = (int(self.resize_factor * img_shape[0]),
                                 int(self.resize_factor * img_shape[1]), 1)
        self.iris_shape = (200, 200, 1)

        self.train_folder = '../../Data/OpenEDS/Identification/train'
        self.val_folder = '../../Data/OpenEDS/Identification/val'
        self.test_folder = '../../Data/OpenEDS/Identification/test'
        self._read_img_path()

        if is_debug and is_train:
            self.debug_augmentation()

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
            self.logger.info('Output img shape: \t\t{}'.format(self.output_img_shape))
            self.logger.info('Resize_factor: \t\t{}'.format(self.resize_factor))

    def _read_img_path(self):
        # Generation task using training and validation data together
        self.train_paths = utils.all_files_under(self.train_folder) + utils.all_files_under(self.val_folder)
        self.val_paths = []
        self.test_paths = utils.all_files_under(self.test_folder)
        self.num_train_imgs = len(self.train_paths)
        self.num_val_imgs = len(self.val_paths)
        self.num_test_imgs = len(self.test_paths)

    def debug_augmentation(self, num_try=8, save_dir='../debug'):
        img_paths = [self.train_paths[idx] for idx in np.random.randint(self.num_train_imgs, size=num_try)]

        for img_path in img_paths:
            # Read data
            img_combine = cv2.imread(img_path)
            img = img_combine[:, :self.img_shape[1], 1]
            seg = img_combine[:, self.img_shape[1]:, :]

            # Translation
            img_tran, seg_tran = self.aug_translate(img, seg)
            # Flip
            img_flip, seg_flip = self.aug_flip(img, seg)
            # Rotation
            img_rot, seg_rot = self.aug_rotate(img, seg)
            # Translation, flip, and rotation
            img_aug, seg_aug = self.data_augmentation(img, seg)

            img_upper = np.hstack([img, img_tran, img_flip, img_rot, img_aug])
            img_upper = np.dstack([img_upper, img_upper, img_upper])
            seg_lower = np.hstack([seg, seg_tran, seg_flip, seg_rot, seg_aug])
            canvas = np.vstack([img_upper, seg_lower])

            canvas = cv2.resize(canvas, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

            cv2.imwrite(os.path.join(save_dir, 'gen_aug_'+os.path.basename(img_path)), canvas)


    def train_random_batch(self, batch_size):
        # img_paths = [self.train_paths[idx] for idx in np.random.randint(self.num_train_imgs, size=batch_size)]
        img_paths = [self.train_paths[idx] for idx in np.random.randint(20, size=batch_size)]
        train_imgs, train_labels, train_segs = self.read_data(img_paths, is_augment=True)
        return train_imgs, train_labels, train_segs

    def train_random_batch_include_iris(self, batch_size):
        img_paths = [self.train_paths[idx] for idx in np.random.randint(self.num_train_imgs, size=batch_size)]
        train_imgs, train_labels, train_segs, train_irises, train_coords = \
            self.read_data_include_iris(img_paths, is_augment=True)
        return train_imgs, train_labels, train_segs, train_irises, train_coords

    def read_data_include_iris(self, img_paths, is_augment=False):
        batch_imgs = np.zeros((len(img_paths), *self.output_img_shape), dtype=np.float32)
        batch_segs = np.zeros((len(img_paths), *self.input_img_shape), dtype=np.float32)
        batch_irises = np.zeros((len(img_paths), *self.iris_shape), dtype=np.float32)
        batch_labels = np.zeros((len(img_paths), 1), dtype=np.uint8)
        batch_coords = np.zeros((len(img_paths), 4), dtype=np.int32)

        for i, img_path in enumerate(img_paths):
            # Read img and seg
            img_combine = cv2.imread(img_path)
            img = img_combine[:, :self.img_shape[1], 1]
            seg = img_combine[:, self.img_shape[1]:, :]

            # Resize
            img = cv2.resize(img, None, fx=self.resize_factor, fy=self.resize_factor, interpolation=cv2.INTER_LINEAR)
            seg = cv2.resize(seg, None, fx=self.resize_factor, fy=self.resize_factor, interpolation=cv2.INTER_NEAREST)

            # Data augmentation
            if is_augment:
                img, seg = self.data_augmentation(img, seg)

            # Iris area
            batch_irises[i, :, :, :], batch_coords[i, :] = utils.extract_iris(
                np.reshape(img, (1, img.shape[0], img.shape[1], 1)), np.expand_dims(seg, axis=0))

            batch_imgs[i, :, :, 0] = img
            batch_segs[i, :, :, :] = seg
            batch_labels[i] = self.convert_to_cls(img_path)

        return batch_imgs, batch_labels, batch_segs, batch_irises, batch_coords

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

        imgs, labels, segs = self.read_data(img_paths, is_augment=False)

        return imgs, labels, segs

    def direct_batch_include_iris(self, batch_size, index, stage='train'):
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

        imgs, labels, segs, irises, coords = self.read_data_include_iris(img_paths, is_augment=False)

        return imgs, labels, segs, irises, coords

    def read_data(self, img_paths, is_augment=False):
        batch_imgs = np.zeros((len(img_paths), *self.output_img_shape), dtype=np.float32)
        batch_segs = np.zeros((len(img_paths), *self.input_img_shape), dtype=np.float32)
        batch_labels = np.zeros((len(img_paths), 1), dtype=np.uint8)

        for i, img_path in enumerate(img_paths):
            # Read img and seg
            img_combine = cv2.imread(img_path)
            img = img_combine[:, :self.img_shape[1], 1]
            seg = img_combine[:, self.img_shape[1]:, :]

            # Resize
            img = cv2.resize(img, None, fx=self.resize_factor, fy=self.resize_factor, interpolation=cv2.INTER_LINEAR)
            seg = cv2.resize(seg, None, fx=self.resize_factor, fy=self.resize_factor, interpolation=cv2.INTER_NEAREST)

            # Data augmentation
            if is_augment:
                img, seg = self.data_augmentation(img, seg)

            batch_imgs[i, :, :, 0] = img
            batch_segs[i, :, :, :] = seg
            batch_labels[i] = self.convert_to_cls(img_path)

        return batch_imgs, batch_labels, batch_segs

    def data_augmentation(self, img, seg):
        img_aug, seg_aug = self.aug_translate(img, seg)         # random translation
        img_aug, seg_aug = self.aug_flip(img_aug, seg_aug)      # random flip
        # img_aug, seg_aug = self.aug_rotate(img_aug, seg_aug)    # random rotate
        return img_aug, seg_aug

    @staticmethod
    def aug_translate(img, label, resize_factor=1.1):
        # Resize originl image
        img_bigger = cv2.resize(src=img.copy(), dsize=None, fx=resize_factor, fy=resize_factor,
                                interpolation=cv2.INTER_LINEAR)
        label_bigger = cv2.resize(src=label.copy(), dsize=None, fx=resize_factor, fy=resize_factor,
                                  interpolation=cv2.INTER_NEAREST)

        # Generate random positions for horizontal and vertical axes
        h_bigger, w_bigger = img_bigger.shape
        h_star = np.random.random_integers(low=0, high=h_bigger - img.shape[0])
        w_star = np.random.random_integers(low=0, high=w_bigger - img.shape[1])

        # Crop image from the bigger one
        img_crop = img_bigger[h_star:h_star + img.shape[0], w_star:w_star + img.shape[1]]
        label_crop = label_bigger[h_star:h_star + img.shape[0], w_star:w_star + img.shape[1]]

        return img_crop, label_crop

    @staticmethod
    def aug_flip(img, label):
        # Random vertical-axis flip
        if np.random.uniform(low=0., high=1.) > 0.5:
            img_flip = cv2.flip(src=img, flipCode=1)
            label_flip = cv2.flip(src=label, flipCode=1)
        else:
            img_flip = img.copy()
            label_flip = label.copy()

        return img_flip, label_flip

    @staticmethod
    def aug_rotate(img, label, min_degree=-10, max_degree=10):
        # Random rotate image
        angle = np.random.randint(low=min_degree, high=max_degree, size=None)
        img_rotate = rotate(input=img, angle=angle, axes=(0, 1), reshape=False, order=3, mode='constant', cval=0.)
        img_rotate = np.clip(img_rotate, a_min=0., a_max=255.)
        label_rotate = rotate(input=label, angle=angle, axes=(0, 1), reshape=False, order=0, mode='constant', cval=0.)

        return img_rotate, label_rotate


    @staticmethod
    def convert_to_cls(img_name):
        user_id = int(img_name[img_name.find('U')+1:img_name.find('.png')])

        if user_id < 199:
            return user_id - 111
        else:
            return user_id - 112
