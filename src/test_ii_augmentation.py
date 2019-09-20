# --------------------------------------------------------------------------
# Tensorflow Implementation of Synthetic Eye Generation Challenge
# Iris Identification
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import cv2
import argparse
import numpy as np
from scipy.ndimage import rotate
from dataset import Dataset


parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', dest='batch_size', default=8, help='number of imgs to test')
parser.add_argument('--resize_factor', dest='resize_factor', default=0.5,
                    help='resize factor of the canvas to save memory')
args = parser.parse_args()


def brightness_augment(img, factor=0.5):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)   # convert to hsv
    hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform())    # scale channel V unfiromly
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255                          # reset out of range values
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return rgb


def rotation_augment(img, label, min_degree=-30, max_degree=30):
    # Random rotate image
    degree = np.random.randint(low=min_degree, high=max_degree, size=None)
    img_rotate = rotate(input=img, angle=degree, axes=(0, 1), reshape=False, order=3, mode='constant', cval=0.)
    img_rotate = np.clip(img_rotate, a_min=0., a_max=255.)
    label_rotate = rotate(input=label, angle=degree, axes=(0, 1), reshape=False, order=0, mode='constant', cval=0.)
    return img_rotate.astype(np.uint8), label_rotate.astype(np.uint8)


def main(batch_size=8, resize_factor=0.5, save_dir='../debug'):
    data = Dataset()
    img_paths = [data.train_paths[idx] for idx in np.random.randint(low=0, high=data.num_train_imgs, size=batch_size)]

    for img_path in img_paths:
        print(img_path)

    # img = cv2.imread('../../Data/OpenEDS/Identification/train/000000002640_U111.png')
    # h, w, c = img.shape
    # w = w // 2
    #
    # img_ori = img[:, :w, :]
    # label_ori = img[:, w:, :]
    # # img_aug = brightness_augment(img_ori)
    # img_rotate, label_rotate = rotation_augment(img_ori, label_ori)
    #
    # canvas_img = np.zeros((h, 2*w, c), dtype=np.uint8)
    # canvas_img[:, :w, :] = img_ori
    # canvas_img[:, w:, :] = img_rotate
    #
    # canvas_label = np.zeros((h, 2*w, c), dtype=np.uint8)
    # canvas_label[:, :w, :] = label_ori
    # canvas_label[:, w:, :] = label_rotate
    #
    # cv2.imshow("Img", canvas_img)
    # cv2.imshow("Label", canvas_label)
    # cv2.waitKey(0)

def save_img(img, save_dir='../debug', margin=5, resize_factor=0.5):
    print("Hello save_img!")

if __name__ == '__main__':
    main(args.batch_size, args.resize_factor)

