# --------------------------------------------------------------------------
# Tensorflow Implementation of Synthetic Eye Generation Challenge
# Iris Identification
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import os
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
        img_comb = cv2.imread(img_path)

        h, w, c = img_comb.shape
        w = w // 2
        img = img_comb[:, :w, :]
        label = img_comb[:, w:, :]

        img_bri = brightness_augment(img)
        label_bri = label.copy()

        img_rota, label_rota = rotation_augment(img, label)
        img_bri_rota, label_bri_rota = rotation_augment(img_bri, label_bri)

        # Mask
        mask1 = np.zeros((h, w, c), np.uint8)
        mask1[:, :, :][label[:, :, 1] == 204] = 1
        img_crop = img * mask1

        mask2 = np.zeros((h, w, c), np.uint8)
        mask2[:, :, :][label_bri_rota[:, :, 1] == 204] = 1
        img_crop_aug = img_bri_rota * mask2

        save_img(imgs=[img, img_bri, img_rota, img_bri_rota, img_crop],
                 labels=[label, label_bri, label_rota, label_bri_rota, img_crop_aug],
                 img_path=img_path,
                 save_dir=save_dir,
                 resize_factor=resize_factor)

def save_img(imgs, labels, img_path, save_dir='../debug', margin=5, resize_factor=0.5):
    num_imgs = len(imgs)
    h, w, c = imgs[0].shape

    canvas = np.zeros((2 * h + 3 * margin, num_imgs * w + (num_imgs + 1) * margin, c), dtype=np.uint8)
    for i in range(num_imgs):
        canvas[margin:margin + h, margin * (i + 1) + w * i:margin * (i + 1) + w * (i + 1), :] = imgs[i]
        canvas[2 * margin + h:2 * margin + h * 2, margin * (i + 1) + w * i:margin * (i + 1) + w * (i + 1), :] = labels[i]

    canvas = cv2.resize(canvas, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(save_dir, 'aug_' + os.path.basename(img_path)), canvas)

if __name__ == '__main__':
    main(args.batch_size, args.resize_factor)

