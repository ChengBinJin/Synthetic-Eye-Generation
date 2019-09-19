# --------------------------------------------------------------------------
# Tensorflow Implementation of Synthetic Eye Generation Challenge
# Iris Identification
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import cv2
import numpy as np
from utils import all_files_under, convert_color_label

train_folder = '../../Data/OpenEDS/Identification/train'
val_folder = '../../Data/OpenEDS/Identification/val'
test_folder = '../../Data/OpenEDS/Identification/test'

def main(data_path, num_threshold=30, num_tests=20, num_vals=10):
    files = all_files_under(data_path)

    for id_ in range(111, 234, 1):
        user_id = 'U' + str(id_)
        num = 0
        candidate_paths = list()

        print('Processing user_id: {}...'.format(user_id))
        for img_path in files:
            if user_id in img_path:
                num += 1
                candidate_paths.append(img_path)

        if num < num_threshold:
            print('ID: {}, num. of images are less then {}'.format(user_id, num_threshold))
            continue
        else:
            candidate_paths = sorted(candidate_paths)

            # Save test images
            for img_path in candidate_paths[:num_tests]:
                canvas = convert_rgb_img(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
                cv2.imwrite(img_path.replace('backup', 'test'), canvas)

            # Save val images
            for img_path in candidate_paths[num_tests:num_tests + num_vals]:
                canvas = convert_rgb_img(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
                cv2.imwrite(img_path.replace('backup', 'val'), canvas)

            # Save train images
            for img_path in candidate_paths[num_tests + num_vals:]:
                canvas = convert_rgb_img(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
                cv2.imwrite(img_path.replace('backup', 'train'), canvas)


def convert_rgb_img(data):
    h, w2 = data.shape
    w = int(0.5 * w2)
    img = data[:, :w]
    label = convert_color_label(data[:, w:])

    canvas = np.zeros((h, w2, 3), np.uint8)
    canvas[:, :w, :] = np.dstack((img, img, img))
    canvas[:, w:, :] = label
    return canvas


if __name__ == '__main__':
    path = '../../Data/OpenEDS/Identification/backup'
    main(path)