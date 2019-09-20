import os
import cv2
import logging
import numpy as np


def convert_color_label(img):
    yellow = [102, 255, 255]
    green = [102, 204, 0]
    cyan = [153, 153, 0]
    violet = [102, 0, 102]

    # 0: background - violet
    # 1: sclera - cyan
    # 2: iris - green
    # 3: pupil - yellow
    img_rgb = np.zeros([*img.shape, 3], dtype=np.uint8)
    for i, color in enumerate([violet, cyan, green, yellow]):
        img_rgb[img == i] = color

    return img_rgb


def all_files_under(folder, subfolder=None, endswith='.png'):
    if subfolder is not None:
        new_folder = os.path.join(folder, subfolder)
    else:
        new_folder = folder

    if os.path.isdir(new_folder):
        file_names =  [os.path.join(new_folder, fname)
                       for fname in os.listdir(new_folder) if fname.endswith(endswith)]
        return sorted(file_names)
    else:
        return []


def init_logger(logger, log_dir, name, is_train):
    logger.propagate = False  # solve print log multiple times problem
    file_handler, stream_handler = None, None

    if is_train:
        formatter = logging.Formatter(' - %(message)s')

        # File handler
        file_handler = logging.FileHandler(os.path.join(log_dir, name + '.log'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        # Stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        # Add handlers
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)

    return logger, file_handler, stream_handler


def make_folders_simple(is_train=True, cur_time=None, subfolder=None):
    model_dir = os.path.join('../model', subfolder, '{}'.format(cur_time))
    log_dir = os.path.join('../log', subfolder, '{}'.format(cur_time))

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    return model_dir, log_dir


def debug_iris_cropping(img_paths, save_dir, h=640, w=400):
    for i, img_path in enumerate(img_paths):
        canvas = np.zeros((h, 3 * w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        img_combine = cv2.imread(img_path)

        # Eye img
        img = img_combine[:, :w, :]
        # Seg img
        seg = img_combine[:, w:, :]
        # Seg iris img
        mask[seg[:, :, 1] == 204] = 1
        # Cropped iris img
        crop_img = img * np.expand_dims(mask, axis=2)

        canvas[:, :w, :] = img
        canvas[:, w:2 * w, :] = seg
        canvas[:, 2 * w:3 * w, :] = crop_img

        # Save img
        cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), canvas)


def padding(img, shape=(200, 200)):
    tmp = np.zeros(shape, dtype=np.float32)
    h, w = img.shape

    if h <= w:
        factor = tmp.shape[1] / w
        re_h = int(factor * h)
        img = cv2.resize(img, (tmp.shape[1], re_h))

        start_h = int(0.5 * (tmp.shape[0] - re_h))
        tmp[start_h:start_h+re_h, :] = img
    else:
        factor = tmp.shape[0] / h
        re_w = int(factor * w)
        img = cv2.resize(img, (re_w, tmp.shape[0]))

        start_w = int(0.5 * (tmp.shape[1] - re_w))
        tmp[:, start_w:start_w + re_w] = img

    return tmp
