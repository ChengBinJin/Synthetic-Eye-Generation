import os
import cv2
import logging
import numpy as np
from scipy.ndimage import rotate


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


def make_folders(is_train=True, cur_time=None, subfolder=None):
    model_dir = os.path.join('../model', subfolder, '{}'.format(cur_time))
    log_dir = os.path.join('../log', subfolder, '{}'.format(cur_time))
    sample_dir = os.path.join('../sample', subfolder, '{}'.format(cur_time))
    val_dir, test_dir = None, None

    if is_train:
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        if not os.path.isdir(sample_dir):
            os.makedirs(sample_dir)
    else:
        val_dir = os.path.join('../val', subfolder, '{}'.format(cur_time))
        test_dir = os.path.join('../test', subfolder, '{}'.format(cur_time))

        if not os.path.isdir(val_dir):
            os.makedirs(val_dir)

        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

    return model_dir, log_dir, sample_dir, val_dir, test_dir


def make_folders_simple(cur_time=None, subfolder=None):
    model_dir = os.path.join('../model', subfolder, '{}'.format(cur_time))
    log_dir = os.path.join('../log', subfolder, '{}'.format(cur_time))

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    return model_dir, log_dir


def debug_iris_cropping(img_paths, save_dir, h=640, w=400, margin=5):
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

        # Cropping iris part
        x_, y_, w_, h_ = cv2.boundingRect(mask[:, :])
        new_x = np.maximum(0, x_ - margin)
        new_y = np.maximum(0, y_ - margin)
        iris_crop_img = crop_img[new_y:new_y + h_ + margin, new_x:new_x + w_ + margin, 1]  # Extract more bigger area

        # Padding to the required size by preserving ratio of height and width
        iris_crop_img = padding(iris_crop_img)

        canvas[:, :w, :] = img
        canvas[:, w:2 * w, :] = seg
        canvas[:, 2 * w:3 * w, :] = crop_img

        # Save img
        cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), canvas)
        cv2.imwrite(os.path.join(save_dir, 'crop_' + os.path.basename(img_path)), iris_crop_img)


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


def data_augmentation(img, label):
    # Random brightness
    img_bri = brightness_augment(img)
    label_bri = label.copy()

    # Rondom rotation
    img_bri_rota, label_bri_rota = rotation_augment(img_bri, label_bri)

    return img_bri_rota, label_bri_rota


def save_imgs(img_stores, iter_time=None, save_dir=None, margin=5, img_name=None, name_append='', is_vertical=True):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    num_categories = len(img_stores)
    num_imgs, h, w = img_stores[0].shape[0:3]

    if is_vertical:
        canvas = np.zeros((num_categories * h + (num_categories + 1) * margin,
                           num_imgs * w + (num_imgs + 1) * margin, 3), dtype=np.uint8)

        for i in range(num_imgs):
            for j in range(num_categories):
                canvas[(j + 1) * margin + j * h:(j + 1) * margin + (j + 1) * h,
                (i + 1) * margin + i * w:(i + 1) * (margin + w), :] = img_stores[j][i]
    else:
        canvas = np.zeros((num_imgs * h + (num_imgs + 1) * margin,
                           num_categories * w + (num_categories + 1) * margin, 3), dtype=np.uint8)

        for i in range(num_imgs):
            for j in range(num_categories):
                canvas[(i+1)*margin+i*h:(i+1)*(margin+h), (j+1)*margin+j*w:(j+1)*margin+(j+1)*w, :] = img_stores[j][i]

    if img_name is None:
        cv2.imwrite(os.path.join(save_dir, str(iter_time).zfill(6) + '.png'), canvas)
    else:
        cv2.imwrite(os.path.join(save_dir, name_append+img_name[0]), canvas)