# --------------------------------------------------------------------------
# Tensorflow Implementation of Synthetic Eye Generation Challenge
# Iris Identification
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import os
import cv2
import matplotlib.pyplot as plt
from utils import all_files_under


def main(data_path, num_tests=20, num_vals=10):
    statistics_num = list()
    files = all_files_under(data_path)

    for id_ in range(111, 234, 1):
        user_id = 'U' + str(id_)
        num = 0

        # print('Processing user_id: {}...'.format(user_id))
        for img_path in files:
            if user_id in img_path:
                num += 1

        if num < num_tests + num_vals:
            print('ID: {}, num. of images are less then {}'.format(user_id, num_tests + num_vals))
            continue
        else:
            statistics_num.append(num)

    draw_fig(statistics_num)


def draw_fig(data, save_dir='../debug'):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig, ax = plt.subplots(figsize=(20, 10))

    ax.bar(x=range(len(data)), height=data, edgecolor='black', color='dodgerblue', linewidth=1.5, )
    ax.set_xlabel('Person ID.', fontsize=16)
    ax.set_ylabel('Num. of samples', fontsize=16)

    ax.set_yticks(range(0, 250, 25))

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    plt.grid(True)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Statistics of OpenEDS Identification.png'), dpi=600)
    plt.close()


if __name__ == '__main__':
    path = '../../Data/OpenEDS/Identification/backup'
    main(path)