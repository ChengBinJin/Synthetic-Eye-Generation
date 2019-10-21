# --------------------------------------------------------------------------
# Tensorflow Implementation of Synthetic Eye Generation Challenge
# Iris Identification
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

from ii_dataset import Dataset

parser = argparse.ArgumentParser(description='')
parser.add_argument('--load_models', dest='load_models', type=list, default=['20191017-085852'],
                    help='model directory')
parser.add_argument('--task', dest='task', type=str, default='generation', help='select from [identification, generation]')
args = parser.parse_args()

PATH = os.path.join('../log', args.task)  #../log/identification/'

def main(load_models, save_dir='../experiments', num_identities=122):
    top_k_accs = np.zeros((num_identities, len(load_models)), np.float32)

    # Read test labels
    data = Dataset()
    gt_labels = np.asarray([data.convert_to_cls(test_path) for test_path in data.test_paths])

    for iter_, load_model in enumerate(load_models):
        print('load_model: {}'.format(load_model))
        print('path: {}'.format(os.path.join(PATH, load_model, 'preds.csv')))

        # Read saved prediction scores
        preds = genfromtxt(os.path.join(PATH, load_model, 'preds.csv'), delimiter=',')
        num_imgs = preds.shape[0]

        # Correct k == 1 accuracy
        acc = np.mean(gt_labels == np.argmax(preds, axis=1))
        print('\n')
        print('='*30)
        print('load model: {}'.format(load_model))
        print('K == 1: Acc. {:.2f}'.format(acc * 100.))
        print('=' * 30)

        for k in range(1, num_identities+1):
            new_preds = preds.argsort(axis=1)[:, -k:][:, ::-1]

            correct = 0
            for i in range(num_imgs):
                if gt_labels[i] in new_preds[i]:
                    correct += 1

            acc = correct / num_imgs * 100.
            top_k_accs[k-1, iter_] = acc

            if (k == 1) or (k == 5):
                print('K == {}: Acc. {:.2f}'.format(k, acc))

        print('Model: {}, AUC: {:.4}'.format(load_model, (np.sum(top_k_accs[:, iter_] / (100. * num_identities)))))

        save_fig(top_k_accs, save_dir, load_model)


def save_fig(top_k_acc, save_dir, load_model):
    # Parameters
    colors = ['tomato', 'lawngreen', 'dodgerblue', 'darkviolet']
    groups = ['IrisGAN without any-constraint', 'IrisGAN with whole-constraint',
              'IrisGAN with iris-constraint', 'IrisGAN with iris-preserving']
    # linestyles = [':', '-.']

    fig, ax = plt.subplots(figsize=(16, 8))
    for i in range(top_k_acc.shape[1]):
        # ax.plot(top_k_acc[:, i], color=colors[i], linewidth=4.0, label=groups[i], linestyle=linestyles[i])
        ax.plot(top_k_acc[:, i], color=colors[i], linewidth=4.0, label=groups[i])

    ax.set_xlabel('Rank', fontsize=16)
    ax.set_ylabel('Identification Accuracy', fontsize=16)
    ax.set_xticks(range(0, 121, 10))
    ax.set_yticks(range(65, 101, 5))

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    plt.title('Cumulative Matching Characteristic (CMC) Curve', fontsize=16)
    plt.xlim(0, 121)
    plt.ylim(top_k_acc[:, 0].min(), 100)
    plt.legend(loc='lower right', fontsize=14)
    plt.grid(True)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, load_model + '.png'), dpi=600)
    plt.close()



if __name__ == '__main__':
    main(args.load_models)