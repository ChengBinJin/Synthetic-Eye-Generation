import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

from dataset import Dataset

parser = argparse.ArgumentParser(description='')
parser.add_argument('--load_model', dest='load_model', type=str, default='20190919-202335', help='model directory')
args = parser.parse_args()

PATH = '../log/identification/'

def main(load_model, save_dir='../debug'):
    # Read saved prediction scores
    preds = genfromtxt(os.path.join(PATH, load_model, 'preds.csv'), delimiter=',')
    num_imgs, num_identities = preds.shape

    # Read test labels
    data = Dataset()
    gt_labels = np.asarray([data.convert_to_cls(test_path) for test_path in data.test_paths])

    # Correct k == 1 accuracy
    acc = np.mean(gt_labels == np.argmax(preds, axis=1))
    print('='*30)
    print('K == 1: Acc. {:.2f}'.format(acc * 100.))
    print('=' * 30)

    top_k_acc = np.zeros(num_identities, np.float32)
    for k in range(1, num_identities+1):
        new_preds = preds.argsort(axis=1)[:, -k:][:, ::-1]

        correct = 0
        for i in range(num_imgs):
            if gt_labels[i] in new_preds[i]:
                correct += 1

        acc = correct / num_imgs * 100.
        print('K == {}: Acc. {:.2f}'.format(k, acc))

        top_k_acc[k-1] = acc

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.plot(top_k_acc, color='dodgerblue', linewidth=2.0)
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
    plt.ylim(65, 100)
    plt.grid(True)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, load_model + '.png'), dpi=600)
    plt.close()



if __name__ == '__main__':
    main(args.load_model)