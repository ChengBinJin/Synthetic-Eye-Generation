# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Synthetic Eye Generation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14})


def main(num_objects=122, target_examples=10):
    real_feats = np.load('./data/real_features.npy')
    fake_feats = np.load('./data/fake_features.npy')

    gen_live_live_dist = genuine_live_to_live(real_feats, num_objects, target_examples)
    # gen_live_syn_same_dist = genuine_live_to_syn_same(real_feats, fake_feats, num_objects, target_examples)
    # gen_live_syn_diff_dist = genuine_live_to_syn_diff(real_feats, fake_feats, num_objects, target_examples)
    # gen_syn_syn_dist = genuine_syn_to_syn(fake_feats, num_objects, target_examples)
    imp_live_live_dist = imposter_live_to_live(real_feats, num_objects, target_examples)
    # imp_live_syn_dist = imposter_live_to_syn(real_feats, fake_feats, num_objects, target_examples)
    imp_syn_syn_dist = imposter_syn_to_syn(fake_feats, num_objects, target_examples)

    # plot_combination_1(gen_live_live_dist, gen_live_syn_same_dist, imp_live_live_dist)
    # plot_combination_2(gen_live_live_dist, gen_live_syn_diff_dist, imp_live_live_dist)
    # plot_combination_3(gen_live_live_dist, gen_syn_syn_dist, imp_live_live_dist)
    # plot_combination_4(gen_live_live_dist, imp_live_syn_dist, imp_live_live_dist)
    plot_combination_5(gen_live_live_dist, imp_syn_syn_dist, imp_live_live_dist)


def plot_combination_5(gen_live_live_dist, imp_syn_syn_dist, imp_live_live_dist):
    # matplotlib histogram
    _, bins, _ = plt.hist(imp_live_live_dist, bins=range(0, 1000, 10), color='red', edgecolor='black', alpha=0.5, normed=True)
    plt.hist(gen_live_live_dist, bins=bins, color='blue', edgecolor='black', alpha=0.5, normed=True)
    plt.hist(imp_syn_syn_dist, bins=bins, color='green', edgecolor='black', alpha=0.5, normed=True)

    plt.grid(axis='y')
    plt.xlabel('Matching Distance')
    plt.ylabel('Frequency')
    plt.legend(['Imposter: live to live',
                'Genuine: live to live',
                'Imposter: synthesis to synthesis'], prop={'size': 18})
    plt.show()


def plot_combination_4(gen_live_live_dist, imp_live_syn_dist, imp_live_live_dist):
    # matplotlib histogram
    _, bins, _ = plt.hist(imp_live_live_dist, bins=range(0, 1000, 10), color='red', edgecolor='black', alpha=0.5, normed=True)
    plt.hist(gen_live_live_dist, bins=bins, color='blue', edgecolor='black', alpha=0.5, normed=True)
    plt.hist(imp_live_syn_dist, bins=bins, color='green', edgecolor='black', alpha=0.5, normed=True)

    plt.grid(axis='y')
    plt.xlabel('Matching Distance')
    plt.ylabel('Frequency')
    plt.legend(['Imposter: live to live',
                'Genuine: live to live',
                'Imposter: live to synthesis'], prop={'size': 18})
    plt.show()


def plot_combination_3(gen_live_live_dist, gen_syn_syn_dist, imp_live_live_dist):
    # matplotlib histogram
    _, bins, _ = plt.hist(imp_live_live_dist, bins=range(0, 1000, 10), color='red', edgecolor='black', alpha=0.5, normed=True)
    plt.hist(gen_live_live_dist, bins=bins, color='blue', edgecolor='black', alpha=0.5, normed=True)
    plt.hist(gen_syn_syn_dist, bins=bins, color='green', edgecolor='black', alpha=0.5, normed=True)

    plt.grid(axis='y')
    plt.xlabel('Matching Distance')
    plt.ylabel('Frequency')
    plt.legend(['Imposter: live to live',
                'Genuine: live to live',
                'Genuine: synthesis to synthesis'], prop={'size': 18})
    plt.show()


def plot_combination_2(gen_live_live_dist, gen_live_syn_diff_dist, imp_live_live_dist):
    # matplotlib histogram
    _, bins, _ = plt.hist(imp_live_live_dist, bins=range(0, 1000, 10), color='red', edgecolor='black', alpha=0.5, normed=True)
    plt.hist(gen_live_live_dist, bins=bins, color='blue', edgecolor='black', alpha=0.5, normed=True)
    plt.hist(gen_live_syn_diff_dist, bins=bins, color='green', edgecolor='black', alpha=0.5, normed=True)

    plt.grid(axis='y')
    plt.xlabel('Matching Distance')
    plt.ylabel('Frequency')
    plt.legend(['Imposter: live to live',
                'Genuine: live to live',
                'Genuine: live to synthesis from the different live'], prop={'size': 18})
    plt.show()


def plot_combination_1(gen_live_live_dist, gen_live_syn_same_dist, imp_live_live_dist):
    # matplotlib histogram
    _, bins, _ = plt.hist(imp_live_live_dist, bins=range(0, 1000, 10), color='red', edgecolor='black', alpha=0.5, normed=True)
    plt.hist(gen_live_live_dist, bins=bins, color='blue', edgecolor='black', alpha=0.5, normed=True)
    plt.hist(gen_live_syn_same_dist, bins=bins, color='green', edgecolor='black', alpha=0.5, normed=True)

    plt.grid(axis='y')
    plt.xlabel('Matching Distance')
    plt.ylabel('Frequency')
    plt.legend(['Imposter: live to live',
                'Genuine: live to live',
                'Genuine: live to synthesis from the same live'], prop={'size': 18})
    plt.show()



def genuine_syn_to_syn(fake_feats, num_objects, target_examples):
    print('Calculating genuine synthesis-to-synthesis matching...')
    dist = list()

    for i in range(num_objects):
        feats = fake_feats[i * target_examples: (i + 1) * target_examples, :]

        for j in range(target_examples):
            ref_feat = feats[j, :]

            for k in range(j+1, target_examples):
                target_feat = feats[k, :]
                dist.append(np.sum(np.linalg.norm(ref_feat - target_feat)))

    return np.asarray(dist)


def genuine_live_to_syn_diff(real_feats, fake_feats, num_objects, target_examples):
    print('Calculating genuine live-to-synthesis with different matching...')
    dist = list()

    for i in range(num_objects):
        ref_feats = real_feats[i * target_examples: (i + 1) * target_examples, :]
        target_feats = fake_feats[i * target_examples: (i + 1) * target_examples, :]

        for j in range(target_examples):
            ref_feat = ref_feats[j, :]

            for k in range(target_examples):
                if k == j:
                    continue

                target_feat = target_feats[k, :]
                dist.append(np.sum(np.linalg.norm(ref_feat - target_feat)))

    return np.asarray(dist)


def genuine_live_to_syn_same(real_feats, fake_feats, num_objects, target_examples):
    print('Calculating genuine live-to-synthesis with same matching...')
    dist = list()

    for i in range(num_objects):
        ref_feats = real_feats[i * target_examples: (i + 1) * target_examples, :]
        target_feats = fake_feats[i * target_examples: (i + 1) * target_examples, :]

        for j in range(target_examples):
            ref_feat = ref_feats[j, :]
            target_feat = target_feats[j, :]
            dist.append(np.sum(np.linalg.norm(ref_feat - target_feat)))

    return np.asarray(dist)


def genuine_live_to_live(real_feats, num_objects, target_examples):
    print('Calculating genuine live-to-live matching...')
    dist = list()

    for i in range(num_objects):
        feats = real_feats[i * target_examples: (i + 1) * target_examples, :]

        for j in range(target_examples):
            ref_feat = feats[j, :]

            for k in range(j+1, target_examples):
                target_feat = feats[k, :]
                dist.append(np.sum(np.linalg.norm(ref_feat - target_feat)))

    return np.asarray(dist)


def imposter_live_to_live(real_feats, num_objects, target_examples):
    print('Calculating imposter live-to-live matching...')
    dist = list()

    for i in range(num_objects):
        ref_feats = real_feats[i * target_examples: (i + 1) * target_examples, :]

        for j in range(i+1, num_objects):
            target_feats = real_feats[j * target_examples: (j + 1) * target_examples, :]

            for k in range(target_examples):
                ref_feat = ref_feats[k, :]

                for l in range(target_examples):
                    target_feat = target_feats[l, :]

                    dist.append(np.sum(np.linalg.norm(ref_feat - target_feat)))

    return np.asarray(dist)


def imposter_live_to_syn(real_feats, fake_feats, num_objects, target_examples):
    print('Calculating imposter live-to-synthesis matching...')
    dist = list()

    for i in range(num_objects):
        ref_feats = real_feats[i * target_examples: (i + 1) * target_examples, :]

        for j in range(i+1, num_objects):
            target_feats = fake_feats[j * target_examples: (j + 1) * target_examples, :]

            for k in range(target_examples):
                ref_feat = ref_feats[k, :]

                for l in range(target_examples):
                    target_feat = target_feats[l, :]

                    dist.append(np.sum(np.linalg.norm(ref_feat - target_feat)))

    return np.asarray(dist)


def imposter_syn_to_syn(fake_feats, num_objects, target_examples):
    print('Calculating imposter live-to-live matching...')
    dist = list()

    for i in range(num_objects):
        ref_feats = fake_feats[i * target_examples: (i + 1) * target_examples, :]

        for j in range(i+1, num_objects):
            target_feats = fake_feats[j * target_examples: (j + 1) * target_examples, :]

            for k in range(target_examples):
                ref_feat = ref_feats[k, :]

                for l in range(target_examples):
                    target_feat = target_feats[l, :]

                    dist.append(np.sum(np.linalg.norm(ref_feat - target_feat)))

    print(len(dist))
    return np.asarray(dist)



if __name__ == '__main__':
    main()


