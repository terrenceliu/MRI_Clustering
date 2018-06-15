from matplotlib import pyplot as plt
import pandas
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets.samples_generator import make_blobs
import csv
import os
import shutil
import sys

def compupte_cluster(X, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    print(len(labels))

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)

    #########
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    txt = "eps: %.2f, min_sam: %d, Clusters: %d" % (eps, min_samples, len(unique_labels))
    plt.xlabel(txt)
    plt.plot()

    out_name = ".\\fig\\eps" + str(eps) + "_min" + str(min_samples) + ".png"

    plt.savefig(out_name)

    # plt.show()

    return labels

def save_result(labels, eps, min_samples):
    # output dir
    out_file = ".\\data\\img_path_tsne_labels.csv"

    df2 = pandas.read_csv('.\\data\\img_path.csv', sep=',')

    f = open(out_file, 'w')
    w = csv.DictWriter(
        f, fieldnames=['id', 'file_path', 'label'], delimiter='\t',
        lineterminator='\n')
    w.writeheader()

    for i in range(len(labels)):
        w.writerow({'id': i, 'file_path': df2['img_path'][i], 'label': labels[i]})

    # Write to files
    df = pandas.read_csv('.\\data\\img_path_tsne_labels.csv', sep='\t')

    out_root = ".\\cluster\\eps" + str(eps) + "_min" + str(min_samples)

    if not os.path.exists(out_root):
        os.mkdir(out_root)

    total_count = df.shape[0]
    for idx, row in df.iterrows():
        out_dir = out_root + "\\" + str(row['label'])

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        if (row['label'] != -1):
            try:
                shutil.copy(row['file_path'], out_dir)
            except:
                print (sys.exc_info())

        if (idx % 500 == 0):
            print("Finish %.2f%%. Index: %d" % (1.0 * idx / total_count * 100, idx))

if __name__ == '__main__':
    input_file = ".\\data\\img_path_features_tsne.tsv"

    df = pandas.read_csv(input_file, sep="\t")
    x = []
    y = []
    for idx, row in df.iterrows():
        x.append(row['x'])
        y.append(row['y'])

    plt.scatter(x, y)
    plt.show()

    X = []
    for i in range(len(x)):
        X.append([x[i], y[i]])
    X = np.array(X)
    # print (X)

    # Compute DBSCAN
    eps = 6
    min_samples = 35

    # Compute cluster
    labels = compupte_cluster(X, eps, min_samples)


    # eps_list = [2, 3, 4, 5, 6, 7, 8]
    # min_samples_list = [10, 20, 40, 50, 60, 70]
    #
    # for eps in eps_list:
    #     for min_samples in min_samples_list:
    #         n_cluster = compupte_cluster(X, eps, min_samples)
    #         print (n_cluster)


    # Out put result

    save_result(labels, eps, min_samples)










