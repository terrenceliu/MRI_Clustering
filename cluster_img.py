import pandas
import os
import shutil


if __name__ == '__main__':
    df = pandas.read_csv('.\\img_path_labels.csv', sep='\t')

    eps = 7
    min_sam = 60

    out_root = ".\\cluster\\eps" + str(eps) + "_min" + str(min_sam)

    if not os.path.exists(out_root):
        os.mkdir(out_root)

    total_count = df.shape[0]
    for idx, row in df.iterrows():
        out_dir = out_root + "\\" + str(row['label'])


        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        if (row['label'] != -1):
            shutil.copy(row['file_path'], out_dir)

        if (idx % 500 == 0):
            print ("Finish %.2f%%. Index: %d" % (1.0 * idx / total_count * 100, idx))


