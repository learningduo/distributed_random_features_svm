import csv
import os
from shutil import copyfile
import random
import numpy as np
import math


def exec_htk(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('mfc'):
                file_name = os.path.join(subdir, file)
                cmd = "./htk2txt '" + file_name + "' 'output/test/data/" + file.replace('mfc', 'txt') + "'"
                os.system(cmd)


def get_files_labeled(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('labels'):
                file_name = os.path.join(subdir, file)
                copyfile(file_name, 'output/test/labeled/' + file)


##

phoneme_map = dict()
phoneme_counter = 0


def parse_labels(f):
    # print f
    global phoneme_map
    global phoneme_counter

    lines = open(f).readlines()
    labels = []
    for line in lines:
        line = line.strip()
        if line in phoneme_map:
            labels.append(phoneme_map[line])
        else:
            phoneme_map[line] = phoneme_counter
            labels.append(phoneme_counter)
            phoneme_counter += 1
    return labels


def normalize(data):
    x = np.array([])
    i = 0
    for row in data:
        line = np.fromstring(row, sep=' ')
        if i == 0:
            x = np.hstack((x, line))
            i += 1
        else:
            x = np.vstack((x, line))

    x_norm = x / np.linalg.norm(x)
    list(x_norm)
    return x_norm.tolist()
    # return x_norm


def parse_file(f, label_file):
    txt = open(f).readlines()
    file_data = map(lambda x: x.strip(), txt)
    labels_data = parse_labels(label_file)
    if len(file_data) + 1 == len(labels_data):
        labels_data = labels_data[:-1]
    elif len(file_data) == len(labels_data) + 1:
        file_data = file_data[:-1]
    elif len(file_data) == len(labels_data):
        pass
    else:
        print "problem with files length: " + len(file_data) + ", " + len(labels_data)
    return normalize(file_data), labels_data


def dump(data, labels, n, folder, test):
    if test:
        folder += 'test_'
    print folder
    data = [item for sublist in data for item in sublist]
    labels = [item for sublist in labels for item in sublist]

    zipped = zip(data, labels)
    random.shuffle(zipped)
    shuf_data, shuf_labels = zip(*zipped)

    lines = len(shuf_data)

    # saving data to files
    step = int(math.floor(lines / n))
    stop = step * n
    file_n = 1
    for i in range(0, stop - step, step):
        print folder + 'data_' + str(n) + '_' + str(file_n) + '.txt'
        out_f = open(folder + 'data_' + str(n) + '_' + str(file_n) + '.txt', 'w')
        out_l = open(folder + 'labels_' + str(n) + '_' + str(file_n) + '.txt', 'w')

        for j in range(0, step):
            ind = i + j
            out_f.write("%s\n" % str(data[ind])[1:-1])
            out_l.write("%s\n" % str(labels[ind]))
        out_f.close()
        out_l.close()
        file_n += 1
    out_f = open(folder + 'data_' + str(n) + '_' + str(file_n) + '.txt', 'w')
    out_l = open(folder + 'labels_' + str(n) + '_' + str(file_n) + '.txt', 'w')
    for i in range(stop - step, lines):
        out_f.write("%s\n" % str(data[i])[1:-1])
        out_l.write("%s\n" % str(labels[i]))
    out_f.close()
    out_l.close()


def build_data_set(split_num, folder, test=False):
    global phoneme_map

    dir_r = 'output'
    for subdir, dirs, files in os.walk(os.path.join(dir_r, folder, 'data')):
        parsed_count = 0
        data = []
        labels = []
        for f in files:
            file_name = os.path.join(subdir, f)
            label_file = subdir + "/../labeled/" + f.replace('txt', 'labels')

            file_output, labels_output = parse_file(file_name, label_file)
            data.append(file_output)
            labels.append(labels_output)
            parsed_count += 1

    global phoneme_map
    print phoneme_map

    dump(data, labels, split_num, os.path.join(dir_r, folder) + '/', test)

    phoneme_f = csv.writer(open(os.path.join(dir_r, folder) + '/phoneme_' + str(split_num) + '.txt', 'w'))
    for key, val in phoneme_map.items():
        phoneme_f.writerow([key, val])


def main():
    # parsing the basic data
    # root_dir_train = '/home/yanai/Downloads/Data/train'
    # root_dir_test = '/home/yanai/Downloads/Data/test'
    # exec_htk(root_dir_train)
    # exec_htk(root_dir_test)
    # print "done parsing htk"
    # get_files_labeled(root_dir_train)
    # get_files_labeled(root_dir_test)
    # print "done copying labels files"

    n = 1
    build_data_set(n, 'train')
    print 'done building training set'
    build_data_set(n, 'test', True)
    print 'done building test set'
    print 'finish job'


if __name__ == "__main__":
    main()
