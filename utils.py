import numpy as np
import h5py
import json
import sys
import scipy.io as sio
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pprint

pp = pprint.PrettyPrinter()

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def load_data(dataset_name, transfer=False):
    """
    dataset: the name of the dataset
    augmented: if true, use youtube and ovp dataset for training
    transfer: if true, train on summe, test on tvsum or vice versa.
    default: train on 80% dataset, test on remaining 20%.
    """
    train_percent = 0.8

    if transfer:
        train_percent = 1.0

    dataset = h5py.File("../data/eccv16_dataset_" + dataset_name + "_google_pool5.h5", "r")
    num_videos = len(dataset.keys())
    num_train = int(math.ceil(num_videos * train_percent))
    num_test = num_videos - num_train
    keys = dataset.keys()

    train_keys, test_keys = [], []
    rnd_idxs = np.random.choice(range(num_videos), size=num_train, replace=False)
    for key_idx, key in enumerate(keys):
        if key_idx in rnd_idxs:
            train_keys.append(key)
        else:
            test_keys.append(key)

    assert len(set(train_keys) & set(test_keys)) == 0

    return dataset, train_keys, test_keys

def load_5cv_data(dataset_name, fold=0):
    train_percent = 0.8

    dataset = h5py.File("../data/eccv16_dataset_" + dataset_name + "_google_pool5.h5", "r")
    num_videos = len(dataset.keys())
    num_train = int(math.ceil(num_videos * train_percent))
    num_test = num_videos - num_train
    keys = dataset.keys()

    train_keys, test_keys = [], []
    test_idxs = np.arange(num_videos)[fold*num_test : (fold+1)*num_test]
    for key_idx, key in enumerate(keys):
        if key_idx in test_idxs:
            test_keys.append(key)
        else:
            train_keys.append(key)

    assert len(set(train_keys) & set(test_keys)) == 0

    return dataset, train_keys, test_keys

