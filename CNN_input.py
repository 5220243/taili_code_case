import os

import numpy as np
import tensorflow as tf

import _pickle as pickle

TRAIN_FILE = ['800']

IMAGESIZE=int(32)

EVAL_FILE = ['yz']


def unpickle(filename):
    '''Decode the dataset files.'''
    with open(filename, 'r+b') as f:
        d = pickle.load(f,encoding='latin1')
        return d


def onehot(labels, cls=None):
    ''' One-hot encoding, zero-based'''
    n_sample = len(labels)
    label=[int(i) for i in labels]
    if not cls:
        n_class = max(label)+1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), label] = 1
    return onehot_labels


def merge_data(dataset_dir, onehot_encoding=False):
    train_images = unpickle(os.path.join(dataset_dir, TRAIN_FILE[0]))['data']
    train_labels = unpickle(os.path.join(dataset_dir, TRAIN_FILE[0]))['labels']
    eval_images = unpickle(os.path.join(dataset_dir, EVAL_FILE[0]))['data']
    eval_labels = unpickle(os.path.join(dataset_dir, EVAL_FILE[0]))['labels']

    for i in range(1, len(TRAIN_FILE)):
        batch = unpickle(os.path.join(dataset_dir, TRAIN_FILE[i]))
        train_images = np.concatenate((train_images, batch['data']), axis=0)
        train_labels = np.concatenate((train_labels, batch['labels']), axis=0)

    for i in range(1, len(EVAL_FILE)):
        batch = unpickle(os.path.join(dataset_dir, TRAIN_FILE[i]))
        eval_images = np.concatenate((eval_images, batch['data']), axis=0)
        eval_labels = np.concatenate((eval_labels, batch['labels']), axis=0)
    if onehot_encoding:
        train_labels = onehot(train_labels)
        eval_labels = onehot(eval_labels)

    return train_images, eval_images, train_labels, eval_labels


class Cifar10(object):

    def __init__(self, images, lables):
        '''dataset_dir: the dir which saves the dataset files.
           onehot: if ont-hot encoding or not'''
        self._num_exzamples = len(lables)
        self._images = images
        self._labels = lables
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def num_exzamples(self):
        return self._num_exzamples

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_exzamples:
            self._epochs_completed += 1
            idx = np.arange(self._num_exzamples)
            np.random.shuffle(idx)
            self._images = self._images[idx, :]
            self._labels = self._labels[idx, :]
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self._images[start:end, :], self._labels[start:end, :]


def read_dataset(dataset_dir, onehot_encoding=False):
    class Datasets(object):
        pass
    dataset = Datasets()
    train_images, eval_images, train_labels, eval_labels = merge_data(dataset_dir, onehot_encoding)
    dataset.train = Cifar10(train_images, train_labels) 
    dataset.eval = Cifar10(eval_images, eval_labels)
    return dataset
