# create data loader
# by ccp, on 2018.08.17

import tensorflow as tf
from PIL import Image
import numpy as np
import os
try:
    import pickle
except:
    import Pickle as pickle

class ROIdatalayer(object):
    def __init__(self, img_lst, roi_file, batch_size, epoch, flip=False, max_roi_num=30, nms=0.3, enlarge_factor=0.2):
        assert os.path.isfile(img_lst), 'image list not exist: %d' % img_lst
        assert os.path.isfile(roi_file), 'roi file not exist: %d' % roi_file
        self.img_lst = img_lst
        self.roi_file = roi_file
        self.batch_size = batch_size
        self.epoch = epoch
        self.flip = flip
        self.max_roi_num = max_roi_num
        self.nms = nms
        # to enlarge roi to include more context
        self.enlarge_factor = enlarge_factor

        self.roi = self._load_roi()
        self.iterator = self._create_iter()


    def _load_roi(self):
        with open(self.roi_file, 'rb') as f:
            rois = pickle.load(f)

    def _parse(self, filename, label):
        """ Reading and resize image"""
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        image_resized = tf.image.resize_images(image_decoded, [28, 28])
        return image_resized, label

    def _create_iter(self):
        file_list = []
        filenames = tf.constant(file_list)
        labels = tf.constant([len(file_list)])

        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(self._parse)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat(self.epoch)
        iterator = dataset.make_one_shot_iterator()

        return iterator

    def next(self):
        return self.iterator.get_next()

