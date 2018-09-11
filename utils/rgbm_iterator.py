"""rgbm iterator, load all the data first, by chencp"""

import sys
sys.path.insert(0, "mxnet/python")
import mxnet as mx
import numpy as np
import cv2
import random


class RgbmIter(mx.io.DataIter):
    """
    data iterator, concat rgb and mask data
    """
    def __init__(self, rgb_lst_name, mask_lst_name, batch_size, aug_params=dict(),
                 shuffle=False, even_keep=True, data_type='rgbm', softmax_label=True,
                 mask_label=False, soft_or_hard=False, mask_shape=(55, 27)):
        """
        data_type: 'rgb' return data with 3 channels, 'rgbm' with 4 channels
        mask_label: if return mask as label
        soft_or_hard: as mask_shape, return mask as label if soft, as input if hard
        softmax_label: return softmax label or not
        """
        super(RgbmIter, self).__init__()
        self.batch_size = batch_size
        self.aug_params = aug_params.copy()
        # to rescale the rgb channels
        if 'img_scale' not in self.aug_params: 
            self.aug_params['img_scale'] = 1.0
        # to rescale the mask channel, no effect on mask output as label
        if 'mask_scale' not in self.aug_params:
            self.aug_params['mask_scale'] = 128.0
        self.shuffle = shuffle
        self.even_keep = even_keep
        self.current = 0
        self.mask_shape = mask_shape
        self.mask_label = mask_label
        self.soft_or_hard = soft_or_hard
        self.data_type = data_type
        self.softmax_label = softmax_label
        assert data_type == 'rgb' or data_type == 'rgbm', 'rgb or rgbm only'
        if self.data_type == 'rgb':
            assert self.aug_params["input_shape"][0] == 3, 'the channel of input should be 3 in the mode of rgb'
        else:
            assert self.aug_params["input_shape"][0] == 4, 'the channel of input should be 4 in the mode of rgbm'

        self.data, self.labels = self.load_data_rgbm(rgb_lst_name, mask_lst_name)
        self.data_num = self.labels.shape[0]
        self.label_num = 1 if len(self.labels.shape) == 1 else self.labels.shape[1]
        self.reset()

    def load_data_rgbm(self, rgb_lst_name, mask_lst_name):
        """load rgb and mask data, no shuffle"""
        data_rgb, labels_rgb = self.load_data(rgb_lst_name, channel=3)
        data_m, labels_m = self.load_data(mask_lst_name, channel=1)

        assert (labels_rgb == labels_m).all(), 'rgb and mask labels not consistent, check the list file'
        return np.concatenate([data_rgb, data_m], axis=3), labels_rgb

    @staticmethod
    def load_data(lst_name, channel):
        img_lst = [x.strip().split('\t') for x in file(lst_name).read().splitlines()]
        im = cv2.imread(img_lst[0][-1])

        h, w = im.shape[:2]
        n, m = len(img_lst), len(img_lst[0]) - 2
        data = np.zeros((n, h, w, channel), dtype=np.uint8)
        labels = np.zeros((n, m), dtype=np.int32) if m > 1 else np.zeros((n, ), dtype=np.int32)

        for i in range(len(img_lst)):
            im = cv2.imread(img_lst[i][-1])
            # for mask data
            if channel == 1:
                im_new = np.zeros_like(im[:, :, 0])  # take only one channel
                ind = np.where(im[:, :, 0] >= 128)
                im_new[ind] = 1
                im = im_new[:, :, np.newaxis]
                # print im.max()

            data[i] = im
            labels[i] = img_lst[i][1:-1] if m > 1 else img_lst[i][1]

        return data, labels

    @staticmethod
    def shuffle_data(labels, even_keep):
        """
        shuffle images lists and make pairs according to even_keep
        """
        if even_keep:
            idx = range(0, len(labels), 2)
            random.shuffle(idx)
            ret = []
            for i in idx:
                ret.append(i)
                ret.append(i+1)
        else:
            idx = range(len(labels))
            random.shuffle(idx)
            ret = []
            for i in idx:
                ret.append(i)
        return ret

    def reset(self):
        self.current = 0
        if self.shuffle:
            idx = self.shuffle_data(self.labels, self.even_keep)
            self.data = self.data[idx]
            self.labels = self.labels[idx]

    @property
    def provide_data(self):
        shape = self.aug_params['input_shape']
        shape_lst = [('data', (self.batch_size, shape[0], shape[1], shape[2]))]
        if self.mask_label and not self.soft_or_hard:
            # return mask as input
            shape_lst.append(('binary_label', (self.batch_size, 1, self.mask_shape[0], self.mask_shape[1])))

        return shape_lst

    @property
    def provide_label(self):
        label_shape = (self.batch_size, self.label_num) if self.label_num > 1 else (self.batch_size, )
        labels = [('softmax_label', label_shape)] if self.softmax_label else []
        if self.mask_label and self.soft_or_hard:
            # return mask as label
            labels.append(('binary_label', (self.batch_size, 1, self.mask_shape[0], self.mask_shape[1])))
        return labels

    @staticmethod
    def process_im(im, aug_params):
        """
        process the image, and take augmentation (resize, crop, mirror)
        """
        crop_h, crop_w = aug_params['input_shape'][1:]
        ori_h, ori_w = im.shape[:2]
        resize = aug_params['resize']
        if ori_h < ori_w:
            h, w = resize, int(float(resize) / ori_h * ori_w)
        else:
            h, w = int(float(resize) / ori_w * ori_h), resize

        if h != ori_h:
            # im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            im = cv2.resize(im, (w, h))

        x, y = (w - crop_w) / 2, (h - crop_h) / 2
        if aug_params['rand_crop']:
            x = random.randint(0, w - crop_w)
            y = random.randint(0, h - crop_h)
        im = im[y:y + crop_h, x:x + crop_w, :]

        im = np.transpose(im, (2, 0, 1))
        newim = np.zeros_like(im)
        newim[0] = im[2]
        newim[1] = im[1]
        newim[2] = im[0]
        newim[3] = im[3]

        if aug_params['rand_mirror'] and random.randint(0, 1) == 1:
            newim = newim[:, :, ::-1]

        return newim

    @staticmethod
    def resize_mask(data, shape=(55, 27)):
        """resize the binary mask to match the feature map shape"""
        data = np.transpose(data, (1, 2, 0))  # change to channel last
        shape = (shape[1], shape[0])  # change from (h, w) to (w, h) for cv2
        data_resize = cv2.resize(data, shape, interpolation=cv2.INTER_LINEAR)
        if data_resize.ndim == 2:
            # for channel=1, i.e., batch_size=1
            data_resize = data_resize[np.newaxis]
        else:
            data_resize = np.transpose(data_resize, (2, 0, 1))  # back to channel first
        data_resize = data_resize[:, np.newaxis]
        return mx.nd.array(data_resize)

    def next(self):
        if self.current + self.batch_size > self.data_num:
            raise StopIteration

        shape = self.aug_params['input_shape']
        x = np.zeros((self.batch_size, 4, shape[1], shape[2]))
        y = np.zeros((self.batch_size, self.label_num) if self.label_num > 1
                     else (self.batch_size, ))
        index = []
        for i in range(self.current, self.current + self.batch_size):
            im = self.data[i]
            im.astype(np.float32)
            im = self.process_im(im, self.aug_params)
            x[i - self.current] = im
            y[i - self.current] = self.labels[i]
            index.append(i)
        labels = [mx.nd.array(y)] if self.softmax_label else []

        x[:, :3] -= self.aug_params['mean']  # do not apply mean removing on mask
        x[:, :3] *= self.aug_params['img_scale']

        if self.data_type == 'rgb':
            img = x[:, :3]  # discard mask channel
        else:
            img = x.copy()
            # img[:, 3] -= 0.5
            img[:, 3] *= self.aug_params['mask_scale']
        data = [mx.nd.array(img)]

        if self.mask_label:
            binary_mask = x[:, 3]
            binary_mask_resize = self.resize_mask(binary_mask, self.mask_shape)
            if self.soft_or_hard:
                labels.append(binary_mask_resize)  # as label
            else:
                data.append(binary_mask_resize)  # as data

        batch = mx.io.DataBatch(data=data, label=labels, pad=0, index=np.array(index))
        self.current += self.batch_size

        return batch
