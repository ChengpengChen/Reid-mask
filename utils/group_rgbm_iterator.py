"""modify from group iterator for rgbm input, by chencp"""
from __future__ import print_function, division, absolute_import
import os
import cv2
import glob
import random

import mxnet as mx
import numpy as np

from threading import Thread
from collections import defaultdict

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from mxnet.io import DataBatch, DataIter

from utils.random_erasing import random_erasing


def pop(x, size):
    return [x.pop(0) for _ in range(size)]

# to generate the mask dir
mask_data_dir_root = '/home/chencp/dataset/annotation-market1501'
rgb_data_dir_root ='/home/chencp/dataset/Market-1501-v15.09.15'

class GroupIteratorRGBM(DataIter):
    def __init__(self, data_dir, p_size, k_size, image_size, rand_mirror=False, rand_crop=False, random_erasing=False,
                 force_resize=None, resize_shorter=None, num_worker=4, random_seed=None,
                 data_type='rgbm', mask_label=False, soft_or_hard=False, mask_shape=(64, 32)):

        if random_seed is None:
            random_seed = random.randint(0, 2 ** 32 - 1)
        np.random.RandomState(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.random_seed = random_seed

        self.p_size = p_size
        self.k_size = k_size
        self.batch_size = p_size * k_size
        self.image_size = image_size
        self.rand_mirror = rand_mirror
        self.rand_crop = rand_crop
        self.random_erasing = random_erasing
        self.resize_shorter = resize_shorter
        self.num_worker = num_worker
        self.force_resize = tuple(force_resize) if force_resize is not None else force_resize
        self.img_list = None
        self.cursor = 0
        self.size = 0

        self.mask_shape = mask_shape
        self.mask_label = mask_label
        self.soft_or_hard = soft_or_hard
        assert data_type == 'rgb' or data_type == 'rgbm', 'rgb or rgbm only'
        # True as rgb and False as rgbm
        self.rgb_or_rgbm = True if data_type == 'rgb' else False
        # self.mask_data_dir = mask_data_dir
        # if not self.mask_data_dir:
        #     self.mask_data_dir = data_dir.replace(rgb_data_dir_root, mask_data_dir_root)

        resize_set = {self.resize_shorter, self.force_resize}
        assert len(resize_set) == 2 and None in resize_set, "resize_shorter and force_resize are mutually exclusive!"

        if self.force_resize is not None:
            if self.force_resize[0] < self.image_size[0] or self.force_resize[1] < self.image_size[1]:
                raise ValueError("each edge of force_resize must be larger than or equal to target image size")

        if self.resize_shorter is not None:
            if self.resize_shorter < min(self.image_size):
                raise ValueError("resize_shorter must be larger than or equal to minimum of target image side size")

        print("Data loading..")
        self.id2imgs = self._preprocess(data_dir)
        print("Data loaded!")

        self.num_id = len(self.id2imgs)
        print(self.num_id)

        # multi-thread primitive
        self.result_queue = Queue(maxsize=8 * num_worker)
        self.index_queue = Queue()
        self.workers = None

        self._thread_start(num_worker)

        self.reset()

    def _preprocess(self, data_dir):
        img_list = glob.glob(os.path.join(data_dir, "*.jpg")) + glob.glob(os.path.join(data_dir, "*.png"))
        img_list.sort()
        self.img_list = img_list

        assert len(img_list) != 0

        id2imgs = defaultdict(list)
        for img_name in img_list:
            idx = int(os.path.basename(img_name).split("_")[0])
            id2imgs[idx].append(img_name)

        id2imgs.pop(0, None)
        id2imgs.pop(-1, None)

        id2imgs_organized = {}
        for i, v in enumerate(id2imgs.values()):
            id2imgs_organized[i] = v

        return id2imgs_organized

    def _insert_queue(self):
        data = []

        # pids = list(self.id2imgs.keys()).copy()
        pids = list(self.id2imgs.keys())[:]
        random.shuffle(pids)

        sample_range = list(range(0, len(pids), self.p_size))[:-1]
        self.size = len(sample_range)

        for i in sample_range:
            start = i
            stop = start + self.p_size

            for pid in pids[start:stop]:
                if len(self.id2imgs[pid]) >= self.k_size:
                    imgs = np.random.choice(self.id2imgs[pid], replace=False, size=self.k_size)
                else:
                    imgs = np.random.choice(self.id2imgs[pid], replace=True, size=self.k_size)

                data.extend(imgs)

            label = np.array(pids[start:stop]).repeat(self.k_size)

            # self.index_queue.put([data.copy(), label])
            self.index_queue.put([data[:], label])

            assert len(data) == self.batch_size
            assert len(label) == self.batch_size

            del data[:]

    def _thread_start(self, num_worker):
        self.workers = [Thread(target=self._worker, args=(self.random_seed + i,)) for i in range(num_worker)]

        for worker in self.workers:
            worker.daemon = True
            worker.start()

    def _worker(self, seed):
        np.random.RandomState(seed)
        np.random.seed(seed)
        random.seed(seed)

        while True:
            indices = self.index_queue.get()
            result = self._get_batch(indices=indices)

            if result is None:
                return

            self.result_queue.put(result)

    def _get_batch(self, indices):
        img_paths = indices[0]
        label = indices[1]

        # Loading rgb data and id labels
        data = []
        for img_path in img_paths:
            img = cv2.imread(img_path)
            cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB, dst=img)
            img = img.astype(np.float32)
            ori_h, ori_w = img.shape[:2]

            # loading mask data and concatenated at the fourth channel
            if self.mask_label or not self.rgb_or_rgbm:
                img_mask_path = img_path.replace(rgb_data_dir_root, mask_data_dir_root)
                img_mask_path = img_mask_path.replace('.jpg', '.png')
                ind = img_mask_path.find('/', len(mask_data_dir_root)+1)
                img_mask_path = img_mask_path[:ind] + '_seg' + img_mask_path[ind:]
                if not os.path.exists(img_mask_path):
                    print(img_mask_path)
                    raise AssertionError, 'mask data path error'
                mask = cv2.imread(img_mask_path)

                im_new = np.zeros_like(mask[:, :, 0])  # take only one channel
                ind = np.where(mask[:, :, 0] >= 128)
                im_new[ind] = 255
                im_new = cv2.resize(im_new, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
                im = im_new[:, :, np.newaxis]
                img = np.concatenate([img, im], axis=2)

            if self.force_resize is not None:
                img = cv2.resize(img, self.force_resize[::-1], interpolation=cv2.INTER_LINEAR)
            else:
                if ori_h < ori_w:
                    h, w = self.resize_shorter, int(round(self.resize_shorter / ori_h * ori_w))
                else:
                    h, w = int(round(self.resize_shorter / ori_w * ori_h)), self.resize_shorter

                if h != ori_h or w != ori_w:
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

            h, w = img.shape[:2]
            x, y = int(round((w - self.image_size[1]) / 2)), int(round((h - self.image_size[0]) / 2))

            # random crop
            if self.rand_crop:
                x = random.randint(0, w - self.image_size[1])
                y = random.randint(0, h - self.image_size[0])

            if self.image_size != (h, w):
                img = img[y:y + self.image_size[0], x:x + self.image_size[1], :]

            # random mirror
            if self.rand_mirror and random.randint(0, 1) == 1:
                img = cv2.flip(img, flipCode=1)

            if self.random_erasing:
                img = random_erasing(img)

            data.append(img)

        data = np.stack(data, axis=0).transpose([0, 3, 1, 2])
        # remove mask channel
        if self.rgb_or_rgbm:
            data_rgb = data[:, :3]
        else:
            data_rgb = data
        assert data_rgb.shape[0] == label.shape[0] == self.batch_size

        data_all = [mx.nd.array(data_rgb)]
        label_all = [mx.nd.array(label)]

        # transform mask data, and set as data or label according to soft_or_hard
        if self.mask_label:
            mask_data = data[:, 3].transpose([1, 2, 0])
            h, w = self.mask_shape
            mask_data = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_data = mask_data.transpose([2, 0, 1])[:, np.newaxis]
            mask_data /= np.max(mask_data)
            mask_data = mx.nd.array(mask_data)
            if self.soft_or_hard:
                label_all.append(mask_data)
            else:
                data_all.append(mask_data)

        return data_all, label_all

    @property
    def provide_data(self):
        if self.rgb_or_rgbm:
            data = [('data', (self.batch_size, 3, self.image_size[0], self.image_size[1]))]
        else:
            data = [('data', (self.batch_size, 4, self.image_size[0], self.image_size[1]))]
        if self.mask_label and not self.soft_or_hard:
            # return mask as input
            data.append(('binary_label', (self.batch_size, 1, self.mask_shape[0], self.mask_shape[1])))

        return data

    @property
    def provide_label(self):
        labels = [('softmax_label', (self.batch_size,))]
        if self.mask_label and self.soft_or_hard:
            # return mask as label
            labels.append(('binary_label', (self.batch_size, 1, self.mask_shape[0], self.mask_shape[1])))
        return labels

    def reset(self):
        self.cursor = 0
        self.index_queue.queue.clear()

        self._insert_queue()

    def next(self):
        if self.cursor >= self.size:
            raise StopIteration
        data, label = self.result_queue.get()
        self.cursor += 1

        # import pdb
        # pdb.set_trace()
        return DataBatch(data=data, label=label, provide_data=self.provide_data, provide_label=self.provide_label)


if __name__ == '__main__':
    import time

    # for rgbm iter
    mask_label = True
    soft_or_hard = False
    data_type = 'rgbm'

    data_dir = "/home/chencp/dataset/Market-1501-v15.09.15"
    train = GroupIteratorRGBM(data_dir=os.path.join(data_dir, "bounding_box_train"),
                          p_size=64, k_size=4, image_size=(128, 64),
                          force_resize=(128, 64), rand_mirror=True, rand_crop=False, random_erasing=False, num_worker=8,
                          data_type=data_type, mask_label=mask_label, soft_or_hard=soft_or_hard)

    tic = time.time()
    tmp = []
    for i in range(100):
        print(i)
        batch = train.next()
        tmp.append(batch.label[0].asnumpy().tolist())
        print(batch.data[0].shape)
        print(batch.label[0].shape)
        print(batch.label[0])
        imgs = batch.data[0].transpose([0, 2, 3, 1]).asnumpy()

        print(imgs.dtype)
        for j in range(imgs.shape[0]):
            img = imgs[j]
            cv2.imwrite("../save_img/%d-%d.jpg" % (i, j), img[:, :, [2, 1, 0]].astype(np.uint8))
            if data_type == 'rgbm':
                mask = img[:, :, 3]
                mask = mask / np.max(mask) * 255
                cv2.imwrite("../save_img/%d-%d-mask-4channel.jpg" % (i, j), mask.astype(np.uint8))
            if mask_label:
                if soft_or_hard:
                    mask = batch.label[1][j].asnumpy()
                else:
                    mask = batch.data[1][j].asnumpy()
                mask = mask.transpose([1, 2, 0]) * 255
                cv2.imwrite("../save_img/%d-%d-mask.jpg" % (i, j), mask.astype(np.uint8))
            import pdb
            pdb.set_trace()

    print((time.time() - tic) / 100)
