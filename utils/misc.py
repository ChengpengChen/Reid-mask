import os
import glob
import cv2
import numpy as np
import mxnet as mx

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

from collections import namedtuple

Record = namedtuple('Record', ['index', 'class_id', 'cam_id', 'img_path'])


def clean_immediate_checkpoints(model_dir, prefix, final_epoch):
    ckpts = glob.glob(os.path.join(model_dir, "%s*.params" % prefix))

    for ckpt in ckpts:
        ckpt_name = os.path.basename(ckpt)
        epoch_idx = int(ckpt_name[:ckpt_name.rfind(".")].split("-")[-1])
        if epoch_idx < final_epoch:
            os.remove(ckpt)


def euclidean_dist(x, y, eps=1e-12, squared=False):
    m, n = x.shape[0], y.shape[0]
    xx = mx.nd.power(x, 2).sum(axis=1, keepdim=True).broadcast_to([m, n])
    yy = mx.nd.power(y, 2).sum(axis=1, keepdim=True).broadcast_to([n, m]).T
    dist = xx + yy
    dist = dist - 2 * mx.nd.dot(x, y.T)
    dist = mx.nd.clip(dist, eps, np.inf)
    return dist if not squared else mx.nd.sqrt(dist)


def viz_heatmap(data, name):
    maximum = np.max(data)
    # minimun = np.min(data)

    plt.figure(0)

    data = data / maximum * 255
    heat_map = cv2.applyColorMap(data.astype(np.uint8), cv2.COLORMAP_JET)[:, :, [2, 1, 0]]
    plt.imshow(heat_map)
    plt.savefig("%s.jpg" % name)

    plt.close(0)


def load_lst(lst_path):
    lines = []
    with open(lst_path) as fin:
        for line in fin:
            idx, class_id, img_path = line.strip().split('\t')

            cam_id = img_path.rsplit('/')[-1].split('.')[0].split('_')[1][1]
            idx = int(idx)
            class_id = int(class_id)
            cam_id = int(cam_id)
            lines.append(Record(idx, class_id, cam_id, img_path))
    return lines
