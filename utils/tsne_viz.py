from __future__ import print_function, division
import os
import yaml
import mxnet as mx
import scipy.io as sio
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from collections import namedtuple

from utils.misc import load_lst, Record
from utils.group_iterator import GroupIterator

import operators.triplet_loss

TEMPERATURE = 1.0

Batch = namedtuple("Batch", ["data"])


def tsne_viz_query2gallery(prefix, dataset, viz_range):
    if not isinstance(viz_range, (list, tuple, int)):
        raise ValueError("viz_range must be list or tuple or int")
    if isinstance(viz_range, int):
        viz_range = list(range(viz_range))

    query_features_path = 'features/%s/query-%s.mat' % (dataset, prefix)
    gallery_features_path = "features/%s/gallery-%s.mat" % (dataset, prefix)
    query_lst_path = "/mnt/truenas/scratch/chuanchen.luo/data/reid/%s-list/query.lst" % dataset
    gallery_lst_path = "/mnt/truenas/scratch/chuanchen.luo/data/reid/%s-list/test.lst" % dataset

    query_lst = np.array(load_lst(query_lst_path))
    gallery_lst = np.array(load_lst(gallery_lst_path))

    query_features = sio.loadmat(query_features_path)["feat"]
    gallery_features = sio.loadmat(gallery_features_path)["feat"]

    query_features = normalize(query_features)
    gallery_features = normalize(gallery_features)

    for i in viz_range:
        q_feat = query_features[i]
        dist = -np.dot(gallery_features, q_feat)
        rank_list = np.argsort(dist)[:150]
        g_feats = gallery_features[rank_list]

        q_record = Record(*query_lst[i])
        g_records = [Record(*item) for item in gallery_lst[rank_list]]

        same_list = [i for i in range(g_feats.shape[0]) if q_record.class_id == g_records[i].class_id]
        diff_list = [i for i in range(g_feats.shape[0]) if q_record.class_id != g_records[i].class_id]

        init_embed = TSNE(n_components=2, init="pca", perplexity=10, metric="cosine").fit_transform(g_feats)

        W = np.dot(g_feats, g_feats.T)
        W = np.exp(W / TEMPERATURE)
        W = W / np.sum(W, axis=1, keepdims=True)

        g_feats = np.dot(W, g_feats)
        res_embed = TSNE(n_components=2, init="pca", perplexity=10, metric="cosine").fit_transform(g_feats)

        plt.figure(0)
        plt.scatter(init_embed[same_list, 0], init_embed[same_list, 1], label="same")
        plt.scatter(init_embed[diff_list, 0], init_embed[diff_list, 1], label="diff")
        plt.savefig("%d_tsne_orignal.png" % i)
        plt.close(0)

        plt.figure(1)
        plt.scatter(res_embed[same_list, 0], res_embed[same_list, 1], label="same")
        plt.scatter(res_embed[diff_list, 0], res_embed[diff_list, 1], label="diff")
        plt.savefig("%d_tsne_transformed_%.2f.png" % (i, TEMPERATURE))
        plt.close(1)

        print(i)


def tsne_viz_iteration(prefix, dataset, query_id):
    query_features_path = 'features/%s/query-%s.mat' % (dataset, prefix)
    gallery_features_path = "features/%s/gallery-%s.mat" % (dataset, prefix)
    query_lst_path = "/mnt/truenas/scratch/chuanchen.luo/data/reid/%s-list/query.lst" % dataset
    gallery_lst_path = "/mnt/truenas/scratch/chuanchen.luo/data/reid/%s-list/test.lst" % dataset

    query_lst = np.array(load_lst(query_lst_path))
    gallery_lst = np.array(load_lst(gallery_lst_path))

    query_features = sio.loadmat(query_features_path)["feat"]
    gallery_features = sio.loadmat(gallery_features_path)["feat"]

    query_features = normalize(query_features)
    gallery_features = normalize(gallery_features)

    q_feat = query_features[query_id]
    dist = -np.dot(gallery_features, q_feat)
    rank_list = np.argsort(dist)[:150]
    g_feats = gallery_features[rank_list]

    q_record = Record(*query_lst[query_id])
    g_records = [Record(*item) for item in gallery_lst[rank_list]]

    same_list = [i for i in range(g_feats.shape[0]) if q_record.class_id == g_records[i].class_id]
    diff_list = [i for i in range(g_feats.shape[0]) if q_record.class_id != g_records[i].class_id]

    init_embed = TSNE(n_components=2, init="pca", perplexity=8, metric="cosine").fit_transform(g_feats)

    W = np.dot(g_feats, g_feats.T)
    W = np.exp(W / TEMPERATURE)
    W = W / np.sum(W, axis=1, keepdims=True)

    plt.figure(0)
    plt.scatter(init_embed[same_list, 0], init_embed[same_list, 1], label="same")
    plt.scatter(init_embed[diff_list, 0], init_embed[diff_list, 1], label="diff")
    plt.savefig("%d_tsne_orignal.png" % query_id)
    plt.close(0)

    for iteration in range(10):
        g_feats = np.dot(W, g_feats)
        res_embed = TSNE(n_components=2, init="pca", perplexity=8, metric="cosine").fit_transform(g_feats)

        plt.figure(iteration)
        plt.scatter(res_embed[same_list, 0], res_embed[same_list, 1], label="same")
        plt.scatter(res_embed[diff_list, 0], res_embed[diff_list, 1], label="diff")
        plt.savefig("%d_%02d_tsne_transformed_%.2f.png" % (query_id, iteration, TEMPERATURE))
        plt.close(iteration)

        print("%d-%d" % (query_id, iteration))


def tsne_viz_train(p_size, k_size, prefix, dataset, epoch, gpu, num):
    with open("config.yml", "r") as f:
        args = yaml.load(f)

    data_dir = args[dataset]["data_dir"]
    crop_size = args["crop_size"]

    print(data_dir)

    sym, arg_params, aux_params = mx.model.load_checkpoint(os.path.join("models", dataset, prefix), epoch)
    sym = sym.get_internals()["flatten_output"]

    model = mx.mod.Module(symbol=sym, context=mx.gpu(gpu))
    model.bind(data_shapes=[("data", (128, 3, 256, 128))], for_training=False, force_rebind=True)
    model.set_params(arg_params, aux_params)

    dataloader = GroupIterator(data_dir=os.path.join(data_dir, "bounding_box_train"), p_size=p_size, k_size=k_size,
                               image_size=(crop_size * 2, crop_size), force_resize=(256, 128), random_seed=0)

    for i, batch in enumerate(dataloader):
        model.forward(Batch(data=batch.data))
        original = model.get_outputs()[0].asnumpy()

        original_embed = TSNE(n_components=2, init="pca", perplexity=8).fit_transform(original)

        original_l2 = normalize(original)
        W = np.dot(original_l2, original_l2.T)
        W = np.exp(W / TEMPERATURE)
        W = W / np.sum(W, axis=1, keepdims=True)

        transformed = np.dot(W, original)
        transformed_embed = TSNE(n_components=2, init="pca", perplexity=8).fit_transform(transformed)

        id_list = np.repeat(np.arange(p_size), k_size)
        plt.figure(0)
        plt.scatter(original_embed[:, 0], original_embed[:, 1], c=id_list, cmap="Set1", marker=".")
        plt.savefig("%03d_tsne_orignal.png" % i)
        plt.close(0)

        plt.figure(1)
        plt.scatter(transformed_embed[:, 0], transformed_embed[:, 1], c=id_list, cmap="Set1", marker=".")
        plt.savefig("%03d_tsne_transformed.png" % i)
        plt.close(1)

        print(i)
        if i > num:
            break


if __name__ == '__main__':
    prefix = "baseline-s16-erase-2loss-1.0"
    dataset = "duke"
    TEMPERATURE = 1.0

    # tsne_viz_query2gallery(prefix, dataset, 25)
    # tsne_viz_iteration(prefix, dataset, 0)

    tsne_viz_train(p_size=32, k_size=4, prefix=prefix, dataset=dataset, epoch=300, gpu=0, num=25)
