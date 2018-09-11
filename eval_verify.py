"""
evaluate the occlusion under the verification setting

extract features first, compute similarities and then calculate the verification metric

2018.08.31, by chencp
"""
import os
import yaml
import numpy as np
import scipy.io as sio
from pprint import pprint
from utils.misc import load_lst
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import normalize
# from utils.evaluation import eval_feature


def compute_ap(dist_label_new, threshold):
    old_recall = 0
    old_precision = 1.0
    ap = 0
    j = 0
    good_now = 0
    intersect_size = 0
    n_good = np.sum(dist_label_new[:, 1])
    l = dist_label_new[:, 0] >= threshold
    l_2 = np.logical_and(dist_label_new[:, 1], l)
    print(np.sum(l_2))
    # import pdb
    # pdb.set_trace()
    for i in np.arange(len(l)):
        if dist_label_new[i, 1] and l[i]:
            intersect_size += 1
            good_now += 1
        j += 1
        recall = intersect_size / n_good
        precision = intersect_size / j
        ap += (recall - old_recall) * ((old_precision + precision) / 2)
        old_recall = recall
        old_precision = precision
        if good_now == n_good:
            break
    return ap


def eval_feature_verify(query_features, gallery_features, query_lst, gallery_lst, metric="cosine"):
    """evaluate the extracted features from query and test set with verification metric"""
    if metric not in ["euclidean", "cosine"]:
        raise ValueError("Invalid metric! ")

    num_query = len(query_lst)
    num_gallery = len(gallery_lst)

    if metric == "cosine":
        gallery_features = normalize(gallery_features, axis=1)
        query_features = normalize(query_features, axis=1)

    if metric == "euclidean":
        dist = euclidean_distances(gallery_features, query_features).squeeze()
    else:
        dist = np.dot(gallery_features, query_features.T)
        print(dist[:].max())
        print(dist[:].min())

    gallery_cam_lst = np.array([x.cam_id for x in gallery_lst], dtype=np.int32)
    gallery_id_lst = np.array([x.class_id for x in gallery_lst], dtype=np.int32)
    query_cam_lst = np.array([x.cam_id for x in query_lst], dtype=np.int32)
    query_id_lst = np.array([x.class_id for x in query_lst], dtype=np.int32)

    # same id
    gallery_id_lst_tile = np.tile(gallery_id_lst, (num_query, 1))
    query_id_lst_tile = np.tile(query_id_lst, (num_gallery, 1))
    label_mat = query_id_lst_tile == gallery_id_lst_tile.T  # gallery num * query num
    label_mat = label_mat.astype(np.int)

    # same camera and id
    gallery_cam_lst_tile = np.tile(gallery_cam_lst, (num_query, 1))
    query_cam_lst_tile = np.tile(query_cam_lst, (num_gallery, 1))
    # same camera, gallery num * query num
    label_cam_mat = query_cam_lst_tile == gallery_cam_lst_tile.T
    # filter by same id
    label_cam_mat = label_cam_mat * label_mat
    # same id but under same camera are labeled as -1,
    label_mat -= 2 * label_cam_mat.astype(np.int)

    ind = np.where(label_mat == 0)
    neg_dist = dist[ind]
    ind = np.where(label_mat == 1)
    pos_dist = dist[ind]

    new_dist = np.concatenate([pos_dist, neg_dist])
    new_label = np.concatenate([np.ones_like(pos_dist), np.zeros_like(neg_dist)])
    dist_label = zip(new_dist, new_label)
    dist_label_new = np.array(sorted(dist_label, key=lambda d: -d[0]))

    # ap = []
    n_good = np.sum(dist_label_new[:, 1])
    n_bad = np.sum(1 - dist_label_new[:, 1])
    print(n_good)
    for thresh in np.arange(0.90, 1, 0.01):
        # print(thresh)
        p = dist_label_new[:, 0] >= thresh
        if np.sum(p) == 0:
            recall, precision, f1 = 0, 0, 0
        else:
            tp = np.logical_and(dist_label_new[:, 1], p)
            recall = np.sum(tp) / n_good
            precision = np.float(np.sum(tp)) / np.sum(p)

            f1 = 2*recall*precision/(recall+precision)

        print('%.4f\t%.4f\t%.4f' % (recall, precision, f1))

        # ap.append(compute_ap(dist_label_new, thresh))
    # print(ap)


if __name__ == '__main__':
    import subprocess
    import sys
    import argparse

    re_extract = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=2, type=int)
    parser.add_argument("--model_path", type=str, default='models/market/rgbm_baseline_0823-v1-300.params')

    parser.add_argument("--data_type", type=str, default='rgb')
    parser.add_argument("--mask_label", default=False, action="store_true")
    parser.add_argument("--soft_or_hard", default=False, action="store_true")
    parser.add_argument("--rgbm_iter", default=False, action="store_true")

    args = parser.parse_args()

    model_path = args.model_path
    basename = os.path.splitext(os.path.basename(model_path))[0]
    prefix = "-".join(basename.split("-")[:-1])
    epoch_idx = int(basename.split("-")[-1])
    gpu = args.gpu
    dataset = model_path.split("/")[1]
    force_resize = True

    # data_type = args.data_type
    # mask_label = args.mask_label
    # soft_or_hard = args.soft_or_hard
    # rgbm_iter =args.rgbm_iter
    data_type = 'rgbm'
    mask_label = False
    soft_or_hard = False
    rgbm_iter = True
    fea_tag = 'verify_v1'  # to indicate the feature files
    # fea_tag = ''

    print("%s/%s-%d" % (dataset, prefix, epoch_idx))

    query_features_path = 'features/%s/query-%s%s.mat' % (dataset, prefix, fea_tag)
    gallery_features_path = "features/%s/gallery-%s%s.mat" % (dataset, prefix, fea_tag)
    # gallery_prefix = "/home/chencp/data/%s-list/test2" % dataset
    gallery_prefix = "/home/chencp/dataset/binary-annotation-market1501/lst_dir/test_fuse_verify"
    # gallery_prefix = "/home/chencp/dataset/binary-annotation-market1501/lst_dir/test_fuse_v2"
    # query_prefix = "/home/chencp/data/%s-list/query2" % dataset
    query_prefix = "/home/chencp/dataset/binary-annotation-market1501/lst_dir/query_fuse_v2"
    # query_prefix = "/home/chencp/dataset/binary-annotation-market1501/lst_dir/query_fuse_verify"

    if re_extract or not (os.path.exists(query_features_path) and os.path.exists(gallery_features_path)):
        # --data_type rgbm --mask_label --soft_or_hard --rgbm_iter
        cmd = "python%c extract.py --prefix %s --gpu-id %d --epoch-idx %d --query-prefix %s --gallery-prefix %s " \
              "--crop-size 128 --dataset %s" % (
            sys.version[0], prefix, gpu, epoch_idx, query_prefix, gallery_prefix, dataset)
        cmd = cmd + ' --data_type %s' % data_type
        if force_resize:
            cmd = cmd + " --force-resize"
        if mask_label:
            cmd = cmd + ' --mask_label'
        if soft_or_hard:
            cmd = cmd + ' --soft_or_hard'
        if rgbm_iter:
            cmd = cmd + ' --rgbm_iter'
        if fea_tag:
            cmd = cmd + ' --fea_tag %s' % fea_tag

        subprocess.check_call(cmd.split(" "))

    assert os.path.exists(query_features_path) and os.path.exists(gallery_features_path)

    query_features = sio.loadmat(query_features_path)["feat"]
    gallery_features = sio.loadmat(gallery_features_path)["feat"]

    query_lst = load_lst(query_prefix + ".lst")
    gallery_lst = load_lst(gallery_prefix + ".lst")

    eval_feature_verify(query_features, gallery_features, query_lst, gallery_lst, metric="cosine")
