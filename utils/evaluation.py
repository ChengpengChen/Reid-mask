from __future__ import print_function, division

import numba
import numpy as np

from tqdm import tqdm
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import normalize

from utils.misc import load_lst


@numba.jit(nopython=True, nogil=True)
def compute_ap(good_index, junk_index, sort_index):
    cmc = np.zeros((len(sort_index),))
    n_good = len(good_index)

    old_recall = 0
    old_precision = 1.0
    ap = 0
    intersect_size = 0
    j = 0
    good_now = 0
    n_junk = 0
    for i in range(len(sort_index)):
        flag = 0
        if np.any(good_index == sort_index[i]):
            cmc[i - n_junk:] = 1
            flag = 1
            good_now = good_now + 1

        if np.any(junk_index == sort_index[i]):
            n_junk = n_junk + 1
            continue

        if flag == 1:
            intersect_size = intersect_size + 1

        recall = intersect_size / n_good if n_good > 0 else 0
        precision = intersect_size / (j + 1)
        ap = ap + (recall - old_recall) * ((old_precision + precision) / 2)
        old_recall = recall
        old_precision = precision
        j = j + 1

        if good_now == n_good:
            break

    return ap, cmc


def eval_feature(query_features, gallery_features, query_lst, gallery_lst, metric="euclidean"):
    if metric not in ["euclidean", "cosine"]:
        raise ValueError("Invalid metric! ")

    num_query = len(query_lst)
    num_gallery = len(gallery_lst)

    if metric == "cosine":
        gallery_features = normalize(gallery_features, axis=1)
        query_features = normalize(query_features, axis=1)
        # dist = np.dot(gallery_features, query_features.T)
        # print(dist[:].max())
        # print(dist[:].min())

    gallery_cam_lst = np.array([x.cam_id for x in gallery_lst], dtype=np.int32)
    gallery_id_lst = np.array([x.class_id for x in gallery_lst], dtype=np.int32)

    ap = np.zeros((num_query,))  # average precision
    cmc = np.zeros((num_query, num_gallery))

    index = np.arange(num_gallery)
    for i in tqdm(range(num_query)):
        q_feat = query_features[i]

        good_flag = np.logical_and((gallery_cam_lst != query_lst[i].cam_id), (gallery_id_lst == query_lst[i].class_id))
        junk_flag_1 = (gallery_id_lst == 0)
        junk_flag_2 = np.logical_and((gallery_cam_lst == query_lst[i].cam_id),
                                     (gallery_id_lst == query_lst[i].class_id))

        good_index = index[good_flag]
        junk_index = index[np.logical_or(junk_flag_1, junk_flag_2)]

        if metric == "euclidean":
            dist = euclidean_distances(gallery_features, [q_feat]).squeeze()
        else:
            dist = -np.dot(gallery_features, q_feat)
        sort_index = np.argsort(dist)

        ap[i], cmc[i, :] = compute_ap(good_index, junk_index, sort_index)

    map = np.mean(ap)
    r1 = np.mean(cmc, axis=0)[0]
    r5 = np.mean(np.clip(np.sum(cmc[:, :5], axis=1), 0, 1), axis=0)

    print('mAP = %f, r1 precision = %f, r5 precision = %f' % (map, r1, r5))

    return cmc


def eval_rank_list(rank_list, query_lst_path, gallery_lst_path):
    query_lst = load_lst(query_lst_path)
    gallery_lst = load_lst(gallery_lst_path)
    gallery_cam_lst = np.array([x.cam_id for x in gallery_lst], dtype=np.int32)
    gallery_id_lst = np.array([x.class_id for x in gallery_lst], dtype=np.int32)

    num_query = len(query_lst)
    num_gallery = len(gallery_lst)

    ap = np.zeros((num_query,))  # average precision
    cmc = np.zeros((num_query, num_gallery))
    for i in tqdm(range(num_query)):
        index = np.arange(num_gallery)
        good_flag = np.logical_and((gallery_cam_lst != query_lst[i].cam_id), (gallery_id_lst == query_lst[i].class_id))
        junk_flag_1 = (gallery_id_lst == 0)
        junk_flag_2 = np.logical_and((gallery_cam_lst == query_lst[i].cam_id),
                                     (gallery_id_lst == query_lst[i].class_id))

        good_index = index[good_flag]
        junk_index = index[np.logical_or(junk_flag_1, junk_flag_2)]

        sort_index = rank_list[i]

        ap[i], cmc[i, :] = compute_ap(good_index, junk_index, sort_index)

    map = np.mean(ap)
    r1 = np.mean(cmc, axis=0)[0]
    r5 = np.mean(np.clip(np.sum(cmc[:, :5], axis=1), 0, 1), axis=0)

    print('mAP = %f, r1 precision = %f, r5 precision = %f' % (map, r1, r5))
