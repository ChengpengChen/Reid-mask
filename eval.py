import os
import yaml
import scipy.io as sio
from pprint import pprint
from utils.misc import load_lst
from utils.evaluation import eval_feature

if __name__ == '__main__':
    import subprocess
    import sys
    import argparse

    re_extract = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument("--model_path", type=str, default='models/market/channel_att_rgb_share_0908-300.params')

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
    data_type = 'rgb'
    mask_label = False
    soft_or_hard = False
    rgbm_iter = True
    fea_tag = ''  # to indicate the feature files
    # fea_tag = ''

    print("%s/%s-%d" % (dataset, prefix, epoch_idx))

    query_features_path = 'features/%s/query-%s%s.mat' % (dataset, prefix, fea_tag)
    gallery_features_path = "features/%s/gallery-%s%s.mat" % (dataset, prefix, fea_tag)
    gallery_prefix = "/home/chencp/data/%s-list/test2" % dataset
    # gallery_prefix = "/home/chencp/dataset/binary-annotation-market1501/lst_dir/test_fuse_v2"
    query_prefix = "/home/chencp/data/%s-list/query2" % dataset
    # query_prefix = "/home/chencp/dataset/binary-annotation-market1501/lst_dir/query_cls_test"
    # query_prefix = "/home/chencp/dataset/binary-annotation-market1501/lst_dir/query_fuse_v3"

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

    cmc = eval_feature(query_features, gallery_features, query_lst, gallery_lst, metric="cosine")
    import numpy as np
    np.save('tmp/{}-cmc'.format(prefix), cmc)
