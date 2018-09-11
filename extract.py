from __future__ import print_function

import os
import numpy as np
import mxnet as mx

from collections import namedtuple
# operators
import operators.triplet_loss
import operators.triplet_loss_inner
import operators.loss_layers_mask
from utils.rgbm_iterator import RgbmIter
Batch = namedtuple('Batch', ['data'])


def get_iterator_rgbm(batch_size, data_shape, lst_path, mask_label, soft_or_hard, force_resize, mask_shape=(64, 32)):
    print('use rgbm iterator')
    if data_type == 'rgb' and data_shape[0] != 3:
        data_shape = (3, data_shape[1], data_shape[2])
        print('change the input channel of data to 3, in order to load to load rgb data')
    aug_params = {}
    aug_params['resize'] = data_shape[2] if force_resize else 144  # to force resize
    # aug_params['resize'] = 144  # to force resize
    aug_params['rand_crop'] = False
    aug_params['rand_mirror'] = False
    aug_params['input_shape'] = data_shape
    aug_params['mean'] = 0.
    aug_params['img_scale'] = 1.
    aug_params['mask_scale'] = 255.

    iterator = RgbmIter(
        lst_path,
        lst_path.replace('.lst', '-mask.lst'),
        batch_size=batch_size,
        aug_params=aug_params,
        shuffle=False,
        even_keep=False,
        data_type=data_type,
        mask_label=mask_label,
        soft_or_hard=soft_or_hard,
        mask_shape=mask_shape)

    return iterator


def get_iterator(batch_size, data_shape, lst_path, rec_path, force_resize):
    print("FORCE_RESIZE: %s" % force_resize)
    if not force_resize:
        iterator = mx.io.ImageRecordIter(
            path_imglist=lst_path,
            path_imgrec=rec_path,
            rand_crop=False,
            rand_mirror=False,
            prefetch_buffer=8,
            preprocess_threads=4,
            shuffle=False,
            label_width=1,
            round_batch=False,
            resize=data_shape[1],
            data_shape=data_shape,
            batch_size=batch_size)
    else:
        aug_list = [
            mx.image.CastAug(),
            mx.image.ForceResizeAug(data_shape[1:][::-1], interp=1),
            mx.image.ColorNormalizeAug(mean=[0, 0, 0], std=[1, 1, 1])
        ]

        iterator = mx.image.ImageIter(
            batch_size=batch_size,
            data_shape=data_shape,
            label_width=1,
            shuffle=False,
            path_imgrec=rec_path,
            path_imglist=lst_path,
            aug_list=aug_list
        )

    return iterator


def extract_feature(model, iterator_or_data, dataset_size=None):
    feature = None

    if isinstance(iterator_or_data, mx.io.DataIter):
        iterator = iterator_or_data

        data_shape = iterator.provide_data
        model.reshape(data_shape)

        batch_size = data_shape[0][1][0]
        num_iter = dataset_size // batch_size
        extra = dataset_size % batch_size

        feature = []
        iterator.reset()
        for i, batch in enumerate(iterator):
            if i < num_iter:
                data = batch.data
            else:
                data_shape = [("data", (extra,) + data_shape[0][1][1:])]
                model.reshape(data_shape)
                data = batch.data[:extra]

            model.forward(Batch(data=data), is_train=False)
            output = model.get_outputs()[0]
            output = output.asnumpy()
            feature.append(output)
            # import pdb
            # pdb.set_trace()

        feature = np.concatenate(feature, axis=0)

    elif isinstance(iterator_or_data, mx.nd.NDArray):
        data = iterator_or_data

        if data.ndim == 3:
            data = data.expand_dims(0)

        model.reshape([("data", data.shape)])

        model.forward(Batch(data=[data]), is_train=False)
        feature = model.get_outputs()[0].asnumpy()

    return feature


if __name__ == '__main__':
    import scipy.io as sio
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--prefix", type=str, default='rgbm_hard_mask_0814-v1')
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--epoch-idx", type=int, default=190)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--query-prefix", type=str, default="/home/chencp/data/market-list/query2")
    parser.add_argument("--gallery-prefix", type=str, default="/home/chencp/data/market-list/test2")
    parser.add_argument("--dataset", type=str, default='market')
    parser.add_argument("--force-resize", action="store_true")

    parser.add_argument("--data_type", type=str, default='rgb')
    parser.add_argument("--mask_label", default=False, action="store_true")
    parser.add_argument("--soft_or_hard", default=False, action="store_true")
    parser.add_argument("--rgbm_iter", default=False, action="store_true")

    parser.add_argument("--fea_tag", type=str, default='')

    args = parser.parse_args()

    force_resize = args.force_resize
    batch_size = args.batch_size
    crop_size = args.crop_size
    query_lst_path = args.query_prefix + ".lst"
    query_rec_path = args.query_prefix + ".rec"
    gallery_lst_path = args.gallery_prefix + ".lst"
    gallery_rec_path = args.gallery_prefix + ".rec"

    # mask_shape = (64, 32)
    mask_shape = (256, 128)  # for the no share attention part
    data_type = args.data_type
    mask_label = args.mask_label
    soft_or_hard = args.soft_or_hard
    rgbm_iter = True if data_type == 'rgbm' else args.rgbm_iter
    batch_size = 4 if rgbm_iter else batch_size
    slice_flatten = True

    fea_tag = args.fea_tag

    context = mx.gpu(args.gpu_id)

    load_model_prefix = "models/%s" % args.prefix if args.dataset is None \
        else "models/%s/%s" % (args.dataset, args.prefix)
    symbol, arg_params, aux_params = mx.model.load_checkpoint(load_model_prefix, args.epoch_idx)
    flatten = symbol.get_internals()["flatten_output"]
    if slice_flatten:
        flatten_sym = mx.symbol.slice_axis(data=flatten, axis=0, begin=0, end=batch_size)
    else:
        flatten_sym = flatten

    data_shape = (3, crop_size * 2, crop_size) if data_type == 'rgb' else (4, crop_size * 2, crop_size)
    data_shapes = [('data', (batch_size, data_shape[0], crop_size * 2, crop_size))]
    data_names = ['data']
    if rgbm_iter and mask_label and not soft_or_hard:
        data_shapes.append(('binary_label', (batch_size, 1, mask_shape[0], mask_shape[1])))
        data_names.append('binary_label')  # as data

    model = mx.mod.Module(symbol=flatten_sym, context=context, data_names=data_names, label_names=None)
    model.bind(data_shapes=data_shapes, for_training=False, force_rebind=True)

    model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

    # extract query feature
    q_len = len(open(query_lst_path).read().splitlines())
    print(q_len)
    if not rgbm_iter:
        q_iterator = get_iterator(batch_size=batch_size,
                                  data_shape=(3, crop_size * 2, crop_size),
                                  lst_path=query_lst_path,
                                  rec_path=query_rec_path,
                                  force_resize=force_resize)
    else:
        q_iterator = get_iterator_rgbm(batch_size=batch_size,
                                       data_shape=data_shape,
                                       lst_path=query_lst_path,
                                       mask_label=mask_label,
                                       soft_or_hard=soft_or_hard,
                                       force_resize=force_resize,
                                       mask_shape=mask_shape)

    q_feat = extract_feature(model, q_iterator, q_len)
    print(q_feat.shape)

    feat_root = "features/" if args.dataset is None else "features/" + args.dataset

    save_name = "{}/query-{}{}".format(feat_root, args.prefix, fea_tag)
    sio.savemat(save_name, {"feat": q_feat})

    # extract gallery feature
    g_len = len(open(gallery_lst_path).read().splitlines())
    print(g_len)
    save_name = "{}/gallery-{}{}".format(feat_root, args.prefix, fea_tag)
    # if not os.path.exists(save_name+'.mat'):
    if not rgbm_iter:
        g_iterator = get_iterator(batch_size=batch_size,
                                  data_shape=(3, crop_size * 2, crop_size),
                                  lst_path=gallery_lst_path,
                                  rec_path=gallery_rec_path,
                                  force_resize=force_resize)
    else:
        g_iterator = get_iterator_rgbm(batch_size=batch_size,
                                       data_shape=data_shape,
                                       lst_path=gallery_lst_path,
                                       mask_label=mask_label,
                                       soft_or_hard=soft_or_hard,
                                       force_resize=force_resize,
                                       mask_shape=mask_shape)

    g_feat = extract_feature(model, g_iterator, g_len)
    print(g_feat.shape)

    sio.savemat(save_name, {"feat": g_feat})
