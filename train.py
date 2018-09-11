from __future__ import print_function, division

import os
import logging
import yaml
import importlib
import mxnet as mx
import numpy as np
from easydict import EasyDict
from pprint import pprint
from scipy.linalg import block_diag

from utils.misc import clean_immediate_checkpoints
from utils.group_iterator import GroupIterator
from utils.initializer import InitWithArray
from utils.debug import forward_debug
from utils.memonger import search_plan
from utils.lr_scheduler import WarmupMultiFactorScheduler, ExponentialScheduler

# operators
import operators.triplet_loss


def build_network(symbol, num_id, p_size, **kwargs):
    triplet_normalization = kwargs.get("triplet_normalization", False)
    use_triplet = kwargs.get("use_triplet", False)
    use_softmax = kwargs.get("use_softmax", False)
    triplet_margin = kwargs.get("triplet_margin", 0.5)

    label = mx.symbol.Variable(name="softmax_label")
    group = [label]

    pooling = mx.symbol.Pooling(data=symbol, kernel=(1, 1), global_pool=True, pool_type='max', name='global_pool')
    flatten = mx.symbol.Flatten(data=pooling, name='flatten')

    # triplet loss
    if use_triplet:
        data_triplet = mx.sym.L2Normalization(flatten, name="triplet_l2") if triplet_normalization else flatten
        triplet = mx.symbol.Custom(data=data_triplet, p_size=p_size, margin=triplet_margin, op_type='TripletLoss',
                                   name='triplet')
        group.append(triplet)

    # softmax cross entropy loss
    if use_softmax:
        fc = mx.symbol.FullyConnected(flatten, num_hidden=bottleneck_dims, name="bottleneck")
        bn = mx.sym.BatchNorm(data=fc, fix_gamma=False, momentum=0.9, eps=2e-5, name='bottleneck_bn')
        relu = mx.sym.Activation(data=bn, act_type='relu', name='bottleneck_relu')
        # relu = bn

        dropout = mx.symbol.Dropout(relu, p=dropout_ratio)

        softmax_w = mx.symbol.Variable("softmax_weight", shape=(num_id, bottleneck_dims))

        if softmax_weight_normalization:
            softmax_w = mx.symbol.L2Normalization(softmax_w, name="softmax_weight_norm")

        if softmax_feat_normalization:
            data_softmax = mx.sym.L2Normalization(dropout, name="softmax_data_norm") * norm_scale
        else:
            data_softmax = dropout

        softmax_fc = mx.symbol.FullyConnected(data_softmax, weight=softmax_w, num_hidden=num_id,
                                              no_bias=True if softmax_weight_normalization else False,
                                              name="softmax_fc")

        softmax = mx.symbol.SoftmaxOutput(data=softmax_fc, label=label, name='softmax')

        group.append(softmax)


    return mx.symbol.Group(group)


def get_iterators(data_dir, p_size, k_size, crop_size, aug_dict, seed):
    rand_mirror = aug_dict.get("rand_mirror", False)
    rand_crop = aug_dict.get("rand_crop", False)
    random_erasing = aug_dict.get("random_erasing", False)
    resize_shorter = aug_dict.get("resize_shorter", None)
    force_resize = aug_dict.get("force_resize", None)

    train = GroupIterator(data_dir=os.path.join(data_dir, "bounding_box_train"), p_size=p_size, k_size=k_size,
                          num_worker=8, image_size=(crop_size * 2, crop_size), resize_shorter=resize_shorter,
                          force_resize=force_resize, rand_mirror=rand_mirror, rand_crop=rand_crop,
                          random_erasing=random_erasing, random_seed=seed)

    # val = GroupIterator(data_dir=os.path.join(data_dir, "bounding_box_test"), p_size=p_size, k_size=k_size,
    #                     image_size=(crop_size * 2, crop_size), resize_shorter=resize_shorter,
    #                     force_resize=force_resize,rand_crop=False, rand_mirror=False, random_seed=seed)

    return train, None


if __name__ == '__main__':
    random_seed = 0
    mx.random.seed(random_seed)

    # load configuration
    args = yaml.load(open("config.yml", "r"))
    selected_dataset = args["dataset"]
    datasets = ["duke", "market", "cuhk"]
    args["prefix"] = selected_dataset + "/" + args["prefix"]
    for dataset in datasets:
        dataset_config = args.pop(dataset)
        if dataset == selected_dataset:
            args.update(dataset_config)

    args = EasyDict(args)
    pprint(args)

    model_load_prefix = args.model_load_prefix
    model_load_epoch = args.model_load_epoch
    network = args.network
    gpus = args.gpus
    data_dir = args.data_dir
    p_size = args.p_size
    k_size = args.k_size
    lr_step = args.lr_step
    optmizer = args.optimizer
    lr = args.lr
    wd = args.wd
    num_epoch = args.num_epoch
    crop_size = args.crop_size
    prefix = args.prefix
    batch_size = p_size * k_size
    use_softmax = args.use_softmax
    # use_gcn = args.use_gcn
    bottleneck_dims = args.bottleneck_dims
    temperature = args.temperature
    num_id = args.num_id
    use_triplet = args.use_triplet
    triplet_margin = args.triplet_margin
    dropout_ratio = args.dropout_ratio
    softmax_weight_normalization = args.softmax_weight_normalization
    softmax_feat_normalization = args.softmax_feat_normalization
    triplet_normalization = args.triplet_normalization
    aug = args.aug
    # residual = args.residual
    memonger = args.memonger
    begin_epoch = args.begin_epoch
    # keep_diag = args.keep_diag
    norm_scale = args.norm_scale

    # config logger
    logging.basicConfig(format="%(asctime)s %(message)s",
                        filename='log/%s/%s.log' % (selected_dataset, os.path.basename(prefix)), filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.info(args)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)

    _, arg_params, aux_params = mx.model.load_checkpoint('%s' % model_load_prefix, model_load_epoch)

    devices = [mx.gpu(int(i)) for i in gpus.split(',')]

    train, val = get_iterators(data_dir=data_dir, p_size=p_size, k_size=k_size, crop_size=crop_size, aug_dict=aug,
                               seed=random_seed)

    steps = [int(x) for x in lr_step.split(',')]

    lr_scheduler = WarmupMultiFactorScheduler(step=[s * train.size for s in steps], factor=0.1, warmup=True,
                                              warmup_lr=1e-4, warmup_step=train.size * 20, mode="gradual")
    # lr_scheduler = ExponentialScheduler(base_lr=lr, exp=0.001, start_step=150 * train.size, end_step=300 * train.size)
    init = mx.initializer.MSRAPrelu(factor_type='out', slope=0.0)

    optimizer_params = {"learning_rate": lr,
                        "wd": wd,
                        "lr_scheduler": lr_scheduler,
                        "rescale_grad": 1.0 / batch_size,
                        "begin_num_update": begin_epoch * train.size}

    symbol = importlib.import_module('symbols.symbol_' + network).get_symbol()

    net = build_network(symbol=symbol, num_id=num_id, batch_size=batch_size, p_size=p_size,
                        softmax_weight_normalization=softmax_weight_normalization, norm_scale=norm_scale,
                        softmax_feat_normalization=softmax_feat_normalization,
                        triplet_normalization=triplet_normalization,
                        bottleneck_dims=bottleneck_dims, dropout_ratio=dropout_ratio,
                        use_softmax=use_softmax, use_triplet=use_triplet, triplet_margin=triplet_margin,
                        temperature=temperature)

    if memonger:
        net = search_plan(net, data=(batch_size, 3, crop_size * 2, crop_size), softmax_label=(batch_size,))

    # Metric
    metric_list = []
    if use_softmax:
        acc = mx.metric.Accuracy(output_names=["softmax_output"], label_names=["softmax_label"], name="acc")
        ce_loss = mx.metric.CrossEntropy(output_names=["softmax_output"], label_names=["softmax_label"], name="ce")
        metric_list.extend([acc, ce_loss])

    if use_triplet:
        triplet_loss = mx.metric.Loss(output_names=["triplet_output"], name="triplet")
        metric_list.append(triplet_loss)

    metric = mx.metric.CompositeEvalMetric(metrics=metric_list)

    model = mx.mod.Module(symbol=net, context=devices, logger=logger)
    model.fit(train_data=train,
              eval_data=None,
              eval_metric=metric,
              validation_metric=metric,
              arg_params=arg_params,
              aux_params=aux_params,
              allow_missing=True,
              initializer=init,
              optimizer=optmizer,
              optimizer_params=optimizer_params,
              num_epoch=num_epoch,
              begin_epoch=begin_epoch,
              batch_end_callback=mx.callback.Speedometer(batch_size=batch_size, frequent=5),
              epoch_end_callback=mx.callback.do_checkpoint("models/" + prefix, period=10),
              kvstore='device')

    clean_immediate_checkpoints("models", prefix, num_epoch)
