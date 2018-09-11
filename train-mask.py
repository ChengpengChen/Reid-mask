# change to mask form, on 2018.08.10, by chencp
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
from utils.group_rgbm_iterator import  GroupIteratorRGBM
from utils.initializer import InitWithArray
from utils.debug import forward_debug
from utils.memonger import search_plan
from utils.lr_scheduler import WarmupMultiFactorScheduler, ExponentialScheduler

# operators
import operators.triplet_loss
import operators.triplet_loss_inner
import operators.loss_layers_mask


def label_to_sim(label):
    label = mx.symbol.expand_dims(label, 1)
    label = mx.symbol.broadcast_axis(label, axis=1, size=batch_size)
    aff_label = mx.symbol.broadcast_equal(label, mx.symbol.transpose(label))
    return aff_label


def gcn_block(data, label=None, sim_normalize=True, **kwargs):
    temperature = kwargs.get("temperature", 1.0)
    keep_diag = kwargs.get("keep_diag", False)
    batch_size = kwargs.get("batch_size", -1)
    residual = kwargs.get("residual", False)

    in_sim = mx.symbol.L2Normalization(data=data, name='l2_norm') if sim_normalize else data
    sim = mx.symbol.dot(in_sim, in_sim, transpose_b=True)
    aff = mx.symbol.exp(sim / temperature)  # * label_to_sim(label)

    if not keep_diag:
        aff = (1 - mx.sym.eye(batch_size)) * aff

    aff = mx.symbol.broadcast_div(aff, mx.symbol.sum(aff, axis=1, keepdims=True))

    feat = mx.symbol.dot(aff, data)

    if residual:
        feat = 0.9 * feat + (1 - 0.9) * data
    return feat, sim


def pcb_classifier(data, label, num_id, num_hidden=256, postfix=""):
    fc = mx.symbol.FullyConnected(data, num_hidden=num_hidden, name="bottleneck%s" % postfix)
    bn = mx.sym.BatchNorm(data=fc, fix_gamma=False, momentum=0.9, eps=2e-5, name='bottleneck%s_bn' % postfix)
    relu = mx.sym.Activation(data=bn, act_type="relu", name='bottleneck%s_relu' % postfix)

    softmax_fc = mx.symbol.FullyConnected(relu, num_hidden=num_id, name="softmax%s_fc" % postfix)

    softmax = mx.symbol.SoftmaxOutput(data=softmax_fc, label=label, name='softmax%s' % postfix)

    return softmax


def build_network(symbol, num_id, p_size, soft_mask=True, gpus=1, **kwargs):
    triplet_normalization = kwargs.get("triplet_normalization", False)
    use_triplet = kwargs.get("use_triplet", False)
    use_softmax = kwargs.get("use_softmax", False)
    triplet_margin = kwargs.get("triplet_margin", 0.5)
    with_relu = kwargs.get("with_relu", True)
    num_parts = kwargs.get("num_parts", 1)
    use_pcb = kwargs.get("use_pcb", False)
    use_gcn = kwargs.get("use_gcn", False)

    use_inner_triplet = kwargs.get("use_inner_triplet", False)
    triplet_inner_weight = kwargs.get("triplet_inner_weight", 1.0)
    triplet_inner_margin = kwargs.get("triplet_inner_margin", 1.0)
    mask_weight = kwargs.get("mask_weight", 0.005)
    softmax_extra_grad_scale = kwargs.get("softmax_extra_grad_scale", 1.0)
    three_streams = kwargs.get("three_streams", True)

    label = mx.symbol.Variable(name="softmax_label")
    group = [label]

    if soft_mask:
        in5b, delta_sigmoid = symbol
        # only get mask branch in the three streams setting
        if three_streams:
            # mask loss
            mask_gt = mx.symbol.Variable(name="binary_label")
            mask_seg = mx.symbol.Custom(data=delta_sigmoid, binary_label=mask_gt,
                                        grad_scale=mask_weight,
                                        op_type='MaskBinaryLoss', name='mask_loss')
            group.append(mask_gt)
            group.append(mask_seg)
    else:
        in5b = symbol

    pooling = mx.symbol.Pooling(data=in5b, kernel=(1, 1), global_pool=True, pool_type='max', name='global_pool')
    flatten = mx.symbol.Flatten(data=pooling, name='flatten')
    # split to 3 streams
    k = args.p_size * args.k_size // gpus
    flatten_full = mx.symbol.slice_axis(data=flatten, axis=0, begin=0, end=k)
    flatten_body = mx.symbol.slice_axis(data=flatten, axis=0, begin=k, end=2 * k)
    flatten_bg = mx.symbol.slice_axis(data=flatten, axis=0, begin=2 * k, end=3 * k)

    # inner triplet loss in the three streams setting
    if use_inner_triplet and three_streams:
        data_triplet_inner = mx.sym.L2Normalization(flatten, name="triplet_inner_l2") if triplet_normalization else flatten
        triplet_inner = mx.symbol.Custom(data=data_triplet_inner,
                                         grad_scale=triplet_inner_weight, margin=triplet_inner_margin,
                                         op_type='TripletLossInner', name='triplet_inner')
        group.append(triplet_inner)

    if use_gcn:
        flatten_full, sim = gcn_block(flatten_full, label=label, sim_normalize=True, **kwargs)

    # triplet loss
    if use_triplet:
        data_triplet = mx.sym.L2Normalization(flatten_full, name="triplet_l2") if triplet_normalization else flatten_full
        triplet = mx.symbol.Custom(data=data_triplet, p_size=p_size, margin=triplet_margin, op_type='TripletLoss',
                                   name='triplet')
        group.append(triplet)

    # softmax cross entropy loss with 3 streams
    if use_softmax:
        def softmax_branch(fea, name_prefix='', grad_scale=1.0):
            fc = mx.symbol.FullyConnected(fea, num_hidden=bottleneck_dims, name="{}bottleneck".format(name_prefix))
            bn = mx.sym.BatchNorm(data=fc, fix_gamma=False, momentum=0.9, eps=2e-5, name='{}bottleneck_bn'.format(name_prefix))
            if not with_relu:
                bn = mx.sym.Activation(data=bn, act_type='relu', name='{}bottleneck_relu'.format(name_prefix))
            dropout = mx.symbol.Dropout(bn, p=dropout_ratio)

            softmax_w = mx.symbol.Variable("{}softmax_weight".format(name_prefix), shape=(num_id, bottleneck_dims))
            if softmax_weight_normalization:
                softmax_w = mx.symbol.L2Normalization(softmax_w, name="{}softmax_weight_norm".format(name_prefix))
            if softmax_feat_normalization:
                data_softmax = mx.sym.L2Normalization(dropout, name="{}softmax_data_norm".format(name_prefix)) * norm_scale
            else:
                data_softmax = dropout

            softmax_fc = mx.symbol.FullyConnected(data_softmax, weight=softmax_w, num_hidden=num_id,
                                                  no_bias=True if softmax_weight_normalization else False,
                                                  name="{}softmax_fc".format(name_prefix))
            return mx.symbol.SoftmaxOutput(data=softmax_fc, label=label, name='{}softmax'.format(name_prefix),
                                           grad_scale=grad_scale)

        softmax_full = softmax_branch(flatten_full)
        group.append(softmax_full)
        # only id loss for body and bg in the three streams setting
        if three_streams:
            softmax_body = softmax_branch(flatten_body, name_prefix='body_', grad_scale=softmax_extra_grad_scale)
            softmax_bg = softmax_branch(flatten_bg, name_prefix='bg_', grad_scale=softmax_extra_grad_scale)
            group.append(softmax_body)
            group.append(softmax_bg)

        # sim_label = label_to_sim(label)
        # sim_reg = mx.symbol.LogisticRegressionOutput(data=sim, label=sim_label, name="sim_reg")
        # group.append(sim_reg)

    # PCB module
    if use_pcb:
        part_pooling = mx.symbol.contrib.AdaptiveAvgPooling2D(data=symbol, output_size=(num_parts, 1), name="part_pool")
        parts = mx.symbol.split(part_pooling, axis=2, num_outputs=num_parts)
        for i in range(num_parts):
            data = mx.symbol.Flatten(parts[i])
            # data = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=0.9, eps=2e-5)
            part_softmax = pcb_classifier(data, label, num_id, postfix=str(i))

            group.append(part_softmax)

    return mx.symbol.Group(group)


def get_iterators(data_dir, p_size, k_size, crop_size, aug_dict, seed,
                  data_type='rgbm', mask_label=True, soft_or_hard=True, mask_shape=(64, 32)):
    rand_mirror = aug_dict.get("rand_mirror", False)
    rand_crop = aug_dict.get("rand_crop", False)
    random_erasing = aug_dict.get("random_erasing", False)
    resize_shorter = aug_dict.get("resize_shorter", None)
    force_resize = aug_dict.get("force_resize", None)

    train = GroupIteratorRGBM(data_dir=os.path.join(data_dir, "bounding_box_train"), p_size=p_size, k_size=k_size,
                              num_worker=8, image_size=(crop_size * 2, crop_size), resize_shorter=resize_shorter,
                              force_resize=force_resize, rand_mirror=rand_mirror, rand_crop=rand_crop,
                              random_erasing=random_erasing, random_seed=seed,
                              data_type=data_type, mask_label=mask_label, soft_or_hard=soft_or_hard, mask_shape=mask_shape)

    # val = GroupIterator(data_dir=os.path.join(data_dir, "bounding_box_test"), p_size=p_size, k_size=k_size,
    #                     image_size=(crop_size * 2, crop_size), resize_shorter=resize_shorter,
    #                     force_resize=force_resize,rand_crop=False, rand_mirror=False, random_seed=seed)

    return train, None


if __name__ == '__main__':
    random_seed = 0
    mx.random.seed(random_seed)

    # load configuration
    args = yaml.load(open("config-mask.yml", "r"))
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
    use_gcn = args.use_gcn
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
    residual = args.residual
    memonger = args.memonger
    begin_epoch = args.begin_epoch
    keep_diag = args.keep_diag
    norm_scale = args.norm_scale
    with_relu = args.with_relu
    use_pcb = args.use_pcb
    num_parts = args.num_parts

    use_inner_triplet = args.use_inner_triplet
    triplet_inner_margin = args.triplet_inner_margin
    triplet_inner_weight = args.triplet_inner_weight
    mask_weight = args.mask_weight
    soft_mask = args.soft_mask
    mask_shape = (64, 32)
    data_type = args.data_type
    init_method = 'zero'
    share_branch = False
    softmax_extra_grad_scale = args.softmax_extra_grad_scale
    three_streams = args.three_streams

    # config logger
    logging.basicConfig(format="%(asctime)s %(message)s",
                        filename='log/%s/%s.log' % (selected_dataset, os.path.basename(prefix)), filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.info(args)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)

    # _, arg_params, aux_params = mx.model.load_checkpoint('%s' % model_load_prefix, model_load_epoch)
    if begin_epoch == 0:
        # from pretrained model
        if data_type == 'rgbm':
            model_load_prefix += '-{}-init'.format(init_method)
        load_model = 'pretrain_models/%s' % model_load_prefix
        _, arg_params, aux_params = mx.model.load_checkpoint(load_model, model_load_epoch)
        print('load from pre-trained model: {}'.format(load_model))
        if three_streams and ('inception-bn' in model_load_prefix or 'resnet' in model_load_prefix):
            if not share_branch:
                print('copy weights to the other 2 shared branches, the weight not copied as follow:')
                # copy the weight to the no-shared branches
                temp_dict = arg_params.copy()
                for k, v in temp_dict.items():
                    if 'bn_1_' in k or 'bn_2_' in k or 'conv_1_' in k or 'conv_2_' in k or 'fc1_' in k\
                            or 'stage1' in k:
                        print('\t' + k)
                        continue
                    arg_params.update({'body_{}'.format(k): v})
                    arg_params.update({'bg_{}'.format(k): v})
                temp_dict = aux_params.copy()
                for k, v in temp_dict.items():
                    if 'bn_1_' in k or 'bn_2_' in k or 'fc1_' in k\
                            or 'stage1' in k:
                        print('\t' + k)
                        continue
                    aux_params.update({'body_{}'.format(k): v})
                    aux_params.update({'bg_{}'.format(k): v})
                print('done!')
    else:
        # to resume from a saved model
        load_model = 'models/%s' % prefix
        _, arg_params, aux_params = mx.model.load_checkpoint(load_model, begin_epoch)
        print('load from saved model: {}'.format(load_model))

    devices = [mx.gpu(int(i)) for i in gpus.split(',')]

    train, val = get_iterators(data_dir=data_dir, p_size=p_size, k_size=k_size, crop_size=crop_size, aug_dict=aug,
                               seed=random_seed, data_type=data_type, mask_label=three_streams,
                               soft_or_hard=soft_mask, mask_shape=mask_shape)

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

    symbol = importlib.import_module('symbols.symbol_' + network).get_symbol_delta(soft_mask=soft_mask)

    net = build_network(symbol=symbol, num_id=num_id, batch_size=batch_size, p_size=p_size, with_relu=with_relu,
                        softmax_weight_normalization=softmax_weight_normalization, norm_scale=norm_scale,
                        softmax_feat_normalization=softmax_feat_normalization, residual=residual, use_pcb=use_pcb,
                        num_parts=num_parts, triplet_normalization=triplet_normalization, use_gcn=use_gcn,
                        keep_diag=keep_diag, bottleneck_dims=bottleneck_dims, dropout_ratio=dropout_ratio,
                        use_softmax=use_softmax, use_triplet=use_triplet, triplet_margin=triplet_margin,
                        temperature=temperature, gpus_num=len(devices), soft_mask=soft_mask,
                        mask_weight=mask_weight, use_inner_triplet=use_inner_triplet,
                        triplet_inner_margin=triplet_inner_margin, triplet_inner_weight=triplet_inner_weight,
                        softmax_extra_grad_scale=softmax_extra_grad_scale, three_streams=three_streams)

    if memonger:
        net = search_plan(net, data=(batch_size, 3, crop_size * 2, crop_size), softmax_label=(batch_size,))

    # Metric
    metric_list = []
    label_list = []
    output_names_list = []
    if use_softmax:
        acc = mx.metric.Accuracy(output_names=["softmax_output"], label_names=["softmax_label"], name="acc")
        ce_loss = mx.metric.CrossEntropy(output_names=["softmax_output"], label_names=["softmax_label"], name="ce")
        metric_list.extend([acc, ce_loss])
        output_names_list = ["softmax_output"]
        label_list.append("softmax_label")
        if three_streams:
            acc_body = mx.metric.Accuracy(output_names=["body_softmax_output"], label_names=["softmax_label"],
                                          name="acc_body")
            acc_bg = mx.metric.Accuracy(output_names=["bg_softmax_output"], label_names=["softmax_label"], name="acc_bg")
            metric_list.extend([acc_body, acc_bg])
            output_names_list.extend(["body_softmax_output", "bg_softmax_output"])

    if use_triplet:
        triplet_loss = mx.metric.Loss(output_names=["triplet_output"], name="triplet")
        metric_list.append(triplet_loss)
        output_names_list.append("triplet_output")

    if use_inner_triplet and three_streams:
        triplet_inner_loss = mx.metric.Loss(output_names=["triplet_inner_output"], name="triplet_inner")
        metric_list.append(triplet_inner_loss)
        output_names_list.append("triplet_inner_output")

    if soft_mask and three_streams:
        mask_loss = mx.metric.Loss(output_names=["mask_loss_output"], name='mask_loss_output')
        metric_list.append(mask_loss)
        output_names_list.append("mask_loss_output")
        label_list.append("binary_label")

    if use_pcb:
        for i in range(num_parts):
            output_names = ["softmax%d_output" % i]
            label_names = ["softmax_label"]
            metric_list.append(mx.metric.Accuracy(output_names=output_names, label_names=label_names, name="acc%d" % i))
            metric_list.append(mx.metric.CrossEntropy(output_names=output_names, label_names=label_names,
                                                      name="ce%d" % i))

    metric = mx.metric.CompositeEvalMetric(metrics=metric_list,
                                           output_names=output_names_list,
                                           label_names=label_list)

    label_names = ["softmax_label"]
    if soft_mask and three_streams: label_names.append("binary_label")  # as label
    data_names = ["data"]
    if not soft_mask and three_streams: data_names.append("binary_label")  # as data
    model = mx.mod.Module(symbol=net, context=devices, logger=logger,
                          data_names=data_names, label_names=label_names)
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
