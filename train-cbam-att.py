# change to spatial and channel attention form, refer to train-mask.py, on 2018.09.05 by chencp
# main modify: 1, get symbol; 2, data initialization; 3, remove mask related codes
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

def build_network(symbol, num_id, p_size, gpus=1, **kwargs):
    triplet_normalization = kwargs.get("triplet_normalization", False)
    use_triplet = kwargs.get("use_triplet", False)
    use_softmax = kwargs.get("use_softmax", False)
    triplet_margin = kwargs.get("triplet_margin", 0.5)
    with_relu = kwargs.get("with_relu", True)

    label = mx.symbol.Variable(name="softmax_label")
    group = [label]

    in5b = symbol

    pooling = mx.symbol.Pooling(data=in5b, kernel=(1, 1), global_pool=True, pool_type='max', name='global_pool')
    flatten = mx.symbol.Flatten(data=pooling, name='flatten')
    # split to 3 streams
    k = args.p_size * args.k_size // gpus
    # flatten_full = mx.symbol.slice_axis(data=flatten, axis=0, begin=0, end=k)
    flatten_full = flatten

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

    return mx.symbol.Group(group)


def get_iterators(data_dir, p_size, k_size, crop_size, aug_dict, seed, data_type='rgbm', att_input_type='rgbm'):
    rand_mirror = aug_dict.get("rand_mirror", False)
    rand_crop = aug_dict.get("rand_crop", False)
    random_erasing = aug_dict.get("random_erasing", False)
    resize_shorter = aug_dict.get("resize_shorter", None)
    force_resize = aug_dict.get("force_resize", None)

    image_size = (crop_size * 2, crop_size)
    if att_input_type == 'rgbm':
        mask_label = False
    else:
        # take mask as input (hard mode) in channel attention branch
        mask_label = True

    train = GroupIteratorRGBM(data_dir=os.path.join(data_dir, "bounding_box_train"), p_size=p_size, k_size=k_size,
                              num_worker=8, image_size=image_size, resize_shorter=resize_shorter,
                              force_resize=force_resize, rand_mirror=rand_mirror, rand_crop=rand_crop,
                              random_erasing=random_erasing, random_seed=seed,
                              data_type=data_type, mask_label=mask_label, soft_or_hard=False, mask_shape=image_size)

    # val = GroupIterator(data_dir=os.path.join(data_dir, "bounding_box_test"), p_size=p_size, k_size=k_size,
    #                     image_size=(crop_size * 2, crop_size), resize_shorter=resize_shorter,
    #                     force_resize=force_resize,rand_crop=False, rand_mirror=False, random_seed=seed)

    return train, None


if __name__ == '__main__':
    random_seed = 0
    mx.random.seed(random_seed)

    # load configuration
    args = yaml.load(open("config-channel-att.yml", "r"))
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
    num_parts = args.num_parts

    mask_shape = (64, 32)
    data_type = args.data_type
    init_method = 'zero'
    share_branch = False
    att_input_type = args.att_input_type
    assert att_input_type == 'rgbm' or att_input_type == 'mask', 'att input_type only support rgbm and mask'
    mlp_channel_att = args.mlp_channel_att
    att_branch_share = args.att_branch_share
    channel_att_enable = args.channel_att_enable
    spatial_att_enalbe = args.spatial_att_enalbe

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
        if att_input_type == 'mask' or not att_branch_share and att_input_type == 'rgbm':
            if 'inception-bn' in model_load_prefix or 'resnet' in model_load_prefix:
                print('copy weights to channel attention branches, the weight copied as follow:')
                # copy the weight to channel attention branches
                temp_dict = arg_params.copy()
                for k, v in temp_dict.items():
                    if 'bn0_' in k or 'conv0_' in k \
                            or 'stage1' in k or 'bn_data' in k:
                        print('\t' + k, '{}'.format(v.shape))
                        arg_params.update({'att_branch_{}'.format(k): v})
                    # else:
                    #     print('\tno copy:' + k)
                temp_dict = aux_params.copy()
                for k, v in temp_dict.items():
                    if 'bn_data_' in k or 'bn0_' in k\
                            or 'stage1' in k:
                        print('\t' + k, '{}'.format(v.shape))
                        aux_params.update({'att_branch_{}'.format(k): v})
                    # else:
                    #     print('\tno copy:' + k)
                if att_input_type == 'mask':
                    print('re-initialize bn_data and conv0 layer for compatibility to mask data with one channel')
                    bn_data_name = ['bn_data_beta', 'bn_data_gamma']
                    for n in bn_data_name:
                        bn_data = arg_params[n].asnumpy()
                        new_bn_data = mx.nd.array([np.mean(bn_data)])
                        arg_params['att_branch_{}'.format(n)] = new_bn_data
                    bn_moving_name = ['bn_data_moving_mean', 'bn_data_moving_var']
                    for n in bn_moving_name:
                        bn_data = aux_params[n].asnumpy()
                        new_bn_data = mx.nd.array([np.mean(bn_data)])
                        aux_params['att_branch_{}'.format(n)] = new_bn_data
                    conv_name = ['conv0_weight']
                    for n in conv_name:
                        conv_w = arg_params[n].asnumpy()
                        conv_w = np.mean(conv_w, axis=1)
                        new_conv_w = mx.nd.array(conv_w[:, np.newaxis])
                        arg_params['att_branch_{}'.format(n)] = new_conv_w

                print('done!')
    else:
        # to resume from a saved model
        load_model = 'models/%s' % prefix
        _, arg_params, aux_params = mx.model.load_checkpoint(load_model, begin_epoch)
        print('load from saved model: {}'.format(load_model))

    devices = [mx.gpu(int(i)) for i in gpus.split(',')]

    train, val = get_iterators(data_dir=data_dir, p_size=p_size, k_size=k_size, crop_size=crop_size, aug_dict=aug,
                               seed=random_seed, data_type=data_type, att_input_type=att_input_type)

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

    symbol = importlib.import_module('symbols.symbol_' + network).get_symbol_cbam_att(att_input_type=att_input_type,
                                                                                      mlp_channel_att=mlp_channel_att,
                                                                                      att_branch_share=att_branch_share,
                                                                                      channel_att_enable=channel_att_enable,
                                                                                      spatial_att_enalbe=spatial_att_enalbe)

    net = build_network(symbol=symbol, num_id=num_id, batch_size=batch_size, p_size=p_size, with_relu=with_relu,
                        softmax_weight_normalization=softmax_weight_normalization, norm_scale=norm_scale,
                        softmax_feat_normalization=softmax_feat_normalization, residual=residual,
                        num_parts=num_parts, triplet_normalization=triplet_normalization,
                        keep_diag=keep_diag, bottleneck_dims=bottleneck_dims, dropout_ratio=dropout_ratio,
                        use_softmax=use_softmax, use_triplet=use_triplet, triplet_margin=triplet_margin,
                        temperature=temperature, gpus_num=len(devices))

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

    if use_triplet:
        triplet_loss = mx.metric.Loss(output_names=["triplet_output"], name="triplet")
        metric_list.append(triplet_loss)
        output_names_list.append("triplet_output")

    metric = mx.metric.CompositeEvalMetric(metrics=metric_list,
                                           output_names=output_names_list,
                                           label_names=label_list)

    label_names = ["softmax_label"]
    data_names = ["data"]
    if att_input_type == 'mask': data_names.append("binary_label")  # as data
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
