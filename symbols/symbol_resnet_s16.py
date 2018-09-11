'''
Reproducing paper:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''
"""
add mask for resnet and cbam (spatial and channel) attention support, by chencp
"""
import mxnet as mx


def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=512,
                  memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
                                   pad=(0, 0), no_bias=1, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride,
                                   pad=(1, 1), no_bias=1, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=1, workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=1,
                                          workspace=workspace, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                                   no_bias=1, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                   no_bias=1, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=1,
                                          workspace=workspace, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut


def resnet(units, num_stage, filter_list, data_type, bottle_neck=True, bn_mom=0.9, workspace=512, memonger=False):
    """Return ResNet symbol of cifar10 and imagenet
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stage : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_class : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert (num_unit == num_stage)
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if data_type == 'cifar10':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                  no_bias=1, name="conv0", workspace=workspace)
    elif data_type == 'imagenet':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                  no_bias=1, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    else:
        raise ValueError("do not support {} yet".format(data_type))
    for i in range(num_stage - 1):
        body = residual_unit(body, filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i] - 1):
            body = residual_unit(body, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)

    # change the  stride of the last stage to 1.
    body = residual_unit(body, filter_list[-1], (1, 1), False, name='stage%d_unit%d' % (num_stage, 1),
                         bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    for j in range(units[-1] - 1):
        body = residual_unit(body, filter_list[num_stage], (1, 1), True, name='stage%d_unit%d' % (num_stage, j + 2),
                             bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)

    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')

    return relu1


def get_symbol(depth=50):
    if depth == 18:
        units = [2, 2, 2, 2]
    elif depth == 34:
        units = [3, 4, 6, 3]
    elif depth == 50:
        units = [3, 4, 6, 3]
    elif depth == 101:
        units = [3, 4, 23, 3]
    elif depth == 152:
        units = [3, 8, 36, 3]
    elif depth == 200:
        units = [3, 24, 36, 3]
    elif depth == 269:
        units = [3, 30, 48, 8]
    else:
        raise ValueError("no experiments done on detph {}, you can do it youself".format(depth))

    symbol = resnet(units=units, num_stage=4,
                    filter_list=[64, 256, 512, 1024, 2048] if depth >= 50 else [64, 64, 128, 256, 512],
                    data_type="imagenet", bottle_neck=True if depth >= 50 else False, workspace=1024, memonger=True)
    return symbol



def resnet_soft_mask_no_share(units, num_stage, filter_list, data_type, bottle_neck=True, bn_mom=0.9, workspace=1024,
                              memonger=False):
    """
    add soft mask after block 2 (stride 4), 2018.08.09

    Return ResNet symbol of cifar10 and imagenet
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stage : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_class : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert (num_unit == num_stage)
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if data_type == 'cifar10':
        net = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                  no_bias=1, name="conv0", workspace=workspace)
    elif data_type == 'imagenet':
        net = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                  no_bias=1, name="conv0", workspace=workspace)
        net = mx.sym.BatchNorm(data=net, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        net = mx.sym.Activation(data=net, act_type='relu', name='relu0')
        net = mx.symbol.Pooling(data=net, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    else:
        raise ValueError("do not support {} yet".format(data_type))
    i = 0
    net = residual_unit(net, filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False,
                         name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                         memonger=memonger)
    for j in range(units[i] - 1):
        net = residual_unit(net, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                             bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    net_full = net

    # delta
    delta_fc = mx.symbol.Convolution(data=net, num_filter=1, kernel=(3, 3), stride=(1, 1),
                                     pad=(1, 1), name='delta_fc')
    delta_sigmoid = mx.symbol.sigmoid(data=delta_fc, name='delta_sigmoid')
    delta_minus = mx.symbol.broadcast_minus(lhs=mx.symbol.ones((1, 1)), rhs=delta_sigmoid,
                                            name='delta_minus')
    net_body = mx.symbol.broadcast_mul(lhs=net, rhs=delta_sigmoid, name='delta_mul_body')
    net_bg = mx.symbol.broadcast_mul(lhs=net, rhs=delta_minus, name='delta_mul_bg')

    # main stream
    for i in range(1, num_stage):
        stride = (1 if i == 0 or i == num_stage-1 else 2, 1 if i == 0 or i == num_stage-1 else 2)
        net_full = residual_unit(net_full, filter_list[i + 1], stride, False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i] - 1):
            net_full = residual_unit(net_full, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    bn1 = mx.sym.BatchNorm(data=net_full, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')

    # body stream
    for i in range(1, num_stage):
        stride = (1 if i == 0 or i == num_stage-1 else 2, 1 if i == 0 or i == num_stage-1 else 2)
        net_body = residual_unit(net_body, filter_list[i + 1], stride, False,
                             name='body_stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i] - 1):
            net_body = residual_unit(net_body, filter_list[i + 1], (1, 1), True, name='body_stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    bn1_body = mx.sym.BatchNorm(data=net_body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='body_bn1')
    relu1_bogy = mx.sym.Activation(data=bn1_body, act_type='relu', name='body_relu1')

    # bg stream
    for i in range(1, num_stage):
        stride = (1 if i == 0 or i == num_stage-1 else 2, 1 if i == 0 or i == num_stage-1 else 2)
        net_bg = residual_unit(net_bg, filter_list[i + 1], stride, False,
                             name='bg_stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i] - 1):
            net_bg = residual_unit(net_bg, filter_list[i + 1], (1, 1), True, name='bg_stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    bn1_bg = mx.sym.BatchNorm(data=net_bg, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bg_bn1')
    relu1_bg = mx.sym.Activation(data=bn1_bg, act_type='relu', name='bg_relu1')

    relu_all = mx.symbol.concat(relu1, relu1_bogy, relu1_bg, dim=0, name='fea_concat_all')

    return relu_all, delta_sigmoid



def resnet_hard_mask_no_share(units, num_stage, filter_list, data_type, bottle_neck=True, bn_mom=0.9, workspace=1024,
                              memonger=False):
    """
    add hard mask after block 2 (stride 4), 2018.08.09

    Return ResNet symbol of cifar10 and imagenet
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stage : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_class : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert (num_unit == num_stage)
    data = mx.sym.Variable(name='data')
    mask = mx.symbol.Variable(name="binary_label")
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if data_type == 'cifar10':
        net = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                  no_bias=1, name="conv0", workspace=workspace)
    elif data_type == 'imagenet':
        net = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                  no_bias=1, name="conv0", workspace=workspace)
        net = mx.sym.BatchNorm(data=net, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        net = mx.sym.Activation(data=net, act_type='relu', name='relu0')
        net = mx.symbol.Pooling(data=net, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    else:
        raise ValueError("do not support {} yet".format(data_type))
    i = 0
    net = residual_unit(net, filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False,
                        name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                        memonger=memonger)
    for j in range(units[i] - 1):
        net = residual_unit(net, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                            bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    net_full = net
    # delta
    delta_minus = mx.symbol.broadcast_minus(lhs=mx.symbol.ones((1, 1)), rhs=mask,
                                            name='delta_minus')
    net_body = mx.symbol.broadcast_mul(lhs=net, rhs=mask, name='delta_mul_body')
    net_bg = mx.symbol.broadcast_mul(lhs=net, rhs=delta_minus, name='delta_mul_bg')

    # main stream
    for i in range(1, num_stage):
        stride = (1 if i == 0 or i == num_stage-1 else 2, 1 if i == 0 or i == num_stage-1 else 2)
        net_full = residual_unit(net_full, filter_list[i + 1], stride, False,
                          name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                          memonger=memonger)
        for j in range(units[i] - 1):
            net_full = residual_unit(net_full, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                              bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    bn1 = mx.sym.BatchNorm(data=net_full, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')

    # body stream
    for i in range(1, num_stage):
        stride = (1 if i == 0 or i == num_stage-1 else 2, 1 if i == 0 or i == num_stage-1 else 2)
        net_body = residual_unit(net_body, filter_list[i + 1], stride, False,
                               name='body_stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                               memonger=memonger)
        for j in range(units[i] - 1):
            net_body = residual_unit(net_body, filter_list[i + 1], (1, 1), True,
                                   name='body_stage%d_unit%d' % (i + 1, j + 2),
                                   bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    bn1_body = mx.sym.BatchNorm(data=net_body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='body_bn1')
    relu1_bogy = mx.sym.Activation(data=bn1_body, act_type='relu', name='body_relu1')

    # bg stream
    for i in range(1, num_stage):
        stride = (1 if i == 0 or i == num_stage-1 else 2, 1 if i == 0 or i == num_stage-1 else 2)
        net_bg = residual_unit(net_bg, filter_list[i + 1], stride, False,
                             name='bg_stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i] - 1):
            net_bg = residual_unit(net_bg, filter_list[i + 1], (1, 1), True, name='bg_stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    bn1_bg = mx.sym.BatchNorm(data=net_bg, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bg_bn1')
    relu1_bg = mx.sym.Activation(data=bn1_bg, act_type='relu', name='bg_relu1')

    relu_all = mx.symbol.concat(relu1, relu1_bogy, relu1_bg, dim=0, name='fea_concat_all')

    return relu_all


def get_symbol_delta(depth=50, soft_mask=True, share=False, bn_global_stats=False, stop_grad_delta=False,large_mask=True):
    assert not share, 'share mode not support yet'
    if depth == 18:
        units = [2, 2, 2, 2]
    elif depth == 34:
        units = [3, 4, 6, 3]
    elif depth == 50:
        units = [3, 4, 6, 3]
    elif depth == 101:
        units = [3, 4, 23, 3]
    elif depth == 152:
        units = [3, 8, 36, 3]
    elif depth == 200:
        units = [3, 24, 36, 3]
    elif depth == 269:
        units = [3, 30, 48, 8]
    else:
        raise ValueError("no experiments done on detph {}, you can do it youself".format(depth))
    get_symbol_func = resnet_soft_mask_no_share if soft_mask else resnet_hard_mask_no_share

    return get_symbol_func(units=units, num_stage=4,
                           filter_list=[64, 256, 512, 1024, 2048] if depth >= 50 else [64, 64, 128, 256, 512],
                           data_type="imagenet", bottle_neck=True if depth >= 50 else False, memonger=True)

def mask_channel_att_branch(bn_mom, filter_list, units, workspace, bottle_neck, memonger):
    """the channel attention branch with mask input
        take conv1 and stage1 from resnet
    """
    mask_label = mx.sym.Variable(name='binary_label')
    att_branch_data = mx.sym.BatchNorm(data=mask_label, fix_gamma=True, eps=2e-5, momentum=bn_mom,
                                        name='att_branch_bn_data')

    att_branch_body = mx.sym.Convolution(data=att_branch_data, num_filter=filter_list[0], kernel=(7, 7),
                                          stride=(2, 2), pad=(3, 3),
                              no_bias=1, name="att_branch_conv0", workspace=workspace)
    att_branch_body = mx.sym.BatchNorm(data=att_branch_body, fix_gamma=False, eps=2e-5, momentum=bn_mom,
                                        name='att_branch_bn0')
    att_branch_body = mx.sym.Activation(data=att_branch_body, act_type='relu', name='att_branch_relu0')
    att_branch_body = mx.symbol.Pooling(data=att_branch_body, kernel=(3, 3), stride=(2, 2), pad=(1, 1),
                                         pool_type='max')

    i = 0
    att_branch_body = residual_unit(att_branch_body, filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2),
                                     False, name='att_branch_stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck,
                                     workspace=workspace, memonger=memonger)
    for j in range(units[i] - 1):
        att_branch_body = residual_unit(att_branch_body, filter_list[i + 1], (1, 1), True,
                                         name='att_branch_stage%d_unit%d' % (i + 1, j + 2),
                                         bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    return att_branch_body

def resnet_channel_att(units, num_stage, filter_list, data_type, bottle_neck=True, bn_mom=0.9, workspace=1024,
                       memonger=False, mlp_channel_att=True, att_branch_share=False, att_input_type='rgbm'):
    """Return ResNet symbol of cifar10 and imagenet, with a channel attention branch (rgbm input)
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stage : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_class : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert (num_unit == num_stage)
    data_ori = mx.sym.Variable(name='data')
    # main branch
    data = mx.sym.BatchNorm(data=data_ori, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if data_type == 'cifar10':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                  no_bias=1, name="conv0", workspace=workspace)
    elif data_type == 'imagenet':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                  no_bias=1, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    else:
        raise ValueError("do not support {} yet".format(data_type))
    i = 0
    body = residual_unit(body, filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False,
                         name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                         memonger=memonger)
    for j in range(units[i] - 1):
        body = residual_unit(body, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                             bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    # channel attention branch
    if att_input_type == 'rgbm':
        if not att_branch_share:
            channel_att_data = mx.sym.BatchNorm(data=data_ori, fix_gamma=True, eps=2e-5, momentum=bn_mom,
                                                name='channel_att_bn_data')
            if data_type == 'cifar10':
                channel_att_body = mx.sym.Convolution(data=channel_att_data, num_filter=filter_list[0], kernel=(3, 3),
                                                      stride=(1, 1), pad=(1, 1),
                                          no_bias=1, name="channel_att_conv0", workspace=workspace)
            elif data_type == 'imagenet':
                channel_att_body = mx.sym.Convolution(data=channel_att_data, num_filter=filter_list[0], kernel=(7, 7),
                                                      stride=(2, 2), pad=(3, 3),
                                          no_bias=1, name="channel_att_conv0", workspace=workspace)
                channel_att_body = mx.sym.BatchNorm(data=channel_att_body, fix_gamma=False, eps=2e-5, momentum=bn_mom,
                                                    name='channel_att_bn0')
                channel_att_body = mx.sym.Activation(data=channel_att_body, act_type='relu', name='channel_att_relu0')
                channel_att_body = mx.symbol.Pooling(data=channel_att_body, kernel=(3, 3), stride=(2, 2), pad=(1, 1),
                                                     pool_type='max')
            else:
                raise ValueError("do not support {} yet".format(data_type))
            i = 0
            channel_att_body = residual_unit(channel_att_body, filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2),
                                             False, name='channel_att_stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck,
                                             workspace=workspace, memonger=memonger)
            for j in range(units[i] - 1):
                channel_att_body = residual_unit(channel_att_body, filter_list[i + 1], (1, 1), True,
                                                 name='channel_att_stage%d_unit%d' % (i + 1, j + 2),
                                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
        else:
            # share wights with the main branch, same as se
            channel_att_body = body
    else:
        channel_att_body = mask_channel_att_branch(bn_mom=bn_mom, filter_list=filter_list, units=units,
                                                   workspace=workspace, bottle_neck=bottle_neck, memonger=memonger)

    if mlp_channel_att:
        print('use mlp in the channel attention branch')
        channel_att_body_avg = mx.sym.Pooling(channel_att_body, name='channel_att_globle_avg', kernel=(1, 1),
                                              pool_type='avg', global_pool=True)
        channel_att_body_max = mx.sym.Pooling(channel_att_body, name='channel_att_globle_max', kernel=(1, 1),
                                              pool_type='max', global_pool=True)
        channel_att_body = mx.symbol.concat(channel_att_body_avg, channel_att_body_max, dim=0,
                                            name='global_pooling_concat')
        channel_att_body = mx.symbol.FullyConnected(channel_att_body, name='channel_att_mlp_1', flatten=True,
                                                    num_hidden=256)
        channel_att_body = mx.symbol.Activation(channel_att_body, act_type='relu')
        channel_att_body = mx.symbol.FullyConnected(channel_att_body, name='channel_att_mlp_2', num_hidden=256)

        channel_att_body_avg, channel_att_body_max = \
            mx.symbol.split(data=channel_att_body, axis=0, num_outputs=2)
        channel_att_body = channel_att_body_avg + channel_att_body_max
    else:
        channel_att_body = mx.sym.Pooling(channel_att_body, name='channel_att_globle_avg', kernel=(1, 1),
                                          pool_type='avg', global_pool=True)

    channel_att_body = mx.symbol.sigmoid(channel_att_body, name='channel_att_sigmoid')
    channel_att_body = mx.symbol.Reshape(channel_att_body, shape=(0, 0, 1, 1), name='channel_att_reshape')
    body = mx.sym.broadcast_mul(lhs=body, rhs=channel_att_body, name='channel_att_multiply')

    # main branch
    for i in range(1, num_stage-1):
        body = residual_unit(body, filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i] - 1):
            body = residual_unit(body, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)

    # change the  stride of the last stage to 1.
    body = residual_unit(body, filter_list[-1], (1, 1), False, name='stage%d_unit%d' % (num_stage, 1),
                         bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    for j in range(units[-1] - 1):
        body = residual_unit(body, filter_list[num_stage], (1, 1), True, name='stage%d_unit%d' % (num_stage, j + 2),
                             bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)

    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')

    return relu1


def resnet_cbam_att(units, num_stage, filter_list, data_type, bottle_neck=True, bn_mom=0.9, workspace=1024,
                    memonger=False, channel_att_enable=False, spatial_att_enalbe=False,
                    mlp_channel_att=True, att_branch_share=False, att_input_type='rgbm'):
    """Return ResNet symbol of cifar10 and imagenet, with a channel attention branch (rgbm input)
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stage : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_class : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    assert channel_att_enable or spatial_att_enalbe, 'channel or spatial attention enable error'
    num_unit = len(units)
    assert (num_unit == num_stage)
    data_ori = mx.sym.Variable(name='data')
    # main branch
    data = mx.sym.BatchNorm(data=data_ori, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if data_type == 'cifar10':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                  no_bias=1, name="conv0", workspace=workspace)
    elif data_type == 'imagenet':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                  no_bias=1, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    else:
        raise ValueError("do not support {} yet".format(data_type))
    i = 0
    body = residual_unit(body, filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False,
                         name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                         memonger=memonger)
    for j in range(units[i] - 1):
        body = residual_unit(body, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                             bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    # attention branch
    if att_input_type == 'rgbm':
        print('rgbm input for attention branch')
        if not att_branch_share:
            print('attention branch not share weights with base network')
            att_branch_data = mx.sym.BatchNorm(data=data_ori, fix_gamma=True, eps=2e-5, momentum=bn_mom,
                                                name='att_branch_bn_data')
            if data_type == 'cifar10':
                att_branch_body = mx.sym.Convolution(data=att_branch_data, num_filter=filter_list[0], kernel=(3, 3),
                                                      stride=(1, 1), pad=(1, 1),
                                          no_bias=1, name="att_branch_conv0", workspace=workspace)
            elif data_type == 'imagenet':
                att_branch_body = mx.sym.Convolution(data=att_branch_data, num_filter=filter_list[0], kernel=(7, 7),
                                                      stride=(2, 2), pad=(3, 3),
                                          no_bias=1, name="att_branch_conv0", workspace=workspace)
                att_branch_body = mx.sym.BatchNorm(data=att_branch_body, fix_gamma=False, eps=2e-5, momentum=bn_mom,
                                                    name='att_branch_bn0')
                att_branch_body = mx.sym.Activation(data=att_branch_body, act_type='relu', name='att_branch_relu0')
                att_branch_body = mx.symbol.Pooling(data=att_branch_body, kernel=(3, 3), stride=(2, 2), pad=(1, 1),
                                                     pool_type='max')
            else:
                raise ValueError("do not support {} yet".format(data_type))
            i = 0
            att_branch_body = residual_unit(att_branch_body, filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2),
                                             False, name='att_branch_stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck,
                                             workspace=workspace, memonger=memonger)
            for j in range(units[i] - 1):
                att_branch_body = residual_unit(att_branch_body, filter_list[i + 1], (1, 1), True,
                                                 name='att_branch_stage%d_unit%d' % (i + 1, j + 2),
                                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
        else:
            # share wights with the main branch, same as se
            print('attention branch share weights with base network')
            att_branch_body = body
    else:
        print('mask input for attention branch, and not share weights with base network')
        att_branch_body = mask_channel_att_branch(bn_mom=bn_mom, filter_list=filter_list, units=units,
                                                  workspace=workspace, bottle_neck=bottle_neck, memonger=memonger)

    # channel attention
    if channel_att_enable:
        print('channel attention enable')
        if mlp_channel_att:
            print('use mlp in the channel attention branch')
            channel_att_body_avg = mx.sym.Pooling(att_branch_body, name='channel_att_globle_avg', kernel=(1, 1),
                                                  pool_type='avg', global_pool=True)
            channel_att_body_max = mx.sym.Pooling(att_branch_body, name='channel_att_globle_max', kernel=(1, 1),
                                                  pool_type='max', global_pool=True)
            channel_att_body = mx.symbol.concat(channel_att_body_avg, channel_att_body_max, dim=0,
                                                name='global_pooling_concat')
            channel_att_body = mx.symbol.FullyConnected(channel_att_body, name='channel_att_mlp_1', flatten=True,
                                                        num_hidden=256)
            channel_att_body = mx.symbol.Activation(channel_att_body, act_type='relu')
            channel_att_body = mx.symbol.FullyConnected(channel_att_body, name='channel_att_mlp_2', num_hidden=256)

            channel_att_body_avg, channel_att_body_max = \
                mx.symbol.split(data=channel_att_body, axis=0, num_outputs=2)
            channel_att_body = channel_att_body_avg + channel_att_body_max
        else:
            channel_att_body = mx.sym.Pooling(att_branch_body, name='channel_att_globle_avg', kernel=(1, 1),
                                              pool_type='avg', global_pool=True)

        channel_att_body = mx.symbol.sigmoid(channel_att_body, name='channel_att_sigmoid')
        channel_att_body = mx.symbol.Reshape(channel_att_body, shape=(0, 0, 1, 1), name='channel_att_reshape')
        body = mx.sym.broadcast_mul(lhs=body, rhs=channel_att_body, name='channel_att_multiply')
        att_branch_body = body

    # spatial attention
    if spatial_att_enalbe:
        print('spatial attention enable')
        spatial_att_body_avg = mx.symbol.mean(att_branch_body, axis=1, keepdims=True, name='spatial_att_global_avg')
        spatial_att_body_max = mx.symbol.max(att_branch_body, axis=1, keepdims=True, name='spatial_att_global_max')
        spatial_att_body = mx.symbol.concat(spatial_att_body_avg, spatial_att_body_max, dim=1,
                                            name='spatial_global_pooling_concat')
        spatial_att_body = mx.symbol.Convolution(spatial_att_body, kernel=(7, 7), num_filter=1, stride=(1, 1),
                                                 pad=(3, 3), name='spatial_att_conv7x7')
        spatial_att_body = mx.symbol.sigmoid(spatial_att_body, name='spatial_att_sigmoid')
        body = mx.symbol.broadcast_mul(lhs=body, rhs=spatial_att_body, name='spatial_att_multiply')

    # main branch
    for i in range(1, num_stage-1):
        body = residual_unit(body, filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i] - 1):
            body = residual_unit(body, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)

    # change the  stride of the last stage to 1.
    body = residual_unit(body, filter_list[-1], (1, 1), False, name='stage%d_unit%d' % (num_stage, 1),
                         bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    for j in range(units[-1] - 1):
        body = residual_unit(body, filter_list[num_stage], (1, 1), True, name='stage%d_unit%d' % (num_stage, j + 2),
                             bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)

    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')

    return relu1

def get_symbol_cbam_att(depth=50, att_input_type='rgbm', mlp_channel_att=True, att_branch_share=False,
                        channel_att_enable=False, spatial_att_enalbe=False,
                        bn_global_stats=False, stop_grad_delta=False,large_mask=True):
    assert att_input_type == 'rgbm' or att_input_type == 'mask', 'input_type only support rgbm and mask'
    if depth == 18:
        units = [2, 2, 2, 2]
    elif depth == 34:
        units = [3, 4, 6, 3]
    elif depth == 50:
        units = [3, 4, 6, 3]
    elif depth == 101:
        units = [3, 4, 23, 3]
    elif depth == 152:
        units = [3, 8, 36, 3]
    elif depth == 200:
        units = [3, 24, 36, 3]
    elif depth == 269:
        units = [3, 30, 48, 8]
    else:
        raise ValueError("no experiments done on detph {}, you can do it youself".format(depth))
    get_symbol_func = resnet_cbam_att

    return get_symbol_func(units=units, num_stage=4,
                           filter_list=[64, 256, 512, 1024, 2048] if depth >= 50 else [64, 64, 128, 256, 512],
                           data_type="imagenet", bottle_neck=True if depth >= 50 else False, memonger=True,
                           channel_att_enable=channel_att_enable, spatial_att_enalbe=spatial_att_enalbe,
                           mlp_channel_att=mlp_channel_att, att_branch_share=att_branch_share,
                           att_input_type=att_input_type)


if __name__ == '__main__':
    # sym = get_symbol(50)
    # soft mask
    # sym = get_symbol_delta(50, soft_mask=True)[0]
    # mx.viz.print_summary(symbol=sym, shape={"data": (8, 4, 256, 128)})
    # a = mx.viz.plot_network(symbol=sym, shape={"data": (8, 4, 256, 128)}, title='soft_mask_plot')
    # a.render()
    # hard mask
    # sym = get_symbol_delta(50, soft_mask=False)[0]
    # mx.viz.print_summary(symbol=sym, shape={"data": (8, 4, 256, 128), "binary_label": (8, 1, 64, 32)})
    # a = mx.viz.plot_network(symbol=sym, shape={"data": (8, 4, 256, 128), "binary_label": (8, 1, 64, 32)}, title='hard_mask_plot')
    # a.render()
    sym = get_symbol_cbam_att(50, channel_att_enable=True, spatial_att_enalbe=True,
                              att_input_type='mask', att_branch_share=False)
    mx.viz.print_summary(symbol=sym, shape={"data": (8, 3, 256, 128), "binary_label": (8, 1, 256, 128)})

    # sym = get_symbol_cbam_att(50, channel_att_enable=True, spatial_att_enalbe=False,
    #                           att_input_type='mask', att_branch_share=True)
    # mx.viz.print_summary(symbol=sym, shape={"data": (8, 3, 256, 128), "binary_label": (8, 1, 256, 128)})
