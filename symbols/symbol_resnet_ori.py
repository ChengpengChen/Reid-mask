'''
Reproducing paper:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''
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
        conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
                                   pad=(0, 0),
                                   no_bias=1, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=1e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride,
                                   pad=(1, 1),
                                   no_bias=1, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=1e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=1,
                                   workspace=workspace, name=name + '_conv3')
        bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=1e-5, momentum=bn_mom, name=name + '_bn3')
        # act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        if dim_match:
            shortcut = data
        else:
            conv_shortcut = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride,
                                               no_bias=1, workspace=workspace, name=name + '_downsample_0')
            shortcut = mx.sym.BatchNorm(data=conv_shortcut, fix_gamma=False, eps=1e-5,
                                        momentum=bn_mom, name=name + '_downsample_1')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        act3 = mx.sym.Activation(data=bn3 + shortcut, act_type='relu', name=name + '_relu3')

        return act3
    else:
        raise NotImplementedError
        # bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '.bn1')
        # act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '.relu1')
        # conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
        #                            no_bias=1, workspace=workspace, name=name + '.conv1')
        # bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '.bn2')
        # act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '.relu2')
        # conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
        #                            no_bias=1, workspace=workspace, name=name + '.conv2')
        # if dim_match:
        #     shortcut = data
        # else:
        #     shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=1,
        #                                   workspace=workspace, name=name + '.sc')
        # if memonger:
        #     shortcut._set_attr(mirror_stage='True')
        # return conv2 + shortcut


def resnet(units, num_stage, filter_list, data_type, bottle_neck=True, bn_mom=0.9, workspace=1024, memonger=False):
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
    data = mx.sym.Variable(name='data')  # remove bn on data, same as original paper
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=1e-5, momentum=bn_mom, name='bn_data')
    if data_type == 'cifar10':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                  no_bias=1, name="conv1", workspace=workspace)
    elif data_type == 'imagenet':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                  no_bias=1, name="conv1", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=1e-5, momentum=bn_mom, name='bn1')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='maxpool')
    else:
        raise ValueError("do not support {} yet".format(data_type))
    for i in range(num_stage):
        body = residual_unit(body, filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False,
                             name='layer%d_%d' % (i + 1, 0), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i] - 1):
            body = residual_unit(body, filter_list[i + 1], (1, 1), True, name='layer%d_%d' % (i + 1, j + 1),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    output = mx.symbol.Pooling(data=body, global_pool=True, pool_type='avg', name='global_avg')
    output = mx.symbol.FullyConnected(data=output, num_hidden=365, name='fc')

    return output


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
                    data_type="imagenet", bottle_neck=True if depth >= 50 else False, memonger=True)
    return symbol

if __name__ == '__main__':
    sym = get_symbol(50)
    mx.viz.print_summary(symbol=sym, shape={"data": (8, 3, 224, 224)})