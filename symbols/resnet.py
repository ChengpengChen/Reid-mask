"""
Reproducing paper:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
"""
import mxnet as mx

use_global_stats = False
fix_gamma = False
eps = 2e-5
bn_mom = 0


def residual_unit(data, num_filter, stride, dilate, dim_match, name):
    s = stride
    d = dilate

    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=fix_gamma, use_global_stats=use_global_stats,
                           eps=eps, momentum=bn_mom, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1),
                               no_bias=True, name=name + '_conv1')

    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=fix_gamma, use_global_stats=use_global_stats,
                           eps=eps, momentum=bn_mom, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), pad=(d, d),
                               stride=(s, s), dilate=(d, d), no_bias=True, name=name + '_conv2')

    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=fix_gamma, use_global_stats=use_global_stats,
                           eps=eps, momentum=bn_mom, name=name + '_bn3')
    act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
    conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), no_bias=True, name=name + '_conv3')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=(s, s),
                                      no_bias=True, name=name + '_sc')

    shortcut._set_attr(mirror_stage='True')
    return conv3 + shortcut


def resnet101_c3():
    # preprocessing
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, use_global_stats=use_global_stats, eps=eps, momentum=bn_mom,
                            name='bn_data')

    # C1, 7x7
    data = mx.sym.Convolution(data=data, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                              no_bias=True, name="conv0")
    data = mx.sym.BatchNorm(data=data, fix_gamma=fix_gamma, use_global_stats=use_global_stats, eps=eps, momentum=bn_mom,
                            name='bn0')
    data = mx.sym.Activation(data=data, act_type='relu', name='relu0')

    # C2, 3 blocks
    data = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    #                          c,   s, d, sc
    data = residual_unit(data, 256, 1, 1, False, "stage1_unit1")
    data = residual_unit(data, 256, 1, 1, True, "stage1_unit2")
    data = residual_unit(data, 256, 1, 1, True, "stage1_unit3")

    # C3, 4 blocks
    data = residual_unit(data, 512, 2, 1, False, "stage2_unit1")
    data = residual_unit(data, 512, 1, 1, True, "stage2_unit2")
    data = residual_unit(data, 512, 1, 1, True, "stage2_unit3")
    data = residual_unit(data, 512, 1, 1, True, "stage2_unit4")

    # os = 8
    return data


def resnet101_c4():
    data = resnet101_c3()

    # C4, 23 blocks
    data = residual_unit(data, 1024, 2, 1, False, "stage3_unit1")
    data = residual_unit(data, 1024, 1, 1, True, "stage3_unit2")
    data = residual_unit(data, 1024, 1, 1, True, "stage3_unit3")
    data = residual_unit(data, 1024, 1, 1, True, "stage3_unit4")
    data = residual_unit(data, 1024, 1, 1, True, "stage3_unit5")
    data = residual_unit(data, 1024, 1, 1, True, "stage3_unit6")
    data = residual_unit(data, 1024, 1, 1, True, "stage3_unit7")
    data = residual_unit(data, 1024, 1, 1, True, "stage3_unit8")
    data = residual_unit(data, 1024, 1, 1, True, "stage3_unit9")
    data = residual_unit(data, 1024, 1, 1, True, "stage3_unit10")
    data = residual_unit(data, 1024, 1, 1, True, "stage3_unit11")
    data = residual_unit(data, 1024, 1, 1, True, "stage3_unit12")
    data = residual_unit(data, 1024, 1, 1, True, "stage3_unit13")
    data = residual_unit(data, 1024, 1, 1, True, "stage3_unit14")
    data = residual_unit(data, 1024, 1, 1, True, "stage3_unit15")
    data = residual_unit(data, 1024, 1, 1, True, "stage3_unit16")
    data = residual_unit(data, 1024, 1, 1, True, "stage3_unit17")
    data = residual_unit(data, 1024, 1, 1, True, "stage3_unit18")
    data = residual_unit(data, 1024, 1, 1, True, "stage3_unit19")
    data = residual_unit(data, 1024, 1, 1, True, "stage3_unit20")
    data = residual_unit(data, 1024, 1, 1, True, "stage3_unit21")
    data = residual_unit(data, 1024, 1, 1, True, "stage3_unit22")
    data = residual_unit(data, 1024, 1, 1, True, "stage3_unit23")

    # os = 16
    return data


def resnet101_c5():
    data = resnet101_c4()

    # C5, 3 blocks
    data = residual_unit(data, 2048, 2, 1, False, "stage4_unit1")
    data = residual_unit(data, 2048, 1, 1, True, "stage4_unit2")
    data = residual_unit(data, 2048, 1, 1, True, "stage4_unit3")

    # os = 32
    return data


# resnet-50
def resnet50_c4(rate):
    c4_stride = 2 / rate
    c4_dilate = 1 * rate

    data = resnet101_c3()

    # C4, 6 blocks
    data = residual_unit(data, 1024, c4_stride, 1, False, "stage3_unit1")
    data = residual_unit(data, 1024, 1, c4_dilate, True, "stage3_unit2")
    data = residual_unit(data, 1024, 1, c4_dilate, True, "stage3_unit3")
    data = residual_unit(data, 1024, 1, c4_dilate, True, "stage3_unit4")
    data = residual_unit(data, 1024, 1, c4_dilate, True, "stage3_unit5")
    data = residual_unit(data, 1024, 1, c4_dilate, True, "stage3_unit6")

    # os = 16
    return data
