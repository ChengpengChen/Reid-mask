# extend pre-trained model with 3 channels input to 4 channels
# use zero init or mean init
# and then save the checkpoint for further training
import mxnet as mx
import numpy as np
import argparse

def extend_chn():
    init_method = args.init_method
    assert init_method == 'zero' or init_method == 'mean', 'only support zero and mean init methods yet'
    # model_load_prefix = 'pretrained_model/Inception-BN'
    # model_load_epoch = 126
    model_load_prefix = args.model_load_prefix
    model_load_epoch = args.model_load_epoch
    symbol, arg_params, aux_params = mx.model.load_checkpoint(model_load_prefix, model_load_epoch)
    conv1_name = 'conv_1_weight' if 'inception-bn' in model_load_prefix else 'conv0_weight'
    conv1_w = arg_params[conv1_name]
    conv1_shape = conv1_w.shape
    assert conv1_shape[1] == 3, 'something wrong in the input dim'
    if init_method == 'zero':
        new_c = np.zeros(shape=(conv1_shape[0], 1, conv1_shape[2], conv1_shape[3]), dtype=np.float32)
    else:
        new_c = np.mean(conv1_w.asnumpy(), axis=1)[:, np.newaxis]

    new_conv1_w = np.concatenate([conv1_w.asnumpy(), new_c], axis=1)
    new_conv1_w = mx.nd.array(new_conv1_w)
    # update the params
    arg_params[conv1_name] = new_conv1_w

    if args.extend_bn:
        bn_data_name = ['bn_data_beta', 'bn_data_gamma']
        for n in bn_data_name:
            bn_data = arg_params[n].asnumpy()
            # new_bn_data = np.concatenate([bn_data, [0.]])
            new_bn_data = np.concatenate([bn_data, [np.mean(bn_data)]])
            new_bn_data = mx.nd.array(new_bn_data)
            arg_params[n] = new_bn_data
        bn_moving_name = ['bn_data_moving_mean', 'bn_data_moving_var']
        for n in bn_moving_name:
            bn_data = aux_params[n].asnumpy()
            # new_bn_data = np.concatenate([bn_data, [0.]])
            new_bn_data = np.concatenate([bn_data, [np.mean(bn_data)]])
            new_bn_data = mx.nd.array(new_bn_data)
            aux_params[n] = new_bn_data

    model_load_prefix_new = '{}-{}-init'.format(model_load_prefix, init_method)
    mx.model.save_checkpoint(model_load_prefix_new, model_load_epoch, symbol, arg_params, aux_params)

    print('done extend the pre-trained model with 3 channels to 4 channels\n\tand saved the model to {}'
          .format(model_load_prefix_new))
    

def parse_args():
    parser = argparse.ArgumentParser(
        description='extend 3 channels to 4 channels')
    parser.add_argument('--model_load_prefix', type=str,
                        default="pretrain_models/resnet-50",
                        help='original model prefix')
    parser.add_argument('--model_load_epoch', type=int, default=0,
                        help='load epoch from original model')
    parser.add_argument('--init_method', type=str, default="zero",
                        help='zero or mean init methods')
    parser.add_argument('--extend_bn', default=False, action='store_true',
                        help='also extend the first bn layer with 4-channels')
    return parser.parse_args()


if __name__ == '__main__':
    global args
    args = parse_args()
    extend_chn()
