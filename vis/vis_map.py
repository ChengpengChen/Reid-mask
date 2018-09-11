# extract attention maps for visualization, by chencp
from __future__ import print_function
import os
import cv2
import numpy as np
import mxnet as mx

from collections import namedtuple
# operators
import operators.triplet_loss
import operators.triplet_loss_inner
import operators.loss_layers_mask

Batch = namedtuple('Batch', ['data'])

rgb_root_dir = '/home/chencp/dataset/Market-1501-v15.09.15/bounding_box_test'
mask_root_dir = '/home/chencp/dataset/annotation-market1501/bounding_box_test_seg'


def heatmap_vis(out_score_map, im, im_shape=(224, 112)):
    # heatmap = cv2.resize(out_score_map.squeeze(), im_shape, interpolation=cv2.INTER_CUBIC)
    # heatmap = cv2.resize(out_score_map.squeeze(), (im_shape[1], im_shape[0]))
    heatmap = out_score_map
    heatmap = heatmap / np.max(heatmap)
    gcam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    # print np.float32(gcam)
    gcam = np.float32(gcam) + np.float32(im)
    # gcam = np.float32(gcam)
    gcam = np.uint8(255 * gcam / np.max(gcam))
    return gcam


def get_img(rgb_path, resize, data_shape, mask_path=None):
    """
    process the image, and take augmentation (resize, crop, mirror)
    """
    im = cv2.imread(rgb_path)
    if mask_path:
        im_mask = cv2.imread(mask_path)
        im_new = np.zeros_like(im_mask[:, :, 0])  # take only one channel
        # import pdb
        # pdb.set_trace()
        ind = np.where(im_mask[:, :, 0] >= 128)
        im_new[ind] = 255
        im_mask = im_new[:, :, np.newaxis]
        im = np.concatenate([im, im_mask], axis=2)

    crop_h, crop_w = data_shape[1:]
    ori_h, ori_w = im.shape[:2]
    # resize = resize
    if ori_h < ori_w:
        h, w = resize, int(float(resize) / ori_h * ori_w)
    else:
        h, w = int(float(resize) / ori_w * ori_h), resize

    if h != ori_h:
        # im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        im = cv2.resize(im, (w, h))

    x, y = (w - crop_w) / 2, (h - crop_h) / 2

    im = im[y:y + crop_h, x:x + crop_w, :]

    im = np.transpose(im, (2, 0, 1))
    newim = np.zeros_like(im)
    newim[0] = im[2]
    newim[1] = im[1]
    newim[2] = im[0]
    if im.shape[0] == 4:
        newim[3] = im[3]

    return mx.nd.array(newim[np.newaxis])


def extract_map(model, list_file, save_dir, resize=128, data_shape=(3, 224, 112), data_type='rgb'):
    mask_data = False if data_type == 'rgb' else True
    fin = open(list_file, 'r')
    for line in fin:
        rgb_path = os.path.join(rgb_root_dir, line[:-1])
        mask_path = os.path.join(mask_root_dir, line[:-1].replace('.jpg', '.png'))
        # map_name = os.path.join(save_dir, line[:-1].replace('.jpg', '.npy'))
        map_save_name = os.path.join(save_dir, line[:-1].replace('.jpg', '_map.png'))
        mask_save_name = os.path.join(save_dir, line[:-1].replace('.jpg', '_mask.png'))
        im_save_name = os.path.join(save_dir, line[:-1])
        gcam_save_name = os.path.join(save_dir, line[:-1].replace('.jpg', '_gcam.jpg'))

        im_all = get_img(rgb_path, resize, data_shape, mask_path)
        im_input = im_all if mask_data else im_all[:, :3]
        im_temp = im_all[0, :3].asnumpy()
        im = np.zeros_like(im_temp)
        im[0] = im_temp[2]
        im[1] = im_temp[1]
        im[2] = im_temp[0]
        im = np.transpose(im, (1, 2, 0))
        im = np.uint8(im)
        im_mask = im_all[0, 3].asnumpy()

        model.forward(Batch(data=[im_input]), is_train=False)
        output = model.get_outputs()[0].asnumpy()
        output_resize = cv2.resize(output.squeeze(), (data_shape[2], data_shape[1]))

        im_gcam = heatmap_vis(output_resize, im, im_shape=data_shape[1:])
        # save the images
        cv2.imwrite(mask_save_name, im_mask*2)
        cv2.imwrite(map_save_name, np.squeeze(output_resize*256))
        cv2.imwrite(im_save_name, im)
        cv2.imwrite(gcam_save_name, im_gcam)

    fin.close()


if __name__ == '__main__':
    list_file = 'sample.lst'
    save_dir = 'extract_map_rgb_seg_0815-v2'
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    resize = 144
    crop_size = 128
    data_type = 'rgb'

    prefix = "rgb_soft_mask_0815-v2"
    dataset = 'market'
    # tag = '_full'  # to indicate the mat files
    epoch_idx = 300
    context = mx.gpu(0)

    symbol, arg_params, aux_params = mx.model.load_checkpoint("../models/%s/%s" % (dataset, prefix), epoch_idx)
    delta_sigmoid = symbol.get_internals()['delta_sigmoid_output']

    if data_type == 'rgbm':
        data_shape = (4, crop_size * 2, crop_size)
    else:
        data_shape = (3, crop_size * 2, crop_size)

    data_shapes = [('data', (1, data_shape[0], crop_size * 2, crop_size))]
    data_names = ['data']

    model = mx.mod.Module(symbol=delta_sigmoid, context=context, data_names=data_names, label_names=None)
    model.bind(data_shapes=data_shapes, for_training=False, force_rebind=True)

    model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

    extract_map(model, list_file, save_dir, resize=resize, data_shape=data_shape, data_type=data_type)
