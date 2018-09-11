"""
to extract feature map of stage2 to check the scale of the output
to validate the scale for soft mask
by chencp, on 2018.08.14
"""
import numpy as np
import cv2
import mxnet as mx
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

img_path = '/home/chencp/dataset/Market-1501-v15.09.15/query/0291_c3s3_079969_00.jpg'
im = cv2.imread(img_path)
img = cv2.resize(im, (128, 256))
img = np.transpose(img, (2, 0, 1))
img=img[np.newaxis][::-1]
print('the image shape:', img.shape)

# load the model
load_model='../pretrain_models/resnet-50'
data_shapes=[('data', (1,3,256,128))]
sym, arg_params, aux_params = mx.model.load_checkpoint(load_model, 0)
fea_stage2=sym.get_internals()['_plus6_output']
model=mx.mod.Module(symbol=fea_stage2, context=mx.gpu(0), label_names=None)
model.bind(data_shapes=data_shapes, for_training=False, force_rebind=True)
model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

# forward the image
data=[mx.nd.array(img)]
model.forward(Batch(data=data), is_train=False)
output = model.get_outputs()[0]
output = output.asnumpy()
print('get feature with shape', output.shape)
import pdb
pdb.set_trace()

