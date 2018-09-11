"""
the operators for mask reid task, including inner triplet loss, binary mask loss and siamese loss
by chencp
"""

import sys
sys.path.insert(0, "mxnet/python/")
import mxnet as mx
import numpy as np


class TripletLossMask(mx.operator.CustomOp):
    '''
    Triplet loss layer for mask guided
    the input features are concatenated along batch size dim, [fc, fc_body, fc_bg]
    '''

    def __init__(self, grad_scale=1.0, threshd=0.5):
        super(TripletLossMask, self).__init__()
        self.grad_scale = grad_scale
        self.threshd = threshd

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        batch_size_all = x.shape[0]
        assert batch_size_all % 3 == 0, 'batch dim error.'
        batch_size = batch_size_all // 3
        y = np.zeros((batch_size_all,))
        ctx = x.context
        for i in range(batch_size):
            pid = i + batch_size
            nid = pid + batch_size
            pdiff = x[i] - x[pid]
            ndiff = x[i] - x[nid]
            y[i] = mx.nd.sum(pdiff * pdiff).asnumpy()[0] - \
                   mx.nd.sum(ndiff * ndiff).asnumpy()[0] + self.threshd
            if y[i] < 0:
                y[i] = 0
        # y /= x.shape[0]
        self.assign(out_data[0], req[0], mx.nd.array(0.5*y, ctx=ctx))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x = in_data[0]
        y = out_data[0].asnumpy()

        # import pdb
        # pdb.set_trace()
        grad = in_grad[0]
        grad[:] = 0

        batch_size_all = x.shape[0]
        batch_size = batch_size_all // 3
        for i in range(batch_size):
            pid = i + batch_size
            nid = pid + batch_size

            if y[i] > 0:
                grad[i] += x[nid] - x[pid]
                grad[pid] += x[pid] - x[i]
                grad[nid] += x[i] - x[nid]
        grad *= self.grad_scale

        self.assign(in_grad[0], req[0], mx.nd.array(grad))


@mx.operator.register("TripletLossMask")
class TripletLossMaskProp(mx.operator.CustomOpProp):
    def __init__(self, grad_scale=1.0, threshd=0.5):
        super(TripletLossMaskProp, self).__init__(need_top_grad=False)
        self.grad_scale = float(grad_scale)
        self.threshd = float(threshd)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        # label_shape = (in_shape[0][0], )
        output_shape = (in_shape[0][0],)
        return [data_shape], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return TripletLossMask(self.grad_scale, self.threshd)


class MaskBinaryLoss(mx.operator.CustomOp):
    """mask binary loss"""
    def __init__(self, grad_scale=0.005,):
        super(MaskBinaryLoss, self).__init__()
        self.grad_scale = grad_scale

    def forward(self, is_train, req, in_data, out_data, aux):
        x, label = in_data
        x = x.asnumpy()
        label = label.asnumpy()

        y = np.square(x - label)

        self.assign(out_data[0], req[0], mx.nd.array(0.5*y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x, label = in_data
        # y = out_data[0].asnumpy()

        # grad = (x - label)/np.float(label.asnumpy().shape[0])
        grad = x - label
        grad *= self.grad_scale

        self.assign(in_grad[0], req[0], mx.nd.array(grad))


@mx.operator.register("MaskBinaryLoss")
class MaskBinaryLossProp(mx.operator.CustomOpProp):
    def __init__(self, grad_scale=0.005):
        super(MaskBinaryLossProp, self).__init__(need_top_grad=False)
        self.grad_scale = float(grad_scale)

    def list_arguments(self):
        return ['data', 'binary_label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        assert len(in_shape) == 2, "MaskBinaryLossOp input data: [data, binary_mask]"
        data_shape = in_shape[0]
        label_shape = in_shape[1]

        assert data_shape == label_shape, \
            'feature dim and mask dim mismatch: {} vs {}'.format(data_shape, label_shape)

        # output_shape = (in_shape[0][0],)
        output_shape = data_shape
        return [data_shape, label_shape], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return MaskBinaryLoss(self.grad_scale)


class SiameseLoss(mx.operator.CustomOp):
    '''
    Siamese loss layer for mask guided (in the form of triplet loss)
    the input features are concatenated along batch size dim, [fc, fc_body, fc_bg],
    but only compute siamese loss on fc,
    the data should be organized with positive and negative pairs (in image level, opposite to region level in mask)
    '''
    def __init__(self, grad_scale=1.0, threshd=0.5):
        super(SiameseLoss, self).__init__()
        self.grad_scale = grad_scale
        self.threshd = threshd

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        batch_size_all = x.shape[0]
        assert batch_size_all % 6 == 0, 'batch dim error.'
        batch_size = batch_size_all // 6
        y = np.zeros((batch_size_all,))
        ctx = x.context
        for i in range(batch_size):
            pid = i + 1 if i % 2 == 0 else i - 1
            nid = pid + batch_size
            pdiff = x[i] - x[pid]
            ndiff = x[i] - x[nid]
            y[i] = mx.nd.sum(pdiff * pdiff).asnumpy()[0] - \
                   mx.nd.sum(ndiff * ndiff).asnumpy()[0] + self.threshd
            if y[i] < 0:
                y[i] = 0
        # y /= x.shape[0]
        self.assign(out_data[0], req[0], mx.nd.array(0.5*y, ctx=ctx))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x = in_data[0]
        y = out_data[0].asnumpy()

        # import pdb
        # pdb.set_trace()
        grad = in_grad[0]
        grad[:] = 0

        batch_size_all = x.shape[0]
        batch_size = batch_size_all // 6
        for i in range(batch_size):
            pid = i + 1 if i % 2 == 0 else i - 1
            nid = pid + batch_size

            if y[i] > 0:
                grad[i] += x[nid] - x[pid]
                grad[pid] += x[pid] - x[i]
                grad[nid] += x[i] - x[nid]
        grad *= self.grad_scale

        self.assign(in_grad[0], req[0], mx.nd.array(grad))


@mx.operator.register("SiameseLoss")
class SiameseLossProp(mx.operator.CustomOpProp):
    def __init__(self, grad_scale=1.0, threshd=0.5):
        super(SiameseLossProp, self).__init__(need_top_grad=False)
        self.grad_scale = float(grad_scale)
        self.threshd = float(threshd)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        # label_shape = (in_shape[0][0], )
        output_shape = (in_shape[0][0],)
        return [data_shape], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return SiameseLoss(self.grad_scale, self.threshd)


def test_loss_operater(loss_fun='triplet_loss'):
    """check the gradient of defined operators
    """
    batch_size_all = 18
    embedding_dim = 120
    grad_scale = 0.7
    triplet_threshd = 0.5
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('binary_label')

    if loss_fun == 'triplet_loss':
        data_shape = (batch_size_all, embedding_dim)
        input_data = {'data': np.random.normal(0, 1, data_shape)}
        sym_custom = mx.symbol.Custom(data=data,
                                      grad_scale=grad_scale, threshd=triplet_threshd,
                                      op_type='TripletLossMask', name='loss_op')
        executor = sym_custom.simple_bind(ctx=mx.cpu(), data=data_shape)
    elif loss_fun == 'siamese_loss':
        data_shape = (batch_size_all, embedding_dim)
        input_data = {'data': np.random.normal(0, 1, data_shape)}
        sym_custom = mx.symbol.Custom(data=data,
                                      grad_scale=grad_scale, threshd=triplet_threshd,
                                      op_type='SiameseLoss', name='loss_op')
        executor = sym_custom.simple_bind(ctx=mx.cpu(), data=data_shape)
    elif loss_fun == 'binary_mask_loss':
        data_shape = (batch_size_all, embedding_dim, embedding_dim)
        input_data = {'data': np.random.uniform(0, 1, data_shape),
                      'binary_label': np.reshape(np.random.binomial(n=1, size=batch_size_all*embedding_dim*embedding_dim, p=0.5),
                                                data_shape)}
        sym_custom = mx.symbol.Custom(data=data, binary_label=label,
                                      grad_scale=grad_scale,
                                      op_type='MaskBinaryLoss', name='loss_op')
        executor = sym_custom.simple_bind(ctx=mx.cpu(), data=data_shape, binary_label=data_shape)
    else:
        raise NotImplemented

    def forward(data, label=0):
        if loss_fun != 'binary_mask_loss':
            executor.forward(is_train=True, data=data)
            return np.sum(executor.output_dict['loss_op_output'].asnumpy())
        else:
            executor.forward(is_train=True, data=data, binary_label=label)
            return np.sum(np.float64(executor.output_dict['loss_op_output'].asnumpy()))

    def backward():
        executor.backward()
        return executor.grad_dict

    def gradient_check(name, i, j):
        '''gradient check on x[i, j]
        '''
        eps = 1e-4
        threshold = 1e-2
        reldiff = lambda a, b: abs(a - b) / (abs(a) + abs(b) + 1e-6)

        # calculate by backward
        label = 0
        if loss_fun == 'binary_mask_loss':
            label = input_data['binary_label']
        output = forward(data=input_data['data'], label=label)
        # print('forward pass:{}'.format(output))
        grad_dict = backward()
        grad = grad_dict[name].asnumpy()[i, j]
        grad = grad[0] if loss_fun == 'binary_mask_loss' else grad

        # calculate by \delta f / 2 * eps
        if loss_fun == 'binary_mask_loss':
            input_data[name][i, j, 0] -= eps
        else:
            input_data[name][i, j] -= eps
        loss1 = forward(data=input_data['data'], label=label)
        if loss_fun == 'binary_mask_loss':
            input_data[name][i, j, 0] += 2 * eps
        else:
            input_data[name][i, j] += 2 * eps
        loss2 = forward(data=input_data['data'], label=label)

        grad_expect = grad_scale * (loss2 - loss1) / (2 * eps)
        grad_expect = np.float32(grad_expect)
        # import pdb
        # pdb.set_trace()
        print(grad_expect, grad)
        # check
        rel_err = reldiff(grad_expect, grad)
        if rel_err > threshold:
            print 'gradient check failed'
            print 'expected %lf given %lf, relative error %lf' % (grad_expect, grad, rel_err)
            return False
        else:
            print 'gradient check pass'
            return True

    data_gc_pass = 0
    for i in range(batch_size_all):
        for j in range(embedding_dim):
            print 'gradient check on data[%d, %d]' % (i, j)
            if gradient_check('data', i, j):
                data_gc_pass += 1

    print '===Summary==='
    print 'gradient pass ratio on data is %lf' % (float(data_gc_pass) / (embedding_dim*batch_size_all))


if __name__ == '__main__':
    loss_fun = 'binary_mask_loss'
    print('gradient checking of {}:'.format(loss_fun))
    test_loss_operater(loss_fun=loss_fun)

