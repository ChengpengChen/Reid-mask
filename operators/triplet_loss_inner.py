"""
inner triplet loss for mask reid task, refer to triplet_loss.py file
note that, the gradient checking code is not modified yet
by chencp
"""

from __future__ import division
import mxnet as mx
from sklearn.metrics import euclidean_distances
import numpy as np


# def euclidean_distances(X, squared=False):
#     XX = mx.nd.sum(mx.nd.square(X), axis=1).expand_dims(1)
#
#     distances = mx.nd.dot(X, X.T)
#     distances *= -2
#     distances += XX
#     distances += XX.T
#     mx.nd.relu(distances, out=distances)
#
#     distances.reshape(-1)[::distances.shape[0] + 1] = 0.0
#
#     return distances if squared else mx.nd.sqrt(distances, out=distances)


class TripletLossInner(mx.operator.CustomOp):
    def __init__(self, margin=0.5, grad_scale=1.0, **kwargs):
        self.grad_scale = grad_scale
        self.margin = margin

        self.dist = None

        self.nid = []
        self.pid = []

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        batch_size_all = x.shape[0]
        assert batch_size_all % 3 == 0, 'batch dim error.'
        batch_size = batch_size_all // 3
        y = mx.nd.zeros((batch_size_all,), ctx=x.context)

        dist = np.clip(euclidean_distances(x.asnumpy(), squared=True), 1e-12, None)
        dist = np.sqrt(dist)
        self.dist = dist.copy()

        for i in range(batch_size):
            pid = i + batch_size
            nid = pid + batch_size
            y[i] = dist[i, pid] - dist[i, nid] + self.margin

        y = mx.nd.relu(y)
        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x = in_data[0]
        y = out_data[0].asnumpy()
        grad = in_grad[0]
        grad[:] = 0
        dist = self.dist

        batch_size_all = x.shape[0]
        batch_size = batch_size_all // 3
        for i in range(batch_size):
            pid = i + batch_size
            nid = pid + batch_size

            if y[i] > 0:
                grad[i] += ((x[nid] - x[i]) / dist[i, nid]) + ((x[i] - x[pid]) / dist[i, pid])
                grad[pid] += (x[pid] - x[i]) / dist[i, pid]
                grad[nid] += (x[i] - x[nid]) / dist[i, nid]

        grad *= self.grad_scale

        self.assign(in_grad[0], req[0], mx.nd.array(grad))


@mx.operator.register("TripletLossInner")
class TripletLossInnerProp(mx.operator.CustomOpProp):
    def __init__(self, margin=0.5, grad_scale=1.0, **kwargs):
        super(TripletLossInnerProp, self).__init__(need_top_grad=False)
        self.grad_scale = float(grad_scale)
        self.margin = float(margin)
        self.kwargs = kwargs

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = (in_shape[0][0],)
        return [data_shape], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return TripletLossInner(self.margin, self.grad_scale, **self.kwargs)


if __name__ == '__main__':
    """
    todo: rewrite the gradient check for inner triplet loss (2018.08.10)
    the codes below are for norm tiplet loss with k_size and p_size
    """

    import sys
    import os
    import torch
    from collections import namedtuple

    Batch = namedtuple("Batch", ['data'])

    sys.path.append(".")
    from utils.group_iterator import GroupIterator

    mx.random.seed(0)

    p_size = 16
    k_size = 3
    batch_size = p_size * k_size
    margin = 0.3
    num_dim = 1024

    data_dir = "/home/chencp/dataset/Market-1501-v15.09.15/"
    train = GroupIterator(data_dir=os.path.join(data_dir, "bounding_box_train"),
                          p_size=p_size, k_size=k_size, image_size=(112 * 2, 112),
                          resize_shorter=128, rand_mirror=True, rand_crop=True, num_worker=1, random_seed=0)

    sym, arg_params, aux_params = mx.model.load_checkpoint("pretrain_models/inception-bn", 126)
    # sym, arg_params, aux_params = mx.model.load_checkpoint("pretrain_models/resnet-50", 0)
    sym = sym.get_internals()["ch_concat_5b_chconcat_output"]
    pooling = mx.symbol.Pooling(data=sym, kernel=(1, 1), global_pool=True, pool_type='avg', name='global_pool')
    flatten = mx.symbol.Flatten(data=pooling, name='flatten')

    model = mx.mod.Module(symbol=flatten, context=mx.gpu(0), label_names=None)
    model.bind(data_shapes=[("data", (batch_size, 3, 224, 112))], for_training=False)
    model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

    ranking_loss = torch.nn.MarginRankingLoss(margin)

    for i in range(50):
        raw_data = next(train)

        model.forward(Batch(data=raw_data.data))
        data = model.get_outputs()[0]

        data = data.as_in_context(mx.cpu())
        data.attach_grad()

        dist = euclidean_distances(data.asnumpy())

        pdists = []
        ndists = []
        with mx.autograd.record():
            loss = 0

            for i in range(batch_size):
                idx = i // k_size
                start = idx * k_size
                end = start + k_size

                tmp = np.zeros((batch_size,))
                tmp[start:end] = 1e8

                pid = np.argmax(dist[i][start:end]) + start
                nid = np.argmin(dist[i] + tmp)

                pdist = mx.nd.sqrt(mx.nd.sum(mx.nd.square(data[i] - data[pid])))
                ndist = mx.nd.sqrt(mx.nd.sum(mx.nd.square(data[i] - data[nid])))
                loss = loss + mx.nd.relu(pdist - ndist + margin)

                pdists.append(pdist)
                ndists.append(ndist)

            loss = loss / batch_size

        loss.backward()
        # print(loss)
        pdists = mx.nd.stack(*pdists, axis=0).asnumpy().squeeze()
        ndists = mx.nd.stack(*ndists, axis=0).asnumpy().squeeze()
        loss = ranking_loss(torch.Tensor(ndists), torch.Tensor(pdists), torch.ones(batch_size))
        # print(loss)

        X = mx.symbol.Variable("data")

        symbol = mx.symbol.Custom(data=X, margin=margin, grad_scale=1 / batch_size,
                                  op_type='TripletLossInner', name='triplet')
        exe = symbol.simple_bind(ctx=mx.cpu(), data=(batch_size, num_dim))

        exe.forward(data=data)
        exe.backward()

        grad = exe.grad_arrays[0]

        print(grad)
        print(data.grad)

        print(mx.nd.abs(grad - data.grad).sum())
 