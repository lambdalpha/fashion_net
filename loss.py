from mxnet.gluon.loss import Loss
from mxnet import nd


# From paper: DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations
# fixme why this is so low
class WeightedCrossEntropyLoss(Loss):
    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(WeightedCrossEntropyLoss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        # batch_size, n = label.shape
        prob = F.softmax(pred)
        # print(prob)
        # print(label)
        new_label = label > 0.5
        new_label = new_label.astype('float32')
        # print('new_label', new_label)
        # print(n)
        # print(F.sum(new_label, axis=1) / n)
        # w_pos = (F.sum(new_label, axis=1) / n).reshape((batch_size, 1))
        # w_neg = 1 - w_pos
        # print('w_pos', w_pos)
        # print(w_neg)

        # return - F.sum(w_pos * new_label * F.log(prob) + w_neg * (1 - new_label) * F.log(1-prob), axis=1)

        return - F.sum(new_label * F.log(prob) + (1 - new_label) * F.log(1 - prob), axis=1)


if __name__ == '__main__':
    loss_func = WeightedCrossEntropyLoss()

    p = nd.random.normal(scale=1., shape=(10, 100))
    l = nd.random.multinomial(nd.array([0.9, 0.1]), shape=[10, 100])

    loss = loss_func(p, l)
    print(loss)
