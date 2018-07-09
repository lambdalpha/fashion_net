from model import FashionNet
from mxnet import gluon
import mxnet as mx
from mxnet import autograd
from data_loader import ImageDataIter
from mxnet import lr_scheduler
from mxnet.gluon import nn
from mxnet import nd
import sys
from loss import WeightedCrossEntropyLoss


LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 1e-8
BATCH_SIZE = 16


class SimpleLRScheduler(lr_scheduler.LRScheduler):
    def __init__(self, learning_rate=0.1):
        super(SimpleLRScheduler, self).__init__()
        self.learning_rate = learning_rate

    def __call__(self, num_update):
        if self.learning_rate < MIN_LEARNING_RATE:
            return MIN_LEARNING_RATE
        return self.learning_rate


def get_data(img_dir, img_attr_file, img_landmark_file, ctx, batch_size):
    data_iter = ImageDataIter(img_dir, img_attr_file, img_landmark_file, ctx, batch_size=batch_size)
    while True:
        data, label = data_iter.next()
        yield data, label


def train(net, trainer, img_dir, img_attr_file, img_landmark_file, ctx, batch_size, epochs, out_model_file, lr_schedule):
    loss_softmax = gluon.loss.SoftmaxCrossEntropyLoss()
    loss_weighted_cross_entropy = WeightedCrossEntropyLoss()
    loss_l2 = gluon.loss.L2Loss()
    # loss_hinge = gluon.loss.HingeLoss()
    for epoch in range(epochs):
        data_iter = get_data(img_dir, img_attr_file, img_landmark_file, ctx, batch_size)
        # total_loss = 0
        loss_iter = 0
        for i, (data, label) in enumerate(data_iter):
            if not data:
                break
            with autograd.record():
                d = data[0]
                img_files = data[1]
                l, vs, classifier_output, _, _ = net(d)

                label, vis, landmarks = label
                # print(vis.shape)
                vis_data = [vis[:, k] for k in range(vis.shape[1])]
                loss_landmark = loss_l2(l, landmarks) / 100
                loss_attr = loss_weighted_cross_entropy(classifier_output, label)
                loss = loss_landmark + loss_attr
                for v, d in zip(vs, vis_data):
                    loss = loss + loss_softmax(v, d)

            # print(loss)
            loss.backward()
            loss_iter += nd.mean(loss).asscalar()
            trainer.step(batch_size)

            if (i + 1) % 40 == 0:
                print(img_files)
                print(l)
                print(landmarks)
                print(loss_landmark)
                print(loss_attr)
                print('epoch: %d, iter: %d, loss: %f' % (epoch, i + 1, loss_iter))
                loss_iter = 0
        # half the learning rate every epoch
        lr_schedule.learning_rate /= 2.0
        net.collect_params().save(out_model_file + '_' + str(epoch))


if __name__ == '__main__':
    img_dir = sys.argv[1]
    img_attr_file = sys.argv[2]
    img_landmark_file = sys.argv[3]
    out_model = sys.argv[4]
    model_file = None
    if len(sys.argv) == 6:
        model_file = sys.argv[5]
    #
    # img_dir = '/data/image_data/category_and_attribute_prediction_benchmark/Img/'
    # img_attr_file = '/data/image_data/category_and_attribute_prediction_benchmark/Anno/list_attr_img.txt'
    # img_landmark_file = '/data/image_data/category_and_attribute_prediction_benchmark/Anno/list_landmarks.txt'

    ctx = mx.gpu()
    fashion_net = FashionNet()

    if issubclass(FashionNet, nn.HybridBlock):
        fashion_net.hybridize()

    if model_file:
        # start from checkpoint
        fashion_net.collect_params().load(model_file, ctx=ctx)
    else:
        # xavier init
        fashion_net.initialize(ctx=ctx, init=mx.init.Xavier())

    # learning rate scheduler
    lr_sch = SimpleLRScheduler(learning_rate=LEARNING_RATE)

    net_trainer = gluon.Trainer(fashion_net.collect_params(), 'Adam', {'lr_scheduler': lr_sch})
    train(fashion_net, net_trainer, img_dir, img_attr_file, img_landmark_file, ctx, BATCH_SIZE, 5, out_model, lr_sch)
    # fashion_net.collect_params().save(out_model)
