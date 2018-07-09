from mxnet import gluon
import mxnet.gluon.nn as nn
import mxnet as mx
from mxnet import nd
from mxnet import autograd


NUM_CLASSES = 1000


# fixme add batch norm
# todo use pre-trained resnet as base net
class FashionNet(nn.Block):
    def __init__(self, **kwargs):
        super(FashionNet, self).__init__(**kwargs)
        self.base_net = nn.Sequential()
        self.pose_net = nn.Sequential()
        self.global_net = nn.Sequential()

        self.loc = nn.Dense(16)
        self.vis1 = nn.Dense(2)
        self.vis2 = nn.Dense(2)
        self.vis3 = nn.Dense(2)
        self.vis4 = nn.Dense(2)
        self.vis5 = nn.Dense(2)
        self.vis6 = nn.Dense(2)
        self.vis7 = nn.Dense(2)
        self.vis8 = nn.Dense(2)

        self.classifier = nn.Dense(NUM_CLASSES)

        with self.base_net.name_scope():
            self.base_net.add(nn.Conv2D(64, 3, activation='relu', padding=1))
            self.base_net.add(nn.Conv2D(64, 3, activation='relu', padding=1))
            self.base_net.add(nn.MaxPool2D(pool_size=2))

            self.base_net.add(nn.Conv2D(128, 3, activation='relu', padding=1))
            self.base_net.add(nn.Conv2D(128, 3, activation='relu', padding=1))
            self.base_net.add(nn.MaxPool2D(pool_size=2))

            self.base_net.add(nn.Conv2D(256, 3, activation='relu', padding=1))
            self.base_net.add(nn.Conv2D(256, 3, activation='relu', padding=1))
            self.base_net.add(nn.Conv2D(256, 3, activation='relu', padding=1))
            self.base_net.add(nn.MaxPool2D(pool_size=2))

            self.base_net.add(nn.Conv2D(512, 3, activation='relu', padding=1))
            self.base_net.add(nn.Conv2D(512, 3, activation='relu', padding=1))
            self.base_net.add(nn.Conv2D(512, 3, activation='relu', padding=1))

        with self.pose_net.name_scope():
            self.pose_net.add(nn.MaxPool2D(pool_size=2))
            self.pose_net.add(nn.Conv2D(512, 3, activation='relu', padding=1))
            self.pose_net.add(nn.Conv2D(512, 3, activation='relu', padding=1))
            self.pose_net.add(nn.Conv2D(512, 3, activation='relu', padding=1))
            self.pose_net.add(nn.MaxPool2D(pool_size=2))
            self.pose_net.add(nn.Flatten())
            self.pose_net.add(nn.Dense(1024, activation='relu'))
            self.pose_net.add(nn.Dense(1024, activation='relu'))

        with self.global_net.name_scope():
            self.global_net.add(nn.MaxPool2D(pool_size=2))
            self.global_net.add(nn.Conv2D(512, 3, activation='relu', padding=1))
            self.global_net.add(nn.Conv2D(512, 3, activation='relu', padding=1))
            self.global_net.add(nn.Conv2D(512, 3, activation='relu', padding=1))
            self.global_net.add(nn.MaxPool2D(pool_size=2))
            self.global_net.add(nn.Flatten())
            self.global_net.add(nn.Dense(4096, activation='relu'))

    def forward(self, x):
        # print(x)
        x = self.base_net(x)
        p = self.pose_net(x)
        g = self.global_net(x)
        loc = self.loc(p)
        vis1 = self.vis1(p)
        vis2 = self.vis2(p)
        vis3 = self.vis3(p)
        vis4 = self.vis4(p)
        vis5 = self.vis5(p)
        vis6 = self.vis6(p)
        vis7 = self.vis7(p)
        vis8 = self.vis8(p)

        classifier = self.classifier(g)

        return loc, (vis1, vis2, vis3, vis4, vis5, vis6, vis7, vis8), classifier, g, p


if __name__ == '__main__':
    ctx = mx.cpu()
    net = FashionNet()
    print(net)
    net.initialize(ctx=ctx)
    batch_size = 10
    X = nd.random_normal(scale=0.2, shape=(batch_size, 3, 224, 224), ctx=ctx)
    loss_softmax = gluon.loss.SoftmaxCrossEntropyLoss()
    loss_l2 = gluon.loss.L2Loss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
    # loc, vis1, vis2, vis3, vis4, g_fc = net(X)
    # print(loc.shape)
    # print(vis1.shape)
    # print(vis2.shape)
    # print(vis3.shape)
    # print(vis4.shape)
    # print(g_fc.shape)

    loc_data = nd.random_normal(scale=0.2, shape=(batch_size, 16), ctx=ctx)
    vis1_data = nd.array([1] * batch_size, ctx=ctx)
    vis2_data = nd.array([1] * batch_size, ctx=ctx)
    vis3_data = nd.array([1] * batch_size, ctx=ctx)
    vis4_data = nd.array([1] * batch_size, ctx=ctx)
    vis5_data = nd.array([1] * batch_size, ctx=ctx)
    vis6_data = nd.array([1] * batch_size, ctx=ctx)
    vis7_data = nd.array([1] * batch_size, ctx=ctx)
    vis8_data = nd.array([1] * batch_size, ctx=ctx)
    label_data = nd.random_normal(scale=0.2, shape=(batch_size, 1000), ctx=ctx)
    vis_data = [vis1_data, vis2_data, vis3_data, vis4_data, vis5_data, vis6_data,
                vis7_data, vis8_data]
    with autograd.record():
        l, vs, classifier_output, _, _ = net(X)

        loss = loss_l2(l, loc_data) + loss_l2(classifier_output, label_data)
        for v, d in zip(vs, vis_data):
            loss = loss + loss_softmax(v, d)

    loss.backward()
    trainer.step(batch_size)

    print(loss)
    import time

    time.sleep(20)
