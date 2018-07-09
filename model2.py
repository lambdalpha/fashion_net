from mxnet import gluon
import mxnet.gluon.nn as nn
import mxnet as mx
from mxnet import nd
from mxnet import autograd
# from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon.model_zoo.vision.resnet import ResNetV1, ResNetV2
from mxnet.gluon.model_zoo.vision.resnet import resnet_spec, resnet_block_versions


NUM_CLASSES = 1000


class MyResNetV1(ResNetV1):
    def __init__(self, block, layers, channels, classes=1000, thumbnail=False, **kwargs):
        super(MyResNetV1, self).__init__(block, layers, channels, classes, thumbnail, **kwargs)

    def hybrid_forward(self, F, x):
        for fea in self.features[:-2]:
            x = fea(x)
        return x


class MyResNetV2(ResNetV2):
    def __init__(self, block, layers, channels, classes=1000, thumbnail=False, **kwargs):
        super(MyResNetV2, self).__init__(block, layers, channels, classes, thumbnail, **kwargs)

    def hybrid_forward(self, F, x):
        for fea in self.features[:-2]:
            x = fea(x)
        return x


MYRESNET_CLASSES = [MyResNetV1, MyResNetV2]


def get_resnet(num_layers, version=2, **kwargs):
    block_type, layers, channels = resnet_spec[num_layers]
    block_class = resnet_block_versions[version-1][block_type]
    myresnet_class = MYRESNET_CLASSES[version - 1]
    net = myresnet_class(block_class, layers, channels, **kwargs)
    return net


# fixme add batch norm
# todo use pre-trained resnet as base net
class FashionNet2(nn.Block):
    def __init__(self, resnet_layers=50, resnet_versoin=2, **kwargs):
        super(FashionNet2, self).__init__(**kwargs)
        self.base_net = get_resnet(resnet_layers, resnet_versoin)
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

    def initialize_basenet(self, filename, ctx):
        self.base_net.load_params(filename, ctx, ignore_extra=False)

    def forward(self, x):
        # print(x)
        x = self.base_net(x)
        # print(x.shape)
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
    net = FashionNet2()
    print(net)
    net.initialize(ctx=ctx)
    net.initialize_basenet('/home/alpha/.mxnet/models/resnet50_v2-eb7a3687.params', ctx=ctx)

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
    trainer.step(batch_size, ignore_stale_grad=True)

    print(loss)
    import time

    time.sleep(20)
