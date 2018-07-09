from model import FashionNet
import sys
import mxnet as mx
from mxnet import nd
from data_loader import read_img
from utils import show_img
import numpy as np


if __name__ == '__main__':
    model_file = sys.argv[1]
    img_file = sys.argv[2]

    net = FashionNet()
    ctx = mx.cpu()
    net.collect_params().load(model_file, ctx=ctx)

    img, height, width = read_img(img_file)

    img = nd.array(img).reshape([1, 3, 224, 224])

    l, vs, classifier_output, _, _ = net(img)
    out = classifier_output[0].asnumpy()

    print(np.where(out > 0))

    landmarks = l[0].asnumpy()
    landmarks = landmarks.reshape([len(landmarks) // 2, 2])
    print(landmarks)
    show_img(img_file, landmarks)
