from model import FashionNet
import sys
import mxnet as mx
from mxnet import nd
from data_loader import read_img
# from utils import show_img
import numpy as np
import base64


def get_feat(d, threshold=0.5):
    return np.where(d > threshold, 1, 0)


def extract_feat(model, image_file):
    img, height, width = read_img(image_file)

    img = nd.array(img).reshape([1, 3, 224, 224])

    l, vs, classifier_output, g, p = model(img)

    feat = get_feat(g[0].asnumpy(), 0.5)
    feat2 = get_feat(p[0].asnumpy(), 0.5)
    # print('feat: ', feat)
    feat = base64.b64encode(
        int(''.join([str(k) for k in feat.tolist()]), base=2).to_bytes(len(feat) // 8, byteorder='big')).decode()
    feat2 = base64.b64encode(
        int(''.join([str(k) for k in feat2.tolist()]), base=2).to_bytes(len(feat2) // 8, byteorder='big')).decode()
    return feat, feat2


if __name__ == '__main__':
    model_file = sys.argv[1]
    img_file = sys.argv[2]

    net = FashionNet()
    ctx = mx.cpu()
    net.collect_params().load(model_file, ctx=ctx)
    f, f2 = extract_feat(net, img_file)
    out = '\t'.join([img_file, f, f2])

