from extract_feature import extract_feat
import sys
import glob
from model import FashionNet
import mxnet as mx

VERBOSE = False


if __name__ == '__main__':
    model_file = sys.argv[1]
    img_dir = sys.argv[2]

    net = FashionNet()
    ctx = mx.cpu()
    net.collect_params().load(model_file, ctx=ctx)

    img_files = glob.glob(img_dir + '/*')

    for img_file in img_files:
        try:
            f1, f2 = extract_feat(net, img_file)
            out = '\t'.join([img_file, f1, f2])
            print(out)
        except Exception as e:
            if VERBOSE:
                print(str(e))
