import mxnet as mx
import re
from mxnet import nd
import numpy as np
import cv2
import random

random.seed(123)


def get_attr_label_index(attr):
    a = re.split(r'\s+', attr.strip())
    a = np.array([float(x.strip()) for x in a])
    # return np.where(a > 0)[0]
    return a


def padding_img(img):
    """
    padding image
    :param img: (height, width, channel)
    """
    height, width, channels = img.shape

    padding_size_1 = abs(height - width) // 2
    padding_size_2 = abs(height - width) - padding_size_1

    if padding_size_1 == 0:
        img_out = img
    else:
        # padding width axis
        if height > width:
            # must use np.uint8 type
            padding1 = np.zeros(shape=(height, padding_size_1, channels), dtype=np.uint8)
            padding2 = np.zeros(shape=(height, padding_size_2, channels), dtype=np.uint8)
            img_out = np.concatenate([padding1, img, padding2], axis=1)
        # padding height axis
        elif height < width:
            padding1 = np.zeros(shape=(padding_size_1, width, channels), dtype=np.uint8)
            padding2 = np.zeros(shape=(padding_size_2, width, channels), dtype=np.uint8)
            img_out = np.concatenate([padding1, img, padding2], axis=0)
        else:
            img_out = img

    return img_out


def transform_landmarks(l, height, width, resize=(224, 224)):
    """
    transform the landmarks while resizing the image
    :param l: input landmarks shape (num_landmarks, 2)
    :param height: original image height
    :param width: original image width
    :param resize: the image must resize to (224, 224) for training
    """
    re_height, re_width = resize
    padding_size_1 = abs(height - width) // 2
    # print(padding_size_1)
    scale = re_height / max(height, width)

    # print('heigth', height)
    # print(l)
    # moving up along width
    if height > width:
        out = np.array([((w + padding_size_1 if w > 0 else 0) * scale, h * scale) for w, h in l])
    else:
        out = np.array([(w * scale, (h + padding_size_1 if h > 0 else 0) * scale) for w, h in l])
    return out


# def inv_transform_landmarks(l, height, width, size=(224, 224)):
#     """
#     transform the landmarks while resizing the image
#     :param l: input landmarks shape of (num_landmarks, 2)
#     :param height:
#     :param width:
#     :param size: the size of image for training and evaluating
#     """
#     re_height, re_width = size
#     padding_size_1 = abs(height - width) // 2
#     height_scale = re_height / height
#     width_scale = re_width / width


def get_vis_landmarks(vis_landmarks, num_landmarks=8):
    """

    :param vis_landmarks:
    :param num_landmarks:
    :return:
    """
    landmarks = [int(x) for x in re.split(r'\s+', vis_landmarks.strip())]
    n = len(landmarks) // 3
    diff = num_landmarks - n
    if diff > 0:
        for _ in range(diff):
            for _ in range(3):
                landmarks.append(0)

    landmarks_tmp = np.array([landmarks[i] for i in range(len(landmarks)) if i % 3 != 0])
    vis = np.array([landmarks[i] for i in range(len(landmarks)) if i % 3 == 0])
    landmarks = landmarks_tmp.reshape((len(landmarks_tmp) // 2, 2))
    return vis, landmarks


def read_img(img_file, resize=(224, 224)):
    """
    read image and padding then resize
    :param img_file:
    :param resize:
    :return:
    """
    img = cv2.imread(img_file, 1)
    # print(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channels = img.shape
    img = padding_img(img)
    img = cv2.resize(img, resize)
    # print(img.shape)
    return img.reshape((3, resize[0], resize[1])), height, width


def read_img_attr_file(attr_file):
    fd = open(attr_file)
    out = []
    for i, line in enumerate(fd):
        line = line.strip()
        if not line:
            continue
        if i == 0:
            continue
        elif i == 1:
            continue
        else:
            img_file, attr = re.split(r'\s+', line, 1)
            attr = get_attr_label_index(attr)
            out.append([img_file, attr])
    return out


def get_length(attr_file):
    """
    get number of all images
    :param attr_file:
    :return:
    """
    fd = open(attr_file)
    for i, line in enumerate(fd):
        line = line.strip()
        if not line:
            continue
        if i == 0:
            length = int(line)
            return length


def read_landmark_file(landmark_file, num_landmarks=8):
    """
    prepare landmark label data for training
    :param landmark_file:
    :param num_landmarks:
    """
    fd = open(landmark_file)
    out = []
    for i, line in enumerate(fd):
        line = line.strip()
        if not line:
            continue
        if i == 0:
            continue
        elif i == 1:
            continue
        else:
            img_file, _, landmarks = re.split(r'\s+', line, 2)
            vis, landmarks = get_vis_landmarks(landmarks, num_landmarks)
            out.append([img_file, vis, landmarks])
    return out


# todo
class ImageDataIter(mx.io.DataIter):
    def __init__(self, img_dir, img_attr_file=None, landmark_file=None, ctx=mx.cpu(), label_file=None, num_landmarks=8,
                 batch_size=32, resize=(224, 224)):
        super(ImageDataIter, self).__init__()
        self.landmark_file = landmark_file
        self.img_dir = img_dir
        self.img_attr_file = img_attr_file
        self.label_file = label_file
        self.batch_size = batch_size
        self.num_landmarks = num_landmarks
        self.cur_batch = 0
        self.resize = resize
        self.img_attr_list = read_img_attr_file(self.img_attr_file)
        self.img_landmark_list = read_landmark_file(self.landmark_file, num_landmarks=self.num_landmarks)
        self.length = get_length(self.landmark_file)
        self.ctx = ctx
        self.order = list(range(self.length))  # used for shuffle
        self.order_iter = self.order_gen()

    def __len__(self):
        return get_length(self.landmark_file)

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    def order_gen(self):
        # shuffle the dataset
        random.shuffle(self.order)
        for i in self.order:
            yield i

    @property
    def provide_data(self):
        return

    @property
    def provide_label(self):
        return

    def _get_data(self):
        data = []
        landmarks = []
        labels = []
        vis_labels = []
        img_files = []

        if self.cur_batch * self.batch_size + self.batch_size <= self.length:
            for i, order in zip(range(self.batch_size), self.order_iter):
                attr, vis_landmark = self.img_attr_list[order], self.img_landmark_list[order]
                img, height, width = read_img(self.img_dir + attr[0])
                label = attr[1]
                vis = vis_landmark[1]
                landmark = transform_landmarks(vis_landmark[2], height, width, self.resize).flatten()
                data.append(img)
                labels.append(label)
                landmarks.append(landmark)
                vis_labels.append(vis)
                img_files.append(attr[0])
            self.cur_batch += 1
        else:
            return None, None, None, None, None
        return np.stack(data), np.stack(labels), np.stack(vis_labels), np.stack(landmarks), img_files

    # todo write data loader for data iter
    # fixme this is not the right style
    def next(self):

        data, labels, vis, landmarks, img_files = self._get_data()
        if data is not None:
            data = [nd.array(data, ctx=self.ctx), img_files]
            label = [nd.array(labels, ctx=self.ctx), nd.array(vis, ctx=self.ctx), nd.array(landmarks, self.ctx)]
            return data, label
        else:
            return None, None


if __name__ == '__main__':
    # landmarks_test = '0 071 067  1 102 063  0 049 069  1 117 070  0 062 180  0 095 185'
    # print(get_vis_landmarks(landmarks_test))
    # read_img_attr_file('/data/image_data/category_and_attribute_prediction_benchmark/Anno/list_attr_img.txt')
    # data_iter = read_landmark_file(
    #     '/data/image_data/category_and_attribute_prediction_benchmark/Anno/list_landmarks.txt')
    # for i, (vis, land) in enumerate(data_iter):
    #     if i > 2:
    #         break
    #     print(vis)
    #     print(land)
    img_dir_test = '/data/image_data/category_and_attribute_prediction_benchmark/Img/'
    img_attr_file_test = '/data/image_data/category_and_attribute_prediction_benchmark/Anno/list_attr_img.txt'
    landmark_file_test = '/data/image_data/category_and_attribute_prediction_benchmark/Anno/list_landmarks.txt'

    data_iter = ImageDataIter(img_dir_test, img_attr_file_test, landmark_file_test, mx.cpu())
    for _ in range(5):
        print(data_iter.next())
