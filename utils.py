# from mxnet import nd
import matplotlib.pyplot as plt
import numpy as np
from data_loader import read_img   # , transform_landmarks


def attribute_accuracy(pred, label):
    """

    :param pred: shape (batch_size, num_attributes)
    :param label: shape (batch_size, num_attributes)
    """
    pass


def vis_accuracy(pred, label):
    """

    :param pred:
    :param label:
    """
    pass


# use MSE to calculate the loss
def landmark_loss(pred, label):
    """

    :param pred:
    :param label:
    """
    pass


def box_to_rect(box, color, linewidth=2):
    """convert an anchor box to a matplotlib rectangle"""
    # box = box.asnumpy()
    box = np.array(box)
    return plt.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
        fill=False, edgecolor=color, linewidth=linewidth)


def show_img(img_file, landmarks=np.array([]), landmarks_true=np.array([])):
    img, height, width = read_img(img_file)
    print(height, width)
    img = img.reshape(224, 224, 3)
    # landmarks_true = transform_landmarks(landmarks_true, height, width)

    # print(img.shape)
    box_size = 3
    boxes = [(x - box_size, y - box_size, x + box_size, y + box_size)
             for x, y in landmarks.tolist()]

    boxes_true = [(x - box_size, y - box_size, x + box_size, y + box_size)
                  for x, y in landmarks_true.tolist()]
    # print(boxes)
    fig = plt.subplot()
    for b in boxes:
        fig.add_patch(box_to_rect(b, 'RED'))

    for b in boxes_true:
        fig.add_patch(box_to_rect(b, 'GREEN'))

    fig.imshow(img)
    plt.show()
