import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


def show_aug(inputs, n_rows=5, n_cols=5, image=True):
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(wspace=0., hspace=0.)
    i_ = 0

    if len(inputs) > 25:
        inputs = inputs[:25]

    for idx in range(len(inputs)):

        # normalization
        if image is True:
            img = inputs[idx].numpy().transpose(1, 2, 0)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            img = (img * std + mean).astype(np.float32)
        else:
            img = inputs[idx].numpy().astype(np.float32)
            img = img[0, :, :]

        # plot
        plt.subplot(n_rows, n_cols, i_+1)
        plt.imshow(img)
        plt.axis('off')

        i_ += 1

    return plt.show()


def dice_coef_loss(inputs, target):
    smooth = 1.0
    intersection = 2.0 * ((target * inputs).sum()) + smooth
    union = target.sum() + inputs.sum() + smooth

    return 1 - (intersection / union)


def bce_dice_loss(inputs, target):
    dice_loss = dice_coef_loss(inputs, target)
    bce_score = nn.BCELoss()
    bce_loss = bce_score(inputs, target)

    return bce_loss + dice_loss


def dice_coef_metric(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union
