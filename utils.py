import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import os
from collections import OrderedDict
from PIL import Image
from glob import glob
from os.path import join


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
    smooth = 1.0
    intersection = (2.0 * (target * inputs).sum()) + smooth
    union = target.sum() + inputs.sum() + smooth

    return intersection / union


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def remove_redundant_keys(state_dict: OrderedDict):
    # remove DataParallel wrapping
    if "module" in list(state_dict.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
    else:
        new_state_dict = state_dict

    return new_state_dict


def save_checkpoint(path, model, epoch, optimizer=None, save_arch=False, params=None):
    attributes = {"epoch": epoch, "state_dict": remove_redundant_keys(model.state_dict())}
    if optimizer is not None:
        attributes["optimizer"] = optimizer.state_dict()
    if save_arch:
        attributes["arch"] = model
    if params is not None:
        attributes["params"] = params

    try:
        torch.save(attributes, path)
    except TypeError:
        if "arch" in attributes:
            print(
                "Model architecture will be ignored because the architecture includes non-pickable objects."
            )
            del attributes["arch"]
            torch.save(attributes, path)


def load_checkpoint(path, model, optimizer=None, params=False, epoch=False):
    resume = torch.load(path)
    rets = dict()

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(remove_redundant_keys(resume["state_dict"]))
    else:
        model.load_state_dict(remove_redundant_keys(resume["state_dict"]))

        rets["model"] = model

    if optimizer is not None:
        optimizer.load_state_dict(resume["optimizer"])
        rets["optimizer"] = optimizer
    if params:
        rets["params"] = resume["params"]
    if epoch:
        rets["epoch"] = resume["epoch"]

    return rets


def plot_plate_overlap(batch_preds, title, num):
    save_path = "./output/gif_images/"
    create_folder(save_path)
    plt.figure(figsize=(15, 15))
    plt.imshow(batch_preds)
    plt.axis("off")

    plt.figtext(0.76, 0.75, "Green - Ground Truth", va="center", ha="center", size=20,color="lime")
    plt.figtext(0.26, 0.75, "Red - Prediction", va="center", ha="center", size=20, color="#ff0d00")
    plt.suptitle(title, y=.80, fontsize=20, weight="bold", color="#00FFDE")

    fn = "_".join((title+str(num)).lower().split()) + ".png"
    plt.savefig(save_path + fn, bbox_inches='tight', pad_inches=0.2, transparent=False, facecolor='black')
    plt.close()


def make_gif(title):
    folder_path = "./output/gif_images/"
    base_name = "_".join(title.lower().split())
    file_path = join(folder_path,base_name)

    base_len = len(file_path)
    end_len = len(".png")
    fp_in = f"{file_path}*.png"
    fp_out = f"{file_path}.gif"

    img, *imgs = [Image.open(f)
                  for f in sorted(glob.glob(fp_in),
                                  key=lambda x: int(x[base_len:-end_len]))]

    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=1000, loop=0)

    return fp_out