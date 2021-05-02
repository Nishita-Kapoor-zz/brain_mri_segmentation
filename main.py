import torch
from dataset import BrainMRIDataset, image_transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from utils import show_aug
from models.unet import UNet
from models.ResNeXtUNet import ResNeXtUNet
import argparse
from train import train_model
from utils import bce_dice_loss, plot_plate_overlap
import os
from test import evaluate, predict_single, batch_preds_overlap
import numpy as np


parser = argparse.ArgumentParser(description="Brain MRI Segmentation")
parser.add_argument('--view_aug', type=bool, default=False, help='Visualize data augmentations')
parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs to train on')
parser.add_argument('--train', default=False, type=bool, help='Train the model')
parser.add_argument('--test', default=True, type=bool, help='Test the model')
parser.add_argument("--image_path", default=None, type=str, help="Path for single image prediction (inference)")
parser.add_argument('--gpus', default="0", type=str, help='Which GPU to use?')
parser.add_argument('--root_dir', default='/home/nishita/datasets/brain_mri/kaggle_3m/', type=str, help='Path of dataset')
parser.add_argument("--run_name", default=0, type=str, help="Name of experiment/run")
parser.add_argument("--model", default='unet', type=str, help="Choose model from 'unet' or 'resnext'")
parser.add_argument("--batch_size", default=32, type=int, help="batch-size to use")
parser.add_argument("--lr_scheduler", default=False, type=bool, help='lr scheduler')

args = parser.parse_args()
print(f'The arguments are {vars(args)}')

# Check for GPUs
if torch.cuda.is_available():
    os.environ["CUDA_VISION_DEVICES"] = str(args.gpus)
    device = torch.device("cuda:" + str(args.gpus))
    print("CUDA GPU {} found.".format(args.gpus))
else:
    device = torch.device("cpu")
    print("No CUDA device found. Using CPU instead.")


dataset_full = BrainMRIDataset(root_dir=args.root_dir, transforms=image_transforms)

# train-test split
train_size = int(0.8 * len(dataset_full))
temp_size = len(dataset_full) - train_size
val_size = int(0.5 * temp_size)
test_size = temp_size - val_size

train_set, temp_set = random_split(dataset_full, [train_size, temp_size])
val_set, test_set = random_split(temp_set, [val_size, test_size])

# Dataloaders
train_dataloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=2, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=args.batch_size, num_workers=2, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=args.batch_size, num_workers=2, shuffle=True)

# Visualize data augmentations
if args.view_aug:
    images, masks = next(iter(train_dataloader))
    print(images.shape, masks.shape)
    show_aug(images)
    show_aug(masks, image=False)


if str(args.model).lower() == 'unet':

    unet = UNet(n_classes=1).to(device)

    if args.train:
        unet_optimizer = torch.optim.Adamax(unet.parameters(), lr=5e-4)
        train_model(args, model_name="Vanila_UNet", model=unet, train_loader=train_dataloader,
                    val_loader=val_dataloader, loss=bce_dice_loss,
                    optimizer=unet_optimizer, device=device)

    if args.test:
        test_dice_unet = evaluate(args, model=unet, test_loader=test_dataloader, device=device)
        print(f"""Vanilla UNet\nMean dice of the test images - {np.around(test_dice_unet, 2) * 100}%""")

    if args.image_path is not None:
        pass
'''
    prediction_overlap_u = batch_preds_overlap(unet, test_samples)
    pred_overlap_5x1_u = []
    pred_overlap_5x3_u = []
    for i in range(5, 105 + 5, 5):
        pred_overlap_5x1_u.append(np.hstack(np.array(prediction_overlap_u[i - 5:i])))
    for i in range(3, 21 + 3, 3):
        pred_overlap_5x3_u.append(np.vstack(pred_overlap_5x1_u[i - 3:i]))
    title1 = "Predictions of Vanilla UNet"
    for num, batch in enumerate(pred_overlap_5x3_u):
        plot_plate_overlap(batch, title1, num)
'''

if str(args.model).lower() == 'resnext':
    rx50 = ResNeXtUNet(n_classes=1).to(device)

    if args.train:
        rx50_optimizer = torch.optim.Adam(rx50.parameters(), lr=5e-4)
        train_model(args, model_name="ResNeXt50", model=rx50, train_loader=train_dataloader,
                    val_loader=val_dataloader, loss=bce_dice_loss,
                    optimizer=rx50_optimizer, device=device)
    if args.test:
        test_dice_resnext = evaluate(args, model=rx50, test_loader=test_dataloader, device=device)
        print(f"""ResNext50\nMean dice of the test images - {np.around(test_dice_resnext, 2) * 100}%""")

    if args.image_path is not None:
        pass
'''
    prediction_overlap_r = batch_preds_overlap(rx50, test_samples)
    pred_overlap_5x1_r = []
    pred_overlap_5x3_r = []
    for i in range(5, 105 + 5, 5):
        pred_overlap_5x1_r.append(np.hstack(np.array(prediction_overlap_r[i - 5:i])))
    for i in range(3, 21 + 3, 3):
        pred_overlap_5x3_r.append(np.vstack(pred_overlap_5x1_r[i - 3:i]))

    title3 = "Predictions of UNet with ResNeXt50 backbone"
    for num, batch in enumerate(pred_overlap_5x3_r):
        plot_plate_overlap(batch, title3, num)
'''

