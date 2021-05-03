import torch
from data.dataset import BrainMRIDataset, image_transforms
from torch.utils.data import random_split

from utils import show_aug
from models.unet import UNet
from models.ResNeXtUNet import ResNeXtUNet
import argparse
from scripts.train import train_model
from utils import bce_dice_loss, plot_plate_overlap
import os
from scripts.test import evaluate, predict_single, batch_preds_overlap, create_gif
from data.dataloader import create_dataloaders
import numpy as np


parser = argparse.ArgumentParser(description="Brain MRI Segmentation")
parser.add_argument('--view_aug', type=bool, default=False, help='Visualize data augmentations')
parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs to train on')
parser.add_argument('--train', default=False, type=bool, help='Train the model')
parser.add_argument('--test', default=False, type=bool, help='Test the model')
parser.add_argument("--image_path",
                    default='/home/nishita/datasets/brain_mri/kaggle_3m/TCGA_HT_7881_19981015/TCGA_HT_7881_19981015_30.tif',
                    type=str, help="Path for single image prediction (inference)")
parser.add_argument('--gpus', default="0", type=str, help='Which GPU to use?')
parser.add_argument('--root_dir', default='/home/nishita/datasets/brain_mri/kaggle_3m/', type=str, help='Path of dataset')
parser.add_argument("--run_name", default=0, type=str, help="Name of experiment/run")
parser.add_argument("--model", default='unet', type=str, help="Choose model from 'unet' or 'resnext'")
parser.add_argument("--batch_size", default=32, type=int, help="batch-size to use")
parser.add_argument("--lr_scheduler", default=False, type=bool, help='lr scheduler')

args = parser.parse_args()
print(f'The arguments are {vars(args)}')
torch.manual_seed(0)

# Check for GPUs
if torch.cuda.is_available():
    os.environ["CUDA_VISION_DEVICES"] = str(args.gpus)
    device = torch.device("cuda:" + str(args.gpus))
    print("CUDA GPU {} found.".format(args.gpus))
else:
    device = torch.device("cpu")
    print("No CUDA device found. Using CPU instead.")


train_dataloader, val_dataloader, test_dataloader, test_samples = create_dataloaders(args, transforms=image_transforms)


# Visualize data augmentations
if args.view_aug:
    images, masks = next(iter(train_dataloader))
    print(images.shape, masks.shape)
    show_aug(images)
    show_aug(masks, image=False)


if str(args.model).lower() == 'unet':
    model = UNet(n_classes=1).to(device)
    create_gif(args, model=model, dataloader=test_dataloader, device=device, threshold=0.5)

elif str(args.model).lower() == 'resnext':
    model = ResNeXtUNet(n_classes=1).to(device)

# Training
if args.train:
    optimizer = torch.optim.Adamax(model.parameters(), lr=5e-4)
    train_model(args, model=model, train_loader=train_dataloader, val_loader=val_dataloader, loss=bce_dice_loss,
                optimizer=optimizer, device=device)

# Testing
if args.test:
    test_dice_unet = evaluate(args, model=model, test_loader=test_dataloader, device=device)
    print(f"""Mean dice of the test images - {np.around(test_dice_unet, 4) * 100}%""")

# Prediction of single image
# if args.image_path is not None:
#    predict_single(args, model=model, device=device)


batch_preds_overlap(args, model=model, samples=test_samples, device=device)


