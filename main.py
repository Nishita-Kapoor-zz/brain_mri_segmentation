import torch
from dataset import BrainMRIDataset, image_transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from utils import show_aug
from models.unet import UNet
from models.ResNeXtUNet import ResNeXtUNet
import argparse
from train import train_model
from utils import bce_dice_loss
import os
from test import evaluate, predict


parser = argparse.ArgumentParser(description="Brain MRI Segmentation")
parser.add_argument('--view_aug', type=bool, default=False, help='Visualize data augmentations')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train on')
parser.add_argument('--train', default=True, type=bool, help='Train the model')
parser.add_argument('--test', default=False, type=bool, help='Test the model')
parser.add_argument("--image_path", default=None, type=str, help="Path for single image prediction (inference)")
parser.add_argument('--gpus', default="0", type=str, help='Which GPU to use?')
parser.add_argument('--root_dir', default='/home/nishita/datasets/brain_mri/kaggle_3m/', type=str, help='Path of dataset')
parser.add_argument("--run_name", default=0, type=str, help="Name of experiment/run")
parser.add_argument("--model", default='unet', type=str, help="Choose model from 'unet or resnext")
parser.add_argument("--batch_size", default=32, type=int, help="batch-size to use")
parser.add_argument("--lr_scheduler", efault=False, type=bool, help='lr scheduler')

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
train_dataloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=args.batch_size, num_workers=4, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=args.batch_size, num_workers=4, shuffle=True)

# Visualize data augmentations
if args.view_aug:
    images, masks = next(iter(train_dataloader))
    print(images.shape, masks.shape)
    show_aug(images)
    show_aug(masks, image=False)

# Models
unet = UNet(n_classes=1).to(device)
rx50 = ResNeXtUNet(n_classes=1).to(device)

# Optimizers
unet_optimizer = torch.optim.Adamax(unet.parameters(), lr=1e-3)
rx50_optimizer = torch.optim.Adam(rx50.parameters(), lr=5e-4)

if args.train:
    if str(args.model).lower() == 'unet':
        train_model(args, model_name="Vanila_UNet", model=unet, train_loader=train_dataloader,
                                                val_loader=val_dataloader, train_loss=bce_dice_loss,
                                                optimizer=unet_optimizer, device=device)
    elif str(args.model).lower() == 'resnext':
        train_model(model_name="ResNeXt50", model=rx50, train_loader=train_dataloader,
                                                val_loader=val_dataloader, train_loss=bce_dice_loss,
                                                optimizer=rx50_optimizer, device=device)


if args.test:
    test_iou = evaluate(unet, test_dataloader)
    print(f"""Vanilla UNet\nMean IoU of the test images - {np.around(test_iou, 2)*100}%""")

    test_iou = evaluate(rx50, test_dataloader)
    print(f"""ResNext50\nMean IoU of the test images - {np.around(test_iou, 2) * 100}%""")
