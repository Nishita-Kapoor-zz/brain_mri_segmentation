import torch
from dataset import BrainMRIDataset, image_transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from utils import show_aug


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
root_dir = '/home/nishita/datasets/brain_mri/kaggle_3m/'

# train-test split
dataset_full = BrainMRIDataset(root_dir=root_dir, transforms=image_transforms)

train_size = int(0.8 * len(dataset_full))
temp_size = len(dataset_full) - train_size
val_size = int(0.5 * temp_size)
test_size = temp_size - val_size

train_set, temp_set = random_split(dataset_full, [train_size, temp_size])
val_set, test_set = random_split(temp_set, [val_size, test_size])

train_dataloader = DataLoader(train_set, batch_size=32, num_workers=4, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=32, num_workers=4, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=32, num_workers=4, shuffle=True)

# Visualize augmentations
images, masks = next(iter(train_dataloader))
print(images.shape, masks.shape)
show_aug(images)
show_aug(masks, image=False)


unet_optimizer = torch.optim.Adamax(unet.parameters(), lr=1e-3)
fpn_optimizer = torch.optim.Adamax(fpn.parameters(), lr=1e-3)
rx50_optimizer = torch.optim.Adam(rx50.parameters(), lr=5e-4)


# lr_scheduler
def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

