from data.dataset import BrainMRIDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader


def create_dataloaders(args, transforms):

    dataset_full = BrainMRIDataset(root_dir=args.root_dir, transforms=transforms)
    test_samples = dataset_full.get_positive_testdf()

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
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, num_workers=2, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, test_samples
