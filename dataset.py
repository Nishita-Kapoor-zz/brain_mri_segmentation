from torch.utils.data import Dataset, DataLoader
import glob
import os
import pandas as pd
import numpy as np
import cv2


class BrainMRIDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.df = self.create_dataframe()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = cv2.imread(self.df.iloc[index, 1])
        mask = cv2.imread(self.df.iloc[index, 2], 0)

        augmented = self.transforms(image=image,
                                    mask=mask)
        if self.transforms is not None:
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

    def create_dataframe(self):

        BASE_LEN = len(self.root_dir) + 44   #e.g.: len(self.root_dir) + len('TCGA_DU_6404_19850629/TCGA_DU_6404_19850629_')
        END_IMG_LEN = 4       # len('.tif')
        END_MASK_LEN = 9      # len('_mask.tif')

        data_map = []
        for sub_dir_path in glob.glob(self.root_dir + "*"):
            if os.path.isdir(sub_dir_path):
                dirname = sub_dir_path.split("/")[-1]
                for filename in os.listdir(sub_dir_path):
                    image_path = sub_dir_path + "/" + filename
                    data_map.extend([dirname, image_path])
            else:
                print("This is not a dir:", sub_dir_path)

        df = pd.DataFrame({"dirname": data_map[::2],
                           "path": data_map[1::2]})

        # Masks/Not masks
        df_imgs = df[~df['path'].str.contains("mask")]
        df_masks = df[df['path'].str.contains("mask")]

        # Data sorting
        imgs = sorted(df_imgs["path"].values, key=lambda x: int(x[BASE_LEN:-END_IMG_LEN]))
        masks = sorted(df_masks["path"].values, key=lambda x: int(x[BASE_LEN:-END_MASK_LEN]))

        df = pd.DataFrame({"patient": df_imgs.dirname.values,
                           "image_path": imgs,
                           "mask_path": masks})

        # Adding A/B column for diagnosis
        def positiv_negativ_diagnosis(mask_path):
            value = np.max(cv2.imread(mask_path))
            if value > 0:
                return 1
            else:
                return 0

        df["mask"] = df["mask_path"].apply(lambda m: positiv_negativ_diagnosis(m))

        return df
