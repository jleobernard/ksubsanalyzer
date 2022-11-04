import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SegmentationSubsDataset(Dataset):

    def __init__(self, df, augmentations):
        self.df = df
        self.augmentations = augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row.filename.absolute()

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if row.subs == 1:
            mask_path = get_mask_name(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = np.expand_dims(mask, axis=-1)
        else:
            mask = np.zeros((3, TARGET_HEIGHT, TARGET_WIDTH))

        if self.augmentations:
            data = self.augmentations(image=image, mask=mask)
            image, mask = data['image'], data['mask']

        # Change from (h, w, c) -> (c, h, w)

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

        image = torch.Tensor(image) / 255.0
        mask = torch.round(torch.Tensor(mask) / 255.0)

        return image, mask, row.subs