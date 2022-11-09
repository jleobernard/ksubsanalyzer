import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

from analyzer.constants import IMAGE_SIZE


class EvalSegmentationSubsDataset(Dataset):

    def __init__(self, df, nb_transforms_per_image: int = 1):
        self.df = df
        self.nb_transforms_per_image = nb_transforms_per_image
        self.augmentations = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.Resize(IMAGE_SIZE, IMAGE_SIZE)
        ])

    def __len__(self):
        return len(self.df) * self.nb_transforms_per_image

    def __getitem__(self, idx):
        row = self.df.iloc[idx // self.nb_transforms_per_image]
        image = cv2.imread(row.filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.augmentations(image=image)["image"]
        # Change from (h, w, c) -> (c, h, w)

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        image = torch.Tensor(image) / 255.0

        return image