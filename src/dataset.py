import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class DIV2KDataset(Dataset):

    def __init__(
        self,
        hr_dir,
        images,
        patch_size=192,
        scale=4,
        train=True
    ):
        self.hr_dir = hr_dir
        self.images = images
        self.patch_size = patch_size
        self.scale = scale
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_path = os.path.join(
            self.hr_dir,
            self.images[idx]
        )

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        # -----------------------------
        # Training: random crop
        # Validation: center crop
        # -----------------------------

        if self.train:

            x = random.randint(
                0,
                width - self.patch_size
            )

            y = random.randint(
                0,
                height - self.patch_size
            )

        else:

            x = (width - self.patch_size) // 2
            y = (height - self.patch_size) // 2

        hr = image.crop((
            x,
            y,
            x + self.patch_size,
            y + self.patch_size
        ))

        # -----------------------------
        # Training augmentation
        # -----------------------------

        if self.train:

            if random.random() < 0.5:
                hr = TF.hflip(hr)

            if random.random() < 0.5:
                hr = TF.vflip(hr)

        # -----------------------------
        # Create LR
        # -----------------------------

        lr = hr.resize(
            (
                self.patch_size // self.scale,
                self.patch_size // self.scale
            ),
            Image.Resampling.BICUBIC
        )

        # -----------------------------
        # Convert to tensors
        # -----------------------------

        lr = TF.to_tensor(lr)
        hr = TF.to_tensor(hr)

        return lr, hr