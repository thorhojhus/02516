import os
import random
from glob import glob

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F
from torchvision import transforms as T

# cache splits to keep them identical across runs
_PH2_SPLITS = None
_DRIVE_SPLITS = None

def _seeded_split(items, train_ratio=0.8, val_ratio=0.1):
    items = list(items)
    random.shuffle(items)
    n_total = len(items)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]
    return {
        'train': sorted(train_items),
        'val': sorted(val_items),
        'test': sorted(test_items)
    }


def _binarize_mask(mask):
    mask = np.array(mask)
    mask = (mask > 0).astype(np.float32)
    mask = torch.from_numpy(mask).unsqueeze(0)
    return mask


def _default_transform(image, mask):
    image = T.ToTensor()(image)
    mask = _binarize_mask(mask)
    return image, mask


class PH2Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir='/dtu/datasets1/02516/PH2_Dataset_images',
        split='train',
        transform=None,
        image_size=(256, 256),
    ):
        assert split in ['train', 'val', 'test']
        global _PH2_SPLITS
        if _PH2_SPLITS is None:
            sample_dirs = sorted(
                [d for d in os.listdir(root_dir) if d.startswith('IMD')]
            )
            _PH2_SPLITS = _seeded_split(sample_dirs)
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.samples = _PH2_SPLITS[split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        image_path = os.path.join(
            self.root_dir,
            sample_id,
            f'{sample_id}_Dermoscopic_Image',
            f'{sample_id}.bmp'
        )
        mask_path = os.path.join(
            self.root_dir,
            sample_id,
            f'{sample_id}_lesion',
            f'{sample_id}_lesion.bmp'
        )
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.image_size is not None:
            image = F.resize(image, self.image_size, interpolation=F.InterpolationMode.BILINEAR)
            mask = F.resize(mask, self.image_size, interpolation=F.InterpolationMode.NEAREST)

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            image, mask = _default_transform(image, mask)

        return image, mask, sample_id


class DRIVEDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir='/dtu/datasets1/02516/DRIVE',
        split='train',
        transform=None,
        image_size=(256, 256),
    ):
        assert split in ['train', 'val', 'test']
        global _DRIVE_SPLITS
        training_dir = os.path.join(root_dir, 'training')
        image_paths = sorted(glob(os.path.join(training_dir, 'images', '*_training.tif')))

        if _DRIVE_SPLITS is None:
            ids = [os.path.basename(path).split('_')[0] for path in image_paths]
            _DRIVE_SPLITS = _seeded_split(ids, train_ratio=0.6, val_ratio=0.2)

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.samples = _DRIVE_SPLITS[split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        subset = 'training'
        image_name = f'{sample_id}_training.tif'
        mask_name = f'{sample_id}_manual1.gif'
        image_path = os.path.join(self.root_dir, subset, 'images', image_name)
        mask_path = os.path.join(self.root_dir, subset, '1st_manual', mask_name)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.image_size is not None:
            image = F.resize(image, self.image_size, interpolation=F.InterpolationMode.BILINEAR)
            mask = F.resize(mask, self.image_size, interpolation=F.InterpolationMode.NEAREST)

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            image, mask = _default_transform(image, mask)

        return image, mask, sample_id


def init_segmentation_transform(img_size):
    resize = lambda img, mask: (
        F.resize(img, img_size, interpolation=F.InterpolationMode.BILINEAR),
        F.resize(mask, img_size, interpolation=F.InterpolationMode.NEAREST)
    )

    def _apply(image, mask):
        image, mask = resize(image, mask)
        image = T.ToTensor()(image)
        mask = _binarize_mask(mask)
        return image, mask

    return _apply
