import os
import random
import xml.etree.ElementTree as ET
from glob import glob

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F
from torchvision import transforms as T

def _default_transform(image):
    image = T.ToTensor()(image)
    return image


class PotholeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir='/dtu/datasets1/02516/potholes',
        split='train',
        transform=None,
    ):
        assert split in ['train', 'val', 'test']
        self.root_dir = root_dir
        self.root_dir_annotation = os.path.join(root_dir, 'annotations')
        self.split = split
        self.transform = transform
        self.samples = [name for name in os.listdir(self.root_dir_annotation)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        # sample_id contains the file potholeXXX.xml; we need to parse it to get bounding box and the image path
        xml_tree = ET.parse(os.path.join(
            self.root_dir_annotation,
            sample_id,
        ))
        xml_root = xml_tree.getroot()
        image_path = xml_root.find('filename').text
        image_path = os.path.join(
            self.root_dir,
            "images",
            image_path,
        )
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        else:
            image = _default_transform(image)

        # Find all bounding boxes
        bounding_boxes = []
        for boxes in xml_root.iter('object'):
            xmin = int(boxes.find('bndbox/xmin').text)
            ymin = int(boxes.find('bndbox/ymin').text)
            xmax = int(boxes.find('bndbox/xmax').text)
            ymax = int(boxes.find('bndbox/ymax').text)
            bounding_boxes.append((xmin, ymin, xmax, ymax))

        return image, bounding_boxes


def init_segmentation_transform(img_size):
    def resize(img):
        return F.resize(img, img_size, interpolation=F.InterpolationMode.BILINEAR)

    def _apply(image):
        image = resize(image)
        image = T.ToTensor()(image)
        return image

    return _apply

if __name__ == '__main__':
    dataset = PotholeDataset(
        root_dir='/dtu/datasets1/02516/potholes',
        split='train',
    )
    print(f'Dataset size: {len(dataset)}')
    image, bounding_boxes = dataset[0]
    print(f'Image shape: {image.shape}, Bounding boxes: {bounding_boxes}')

    # plot the image, with the different bounding boxes, and save to a file test.png
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0).numpy())
    for box in bounding_boxes:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
    plt.savefig('test.png')