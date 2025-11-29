import os
import random
import typing
import xml.etree.ElementTree as ET
from glob import glob
from models import edge_boxes, selecting_search
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F
from torchvision import transforms as T
from tqdm import tqdm

def _default_transform(image):
    image = T.ToTensor()(image)
    return image

# Should not really be a dataset, as it just pre-processes and saves. Will fix later
class PotholeDatasetPreprocessor(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir='/dtu/datasets1/02516/potholes',
        split='train',
        region_proposal_method='edge_boxes',
        threshold_pothole=0.5,
        threshold_background=0.1,
    ):
        assert split in ['train', 'val', 'test']
        self.root_dir = root_dir
        self.root_dir_annotation = os.path.join(root_dir, 'annotations')
        self.split = split
        self.samples = sorted(os.listdir(self.root_dir_annotation))

        random.seed(42) 
        random.shuffle(self.samples)

        total = len(self.samples)
        train_end = int(0.8 * total)
        val_end = int(0.9 * total)

        if split == 'train':
            self.samples = self.samples[:train_end]
        elif split == 'val':
            self.samples = self.samples[train_end:val_end]
        elif split == 'test':
            self.samples = self.samples[val_end:]
        else:
            raise ValueError(f'Unknown split: {split}')
        
        self.threshold_pothole = threshold_pothole
        self.threshold_background = threshold_background

        if region_proposal_method == 'edge_boxes':
            self.region_proposer = edge_boxes.EdgeBoxesExtractor()
        elif region_proposal_method == 'selective_search':
            self.region_proposer = selecting_search.SelectiveSearchExtractor()
        else:
            raise ValueError(f'Unknown region proposal method: {region_proposal_method} - please choose either "edge_boxes" or "selective_search"')

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

        proposed_bounding_boxes = self.region_proposer.get_regions(np.array(image))

        image = _default_transform(image)

        # Find all bounding boxes
        bounding_boxes = []
        for boxes in xml_root.iter('object'):
            xmin = int(boxes.find('bndbox/xmin').text)
            ymin = int(boxes.find('bndbox/ymin').text)
            xmax = int(boxes.find('bndbox/xmax').text)
            ymax = int(boxes.find('bndbox/ymax').text)
            bounding_boxes.append((xmin, ymin, xmax, ymax))

        # For each proposed bounding box, find the bounding box with the highest IoU
        # if that score is > 0.5, we consider it a positive proposal (pothole) otherwise not (background)
        labeled_proposals = []
        for prop_box in proposed_bounding_boxes:
            x, y, w, h = prop_box
            prop_xmin = x
            prop_ymin = y
            prop_xmax = x + w
            prop_ymax = y + h

            best_iou = 0.0
            for gt_box in bounding_boxes:
                gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_box

                # Calculate IoU
                inter_xmin = max(prop_xmin, gt_xmin)
                inter_ymin = max(prop_ymin, gt_ymin)
                inter_xmax = min(prop_xmax, gt_xmax)
                inter_ymax = min(prop_ymax, gt_ymax)

                inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
                prop_area = (prop_xmax - prop_xmin) * (prop_ymax - prop_ymin)
                gt_area = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)

                union_area = prop_area + gt_area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0.0

                best_iou = max(best_iou, iou)

            if best_iou >= self.threshold_pothole:
                label = 1  # pothole
                # print(f'Proposed box: {prop_box}, Label: {label}')
            elif best_iou <= self.threshold_background:
                label = 0  # background
            else:
                continue  # ignore proposals with IoU in between
            labeled_proposals.append((prop_box.tolist(), label))

        return image, image_path, bounding_boxes, proposed_bounding_boxes, labeled_proposals


def sample_usage():
    dataset = PotholeDatasetPreprocessor(
        root_dir='/dtu/datasets1/02516/potholes',
        split='train',
        region_proposal_method='selective_search',
    )
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    print(f'Dataset size: {len(dataset)}')
    image, bounding_boxes, proposed_bounding_boxes, labeled_proposals = dataset[5]
    print(f'Image shape: {image.shape}, Bounding boxes: {len(bounding_boxes)}, Proposed bounding boxes: {len(proposed_bounding_boxes)}')

    fix, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0).numpy())
    for box, label in labeled_proposals:
        if label == 1: # pothole
            #print(f'Proposed box: {box}, Label: {label}')
            x, y, w, h = box
            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=2,
                edgecolor='g',
                facecolor='none'
            )
            ax.add_patch(rect)
        else:
            x, y, w, h = box
            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=0.5,
                edgecolor='b',
                facecolor='none'
            )
            ax.add_patch(rect)
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
    plt.savefig('project4/results/sample_pothole.png')

    # plot the image, with the different bounding boxes, and save to a file test.png
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
    for box in proposed_bounding_boxes:
        x, y, w, h = box
        rect = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=1,
            edgecolor='b',
            facecolor='none'
        )
        ax.add_patch(rect)
    plt.savefig('project4/results/sample_eb.png')


def preprocess(split: typing.Literal["train", "val", "test"] = "train"):
    dataset = PotholeDatasetPreprocessor(
        root_dir='/dtu/datasets1/02516/potholes',
        split=split,
        region_proposal_method='selective_search',
    )
    print(f'Dataset size: {len(dataset)} [{split}]')

    json_data = []

    for i in tqdm(iter(range(len(dataset)))):
        _, image_path, bounding_boxes, _, labeled_proposals = dataset[i]
        json_data.append({
            "image_path": image_path,
            "ground_truths": [(box, 1) for box in bounding_boxes],
            "labeled_proposals": labeled_proposals,
        })
    # save the file in processed_{split}.json
    import json
    with open(f'project4/processed_data/{split}.json', 'w') as f:
        json.dump(json_data, f)
    

def openjson(file: str):
    import json
    with open(file, 'r') as f:
        data = json.load(f)
    print(f'Loaded {len(data)} samples from {file}')
    for sample in data[:5]:
        print(f'Image path: {sample["image_path"]}')
        print(f'Number of ground truths: {len(sample["ground_truths"])}')
        # print(f'Number of labeled proposals: {len(sample["labeled_proposals"])}')
        num_potholes = len([label for _, label in sample["labeled_proposals"] if label == 1])
        print(f'Number of pothole proposals: {num_potholes}')
        print(f"Number of background proposals: {len(sample['labeled_proposals']) - num_potholes}")


if __name__ == '__main__':
    preprocess("train")
    preprocess("val")
    preprocess("test")