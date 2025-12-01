import json
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image
from torchvision import transforms as T

class PotholeProposalDataset(Dataset):
    """
    Dataset over region proposals for pothole detection.

    Expects JSON created by `preprocessor.preprocess`, with entries like:
    {
        "image_path": "...",
        "ground_truths": [((xmin, ymin, xmax, ymax), 1), ...],
        "labeled_proposals": [([x, y, w, h], label), ...]
    }

    each dataset sample is a single proposal crop resized to 224x224
    binary label: 0 = background, 1 = pothole.
    """

    def __init__(
        self,
        json_path: str | Path,
        transform: Optional[Callable] = None,
        image_size: int | Tuple[int, int] = 224,
    ) -> None:
        super().__init__()
        self.json_path = Path(json_path)
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size

        self.transform = transform or T.ToTensor()
        # Simple per-worker in-memory image cache: image_path -> PIL.Image
        self._image_cache: dict[str, Image.Image] = {}

        with self.json_path.open("r") as f:
            data = json.load(f)

        # Flatten all proposals into a single list of (image_path, box, label)
        # where box is [x, y, w, h] in image coordinates.
        samples: List[Tuple[str, Sequence[float], int]] = []
        for entry in data:
            image_path = entry["image_path"]
            for box, label in entry["labeled_proposals"]:
                # JSON may store tuples as lists; keep as list[float | int]
                samples.append((image_path, box, int(label)))
            for box, label in entry["ground_truths"]:
                # convert box to [x, y, w, h]
                x, y, xmax, ymax = box
                w = xmax - x
                h = ymax - y
                samples.append((image_path, [x, y, w, h], int(label)))

        self.samples = samples

    def _load_image(self, path: str) -> Image.Image:
        """Load image from disk with simple per-instance cache."""
        img = self._image_cache.get(path)
        if img is None:
            img = Image.open(path).convert("RGB")
            self._image_cache[path] = img
        return img

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, box, label = self.samples[idx]
        # Load full image (cached per worker)
        image = self._load_image(image_path)

        # Proposal box is [x, y, w, h]
        x, y, w, h = box
        # Ensure ints for PIL crop
        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)

        patch = image.crop((x1, y1, x2, y2))
        patch = patch.resize(self.image_size, resample=Image.BICUBIC) 

        patch_tensor = self.transform(patch)
        return patch_tensor, int(label)


def _compute_undersampling_indices(
    labels: Sequence[int],
    target_pos_fraction: float = 0.33,
    seed: Optional[int] = 42,
) -> List[int]:
    """
    Compute indices for undersampling the background class so that
    the positive (label==1) samples make up approximately `target_pos_fraction`
    of the resulting dataset.
    
    Returns a list of indices to keep (all positives + subsampled negatives).
    """
    import random
    if seed is not None:
        random.seed(seed)
    
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    pos_indices = (labels_tensor == 1).nonzero(as_tuple=True)[0].tolist()
    neg_indices = (labels_tensor == 0).nonzero(as_tuple=True)[0].tolist()
    
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    
    r = float(target_pos_fraction)
    r = max(1e-3, min(1.0 - 1e-3, r))
    
    # To achieve target_pos_fraction = n_pos / (n_pos + n_neg_keep):
    # => n_neg_keep = n_pos * (1 - r) / r
    n_neg_keep = int(n_pos * (1.0 - r) / r)
    n_neg_keep = min(n_neg_keep, n_neg)  # Can't keep more than we have
    
    print(f"pos: {n_pos}, neg: {n_neg}, keeping {n_neg_keep} negatives (target_pos_fraction={r:.2f})")
    
    # Randomly select negatives to keep
    neg_indices_keep = random.sample(neg_indices, n_neg_keep)
    
    # Combine all positives with subsampled negatives
    all_indices = pos_indices + neg_indices_keep
    random.shuffle(all_indices)
    
    return all_indices


def make_pothole_proposal_loaders(
    processed_dir: str | Path = "project4/processed_data",
    trainset: str = "train_selective_search_v2.json",
    valset: str = "val_selective_search_v2.json",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 64,
    target_pos_fraction: float = 0.33,
    transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders for proposal-level pothole detection.

    Returns DataLoaders that yield:
        images: [B, 3, 224, 224]
        labels: [B] with values {0, 1}
    matching the expected input/output of `models.object_detector.CNN`.
    """
    processed_dir = Path(processed_dir)

    train_dataset = PotholeProposalDataset(
        processed_dir / trainset,
        transform=transform,
        image_size=image_size,
    )
    val_dataset = PotholeProposalDataset(
        processed_dir / valset,
        transform=transform,
        image_size=image_size,
    )

    # Undersample background class so that positives are at least ~33%.
    # samples are stored as (image_path, box, label)
    train_labels = [sample[2] for sample in train_dataset.samples]
    undersample_indices = _compute_undersampling_indices(train_labels, target_pos_fraction)
    sampler = SubsetRandomSampler(undersample_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train_loader, val_loader = make_pothole_proposal_loaders(
        processed_dir="project4/processed_data",
        trainset="train_selective_search_v2.json",
        valset="val_selective_search_v2.json",
        batch_size=32,
        num_workers=4,
        image_size=224,
        target_pos_fraction=0.33,
    )
    i = 0
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        # save the images and labels to a file
        plt.imshow(images[0].permute(1, 2, 0).numpy())
        plt.title(f"Label: {labels[0]}")
        plt.savefig(f"project4/results/sample_pothole_{i}.png")
        plt.close()
        i += 1
        if i > 10:
            break
