import warnings
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets import PH2Dataset, DRIVEDataset, init_segmentation_transform
from models.cnn import SegmentationCNN
from models.unet import UNet
from utils import set_seed, set_default_dtype_based_on_arch
set_seed(42)

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_float32_matmul_precision('high')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)


def init_model(model_name, base_channels=32, num_classes=1):
    if model_name == 'cnn':
        return SegmentationCNN(num_classes=num_classes, base_channels=base_channels).to(device)
    if model_name == 'unet':
        return UNet(num_classes=num_classes, base_channels=base_channels).to(device)
    raise ValueError(f'Unknown model name: {model_name}')


class FocalLoss(torch.nn.Module): # https://arxiv.org/pdf/1708.02002 # 
    """
    modified version of the standard cross-entropy loss function that is designed to address 
    the problem of class imbalance by down-weighting the loss from easy-to-classify examples 
    and increasing the importance of hard-to-classify ones
    """
    def __init__(self, alpha=0.25, gamma=2.0): # FROM PAPER: In general α should be decreased slightly as γ is increased (for γ = 2, α = 0.25 works best).
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


def init_loss(loss_name, pos_weight=None):
    if loss_name == 'bce':
        return torch.nn.BCEWithLogitsLoss()
    if loss_name == 'weighted_bce':
        if pos_weight is None:
            raise ValueError('pos_weight must be provided for weighted_bce')
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if loss_name == 'focal':
        return FocalLoss()
    raise ValueError(f'Unknown loss: {loss_name}')


def compute_metrics(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()

    dims = tuple(range(1, targets.dim()))

    tp = (preds * targets).sum(dim=dims)
    fp = (preds * (1 - targets)).sum(dim=dims)
    fn = ((1 - preds) * targets).sum(dim=dims)
    tn = ((1 - preds) * (1 - targets)).sum(dim=dims)

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    accuracy = (preds == targets).float().mean(dim=dims)
    sensitivity = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    metrics = {
        'dice': dice.mean().item(),
        'iou': iou.mean().item(),
        'accuracy': accuracy.mean().item(),
        'sensitivity': sensitivity.mean().item(),
        'specificity': specificity.mean().item()
    }
    return metrics


def train_one_epoch(model, dataloader, optimizer, loss_fn, threshold):
    model.train()
    total_loss = 0.0
    agg = {k: 0.0 for k in ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']}

    for images, masks, _ in tqdm(dataloader, desc='Train', leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        batch_metrics = compute_metrics(logits.detach(), masks, threshold=threshold)
        for key in agg:
            agg[key] += batch_metrics[key]

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: agg[k] / len(dataloader) for k in agg}
    return avg_loss, avg_metrics


def evaluate(model, dataloader, loss_fn, threshold):
    model.eval()
    total_loss = 0.0
    agg = {k: 0.0 for k in ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']}
    with torch.inference_mode():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = loss_fn(logits, masks)
            total_loss += loss.item()

            batch_metrics = compute_metrics(logits, masks, threshold=threshold)
            for key in agg:
                agg[key] += batch_metrics[key]

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: agg[k] / len(dataloader) for k in agg}
    return avg_loss, avg_metrics


def estimate_pos_weight(dataloader):
    total_pos = 0.0
    total_neg = 0.0
    for images, masks, _ in dataloader:
        _ = images  # unused, keeps style consistent
        pos = masks.sum().item()
        total_pos += pos
        total_neg += masks.numel() - pos
    if total_pos == 0:
        return torch.tensor(1.0, device=device)
    weight = total_neg / (total_pos + 1e-6)
    return torch.tensor(weight, device=device)


def init_datasets(dataset_name, img_size, batch_size):
    img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
    if dataset_name == 'ph2':
        transform = init_segmentation_transform(img_size)
        train_dataset = PH2Dataset(split='train', transform=transform, image_size=None)
        val_dataset = PH2Dataset(split='val', transform=transform, image_size=None)
        test_dataset = PH2Dataset(split='test', transform=transform, image_size=None)
    elif dataset_name == 'drive':
        transform = init_segmentation_transform(img_size)
        train_dataset = DRIVEDataset(split='train', transform=transform, image_size=None)
        val_dataset = DRIVEDataset(split='val', transform=transform, image_size=None)
        test_dataset = DRIVEDataset(split='test', transform=transform, image_size=None)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    return train_loader, val_loader, test_loader


def train_model(args):
    train_loader, val_loader, test_loader = init_datasets(
        dataset_name=args.dataset,
        img_size=args.img_size,
        batch_size=args.batch_size
    )

    pos_weight = None
    if args.loss == 'weighted_bce' and args.pos_weight is None:
        print('Estimating positive class weight from training data...')
        pos_weight = estimate_pos_weight(train_loader)
    elif args.loss == 'weighted_bce' and args.pos_weight is not None:
        pos_weight = torch.tensor(args.pos_weight, device=device)

    loss_fn = init_loss(args.loss, pos_weight=pos_weight)
    model = init_model(args.model, base_channels=args.base_channels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
    }

    best_val_dice = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(args.epochs):
        print(f'Epoch [{epoch + 1}/{args.epochs}]')
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, args.threshold)
        val_loss, val_metrics = evaluate(model, val_loader, loss_fn, args.threshold)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])

        print(f'Train Loss: {train_loss:.4f}, Dice: {train_metrics["dice"]:.4f}, IoU: {train_metrics["iou"]:.4f}')
        print(f'Val   Loss: {val_loss:.4f}, Dice: {val_metrics["dice"]:.4f}, IoU: {val_metrics["iou"]:.4f}')

        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            best_epoch = epoch + 1
            best_state = model.state_dict()
            print(f'New best model (val dice: {best_val_dice:.4f}) at epoch {best_epoch}')

    print(f'\nEvaluating best model from epoch {best_epoch}')
    if best_state:
        model.load_state_dict(best_state)

    test_loss, test_metrics = evaluate(model, test_loader, loss_fn, args.threshold)
    print(f'Test Loss: {test_loss:.4f}')
    for key, value in test_metrics.items():
        print(f'Test {key.capitalize()}: {value:.4f}')

    history['test'] = test_metrics
    history['best_epoch'] = best_epoch
    return history


def parse_args():
    parser = argparse.ArgumentParser(description='Segmentation training script')
    parser.add_argument('--dataset', default='ph2', choices=['ph2', 'drive'])
    parser.add_argument('--model', default='cnn', choices=['cnn', 'unet'])
    parser.add_argument('--loss', default='bce', choices=['bce', 'weighted_bce', 'focal'])
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--pos-weight', type=float, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    _ = train_model(args)
