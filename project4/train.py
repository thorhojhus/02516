import warnings
import argparse
from PIL import Image
from pathlib import Path
import json
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from project4.dataloader import make_pothole_proposal_loaders
from models.object_detector import CNN, PotholeResNet
from utils import set_seed, set_default_dtype_based_on_arch
set_seed(42)

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_float32_matmul_precision('high')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = torch.nn.CrossEntropyLoss()

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)


def init_model(model_name):
    if model_name == 'cnn':
        return CNN().to(device)
    if model_name == 'resnet':
        return PotholeResNet().to(device)
    raise ValueError(f'Unknown model name: {model_name}')


def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer):
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for sliced_images, labels in tqdm(dataloader, desc='Train', leave=False):
        sliced_images = sliced_images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(sliced_images)

        loss = loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean().item()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)

    return avg_loss, avg_acc


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_acc += (preds == labels).float().mean().item()

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    return avg_loss, avg_acc


def load_ground_truths(split: str) -> list[tuple[str, list, list]]:
    with open(f"project4/processed_data/{split}.json", "r") as f:
        data = json.load(f)
    return [(item["image_path"], item["ground_truths"], item["labeled_proposals"]) for item in data]


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes.
    box1: [xmin, ymin, xmax, ymax]
    box2: [xmin, ymin, xmax, ymax]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def compute_ap(precisions, recalls):
    """
    Compute Average Precision using 11-point interpolation.
    """
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        precisions_above_recall = [p for p, r in zip(precisions, recalls) if r >= t]
        if precisions_above_recall:
            ap += max(precisions_above_recall)
    return ap / 11


def nms(predictions, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to filter overlapping detections.
    
    Args:
        predictions: List of (confidence, box) where box is [xmin, ymin, xmax, ymax]
        iou_threshold: IoU threshold for suppression
    
    Returns:
        Filtered list of (confidence, box)
    """
    if len(predictions) == 0:
        return []
    
    # Sort by confidence (descending)
    predictions = sorted(predictions, key=lambda x: x[0], reverse=True)
    
    keep = []
    while predictions:
        # Take the highest confidence prediction
        best = predictions.pop(0)
        keep.append(best)
        
        # Filter out predictions with high IoU overlap
        remaining = []
        for pred in predictions:
            iou = compute_iou(best[1], pred[1])
            if iou < iou_threshold:
                remaining.append(pred)
        predictions = remaining
    
    return keep


def MAP(model, iou_threshold=0.5, nms_threshold=0.5, image_size=224):
    """
    Compute Mean Average Precision for pothole detection.
    
    Args:
        model: The trained classifier model
        iou_threshold: IoU threshold for considering a detection as correct
        nms_threshold: IoU threshold for Non-Maximum Suppression
        image_size: Size to resize proposal patches to
    
    Returns:
        mAP: Mean Average Precision score
    """
    model.eval()
    
    all_detections = []  # List of (confidence, is_tp)
    total_gt = 0  # Total number of ground truth boxes
    
    with torch.inference_mode():
        for _, (image_path, ground_truths, labeled_proposals) in enumerate(tqdm(load_ground_truths("val"), desc="Computing mAP")):
            img = Image.open(image_path).convert("RGB")
            
            # Extract ground truth boxes (format: [[xmin, ymin, xmax, ymax], label])
            gt_boxes = [gt[0] for gt in ground_truths if gt[1] == 1]
            gt_matched = [False] * len(gt_boxes)
            total_gt += len(gt_boxes)
            
            # Get predictions for all proposals
            image_predictions = []
            
            for proposal, _ in labeled_proposals:
                x, y, w, h = map(int, proposal)
                # Convert to [xmin, ymin, xmax, ymax] format
                prop_box = [x, y, x + w, y + h]
                
                # Extract and preprocess patch
                patch = img.crop((x, y, x + w, y + h))
                patch = patch.resize((image_size, image_size), resample=Image.BICUBIC)
                patch_tensor = torch.tensor(np.array(patch)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                patch_tensor = patch_tensor.to(device)
                
                # Get model prediction
                logits = model(patch_tensor)
                probs = F.softmax(logits, dim=1)
                
                # Get confidence for pothole class (class 1)
                pothole_confidence = probs[0, 1].item()
                
                image_predictions.append((pothole_confidence, prop_box))
            
            # Apply Non-Maximum Suppression
            image_predictions = nms(image_predictions, iou_threshold=nms_threshold)
            
            # Sort predictions by confidence (descending)
            image_predictions.sort(key=lambda x: x[0], reverse=True)
            
            # Match predictions to ground truths
            for confidence, pred_box in image_predictions:
                best_iou = 0.0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_matched[gt_idx]:
                        continue
                    
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # Determine if this is a true positive
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    is_tp = True
                    gt_matched[best_gt_idx] = True
                else:
                    is_tp = False
                
                all_detections.append((confidence, is_tp, pred_box))
            
            # fig, ax = plt.subplots()
            # img_np = np.array(img)
            # ax.imshow(img_np)
            # for confidence, is_tp, pred_box in all_detections:
            #     # add confidence as text label to the rectangle
            #     print(confidence, end=", ")
            #     xmin, ymin, xmax, ymax = pred_box
            #     rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, edgecolor='red' if is_tp else 'blue', facecolor='none', linewidth=1)
            #     if is_tp:
            #         ax.text(xmin, ymin, f"{confidence:.2f}", color='red', fontsize=16, verticalalignment='top')
            #     ax.add_patch(rect)
            # for gt_box in gt_boxes:
            #     xmin, ymin, xmax, ymax = gt_box
            #     rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, edgecolor='green', facecolor='none', linewidth=1)
            #     ax.add_patch(rect)
            # plt.savefig("project4/results/detection_visualization.png")
            # break
    
    if total_gt == 0:
        print("No ground truth boxes found!")
        return 0.0
    
    # Sort all detections by confidence
    all_detections.sort(key=lambda x: x[0], reverse=True)
    
    # Compute precision and recall at each detection
    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []
    
    for confidence, is_tp, _ in all_detections:
        if is_tp:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / total_gt
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Compute AP using 11-point interpolation
    ap = compute_ap(precisions, recalls)
    
    print(f"mAP@{iou_threshold}: {ap:.4f}")
    print(f"Total GT boxes: {total_gt}, Total detections: {len(all_detections)}")
    print(f"True Positives: {tp_cumsum}, False Positives: {fp_cumsum}")
    
    return ap

if __name__ == "__main__":
    model = init_model('resnet')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_loader, val_loader, test_loader = make_pothole_proposal_loaders(
        processed_dir="project4/processed_data",
        batch_size=32,
        num_workers=4,
        image_size=224,
        target_pos_fraction=0.33,
    )
    num_epochs = 3
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    MAP(model=model)