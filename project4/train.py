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
from torchvision import transforms as T

from dataloader import make_pothole_proposal_loaders
from models.object_detector import CNN, PotholeResNet, PotholeResNet50
from utils import set_seed, set_default_dtype_based_on_arch
set_seed(42)

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_float32_matmul_precision('high')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = torch.nn.CrossEntropyLoss()

print(f'Using device: {device}')

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)


def init_model(model_name):
    if model_name == 'cnn':
        return CNN().to(device)
    if model_name == 'resnet':
        return PotholeResNet().to(device)
    if model_name == 'resnet50':
        return PotholeResNet50().to(device)
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
    with open(f"project4/processed_data/{split}", "r") as f:
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


def MAP(model, iou_threshold=0.5, nms_threshold=0.5, image_size=224, testset="test_selective_search_v2.json"):
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
    transform = T.ToTensor()
    
    with torch.inference_mode():
        for image_path, ground_truths, labeled_proposals in tqdm(load_ground_truths(testset), desc="Computing mAP"):
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
                patch_tensor = transform(patch).unsqueeze(0)
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



# Dont use for report
def MaxMAP(model, iou_threshold=0.5, testset="test_selective_search_v2.json"):
    """
    Compute theoretical maximum MAP by treating all proposals as detections
    regardless of classifier confidence. This gives the upper bound based on
    proposal quality alone.
    
    Args:
        model: Not used, kept for consistency
        iou_threshold: IoU threshold for considering a detection as correct
    
    Returns:
        max_mAP: Theoretical maximum mAP achievable with current proposals
    """
    all_detections = []  # List of (dummy_confidence, is_tp, pred_box)
    total_gt = 0
    
    for image_path, ground_truths, labeled_proposals in tqdm(load_ground_truths(testset), desc="Computing Max mAP"):
        img = Image.open(image_path).convert("RGB")
        
        # Extract ground truth boxes
        gt_boxes = [gt[0] for gt in ground_truths if gt[1] == 1]
        gt_matched = [False] * len(gt_boxes)
        total_gt += len(gt_boxes)
        
        # Process all proposals
        for proposal, _ in labeled_proposals:
            x, y, w, h = map(int, proposal)
            prop_box = [x, y, x + w, y + h]
            
            # Find best matching GT box
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                
                iou = compute_iou(prop_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Determine if this is a true positive
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                is_tp = True
                gt_matched[best_gt_idx] = True
            else:
                is_tp = False
            
            # Use IoU as confidence (higher IoU = higher confidence)
            all_detections.append((best_iou, is_tp, prop_box))
    
    if total_gt == 0:
        print("No ground truth boxes found!")
        return 0.0
    
    # Sort by "confidence" (IoU with best matching GT)
    all_detections.sort(key=lambda x: x[0], reverse=True)
    
    # Compute precision and recall
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
    max_ap = compute_ap(precisions, recalls)
    
    print(f"\nTheoretical Max mAP@{iou_threshold}: {max_ap:.4f}")
    print(f"Total GT boxes: {total_gt}, Total proposals: {len(all_detections)}")
    print(f"Max possible True Positives: {tp_cumsum}, False Positives: {fp_cumsum}")
    print(f"Recall upper bound: {tp_cumsum/total_gt:.4f}")
    
    return max_ap


def visualize_first_test_image(model, iou_threshold=0.5, nms_threshold=0.5, image_size=224, testset="test_selective_search_v2.json", output_path="project4/results/detection_visualization.png"):
    """
    Visualize detections on the first test image with NMS applied.
    Green: Ground truth boxes
    Red: Best matching predictions (IoU >= threshold)
    Blue: Predictions with confidence > 0.5
    Orange: Predictions with confidence <= 0.5
    """
    model.eval()
    transform = T.ToTensor()
    
    with torch.inference_mode():
        # Get first test image
        test_data = load_ground_truths(testset)
        if not test_data:
            print("No test data found!")
            return
            
        image_path, ground_truths, labeled_proposals = test_data[0]
        img = Image.open(image_path).convert("RGB")
        
        # Extract ground truth boxes
        gt_boxes = [gt[0] for gt in ground_truths if gt[1] == 1]
        
        # Get predictions for all proposals
        all_predictions = []
        
        for proposal, _ in labeled_proposals:
            x, y, w, h = map(int, proposal)
            prop_box = [x, y, x + w, y + h]
            
            # Extract and preprocess patch
            patch = img.crop((x, y, x + w, y + h))
            patch = patch.resize((image_size, image_size), resample=Image.BICUBIC)
            patch_tensor = transform(patch).unsqueeze(0)
            patch_tensor = patch_tensor.to(device)
            
            # Get model prediction
            logits = model(patch_tensor)
            probs = F.softmax(logits, dim=1)
            pothole_confidence = probs[0, 1].item()
            
            all_predictions.append((pothole_confidence, prop_box))
        
        # Apply Non-Maximum Suppression
        all_predictions = nms(all_predictions, iou_threshold=nms_threshold)
        
        # Find best matches for each GT box
        best_matches = []
        for gt_box in gt_boxes:
            best_iou = 0.0
            best_pred = None
            
            for confidence, pred_box in all_predictions:
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pred = (confidence, pred_box)
            
            if best_iou >= iou_threshold and best_pred:
                best_matches.append(best_pred)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        img_np = np.array(img)
        ax.imshow(img_np)
        
        # Draw ground truth boxes (green)
        for gt_box in gt_boxes:
            xmin, ymin, xmax, ymax = gt_box
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                edgecolor='green', facecolor='none', linewidth=2)
            ax.add_patch(rect)
        
        # Draw predictions
        best_match_boxes = [box for _, box in best_matches]
        
        for confidence, pred_box in all_predictions:
            xmin, ymin, xmax, ymax = pred_box
            
            # Determine color based on matching and confidence
            if pred_box in best_match_boxes:
                color = 'red'
                label = 'Best Match'
                linewidth = 2
                show_text = True
            elif confidence > 0.5:
                color = 'blue'
                label = 'Pred > 0.5'
                linewidth = 1.5
                show_text = True
            else:
                color = 'orange'
                label = 'Pred ≤ 0.5'
                linewidth = 1
                show_text = False
            
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                edgecolor=color, facecolor='none', linewidth=linewidth)
            ax.add_patch(rect)
            
            # Add confidence text only for red and blue boxes
            if show_text:
                ax.text(xmin, ymin - 5, f"{confidence:.2f}", 
                       color=color, fontsize=8, verticalalignment='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='none', edgecolor='green', linewidth=2, label='Ground Truth'),
            Patch(facecolor='none', edgecolor='red', linewidth=2, label='Best Match'),
            Patch(facecolor='none', edgecolor='blue', linewidth=1.5, label='Pred > 0.5'),
            Patch(facecolor='none', edgecolor='orange', linewidth=1, label='Pred ≤ 0.5')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.axis('off')
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved to {output_path}")
        print(f"Total GT boxes: {len(gt_boxes)}")
        print(f"Proposals before NMS: {len(labeled_proposals)}")
        print(f"Proposals after NMS: {len(all_predictions)}")
        print(f"Best matches: {len(best_matches)}")
        print(f"Predictions > 0.5: {sum(1 for c, _ in all_predictions if c > 0.5)}")
        print(f"Predictions ≤ 0.5: {sum(1 for c, _ in all_predictions if c <= 0.5)}")




if __name__== "__main__":
    model = init_model('resnet')
    
    trainset = "train_selective_search_v2.json"
    valset = "val_selective_search_v2.json"
    testset = "test_selective_search_v2.json"
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    train_loader, val_loader = make_pothole_proposal_loaders(
        processed_dir="project4/processed_data",
        trainset=trainset,
        valset=valset,
        batch_size=64,
        num_workers=4,
        image_size=224,
        target_pos_fraction=0.30,
    )
    
    num_epochs = 2
    best_val_acc = 0.0
    best_model_path = "project4/checkpoints/best_model.pth"
    
    # Create checkpoint directory if it doesn't exist
    Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"Saved best model with validation accuracy: {val_acc:.4f}")
    
    # Load best model for evaluation
    print(f"\nLoading best model (val_acc: {best_val_acc:.4f})...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Visualize first test image with NMS
    visualize_first_test_image(model, iou_threshold=0.5, nms_threshold=0.5, testset=valset)
    
    MAP(model=model, iou_threshold=0.5, nms_threshold=0.5, testset=testset)
    
    # Dont use for report
    # MaxMAP(model=model, testset=testset)