"""
Evaluation script for Project 4 - Pothole Detection
Generates training curves, tables, and mAP evaluation for the report.
"""
import warnings
import json
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T

from dataloader import make_pothole_proposal_loaders
from models.object_detector import PotholeResNet
from utils import set_seed

set_seed(42)
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_float32_matmul_precision('high')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.CrossEntropyLoss()

print(f'Using device: {device}')

# ============ Training Functions ============

def train_one_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(dataloader, desc='Train', leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (total_loss / total_samples,
            total_correct / total_samples,
            np.array(all_preds),
            np.array(all_labels))


# ============ Detection Functions ============

def load_ground_truths(split: str):
    with open(f"project4/processed_data/{split}", "r") as f:
        data = json.load(f)
    return [(item["image_path"], item["ground_truths"], item["labeled_proposals"]) for item in data]


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def nms(predictions, iou_threshold=0.5):
    if len(predictions) == 0:
        return []

    predictions = sorted(predictions, key=lambda x: x[0], reverse=True)
    keep = []

    while predictions:
        best = predictions.pop(0)
        keep.append(best)
        predictions = [p for p in predictions if compute_iou(best[1], p[1]) < iou_threshold]

    return keep


def compute_ap(precisions, recalls):
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        precisions_above = [p for p, r in zip(precisions, recalls) if r >= t]
        if precisions_above:
            ap += max(precisions_above)
    return ap / 11


def compute_map(model, iou_threshold=0.5, nms_threshold=0.5, image_size=224,
                testset="test_selective_search_v2.json"):
    model.eval()
    all_detections = []
    total_gt = 0
    transform = T.ToTensor()

    with torch.inference_mode():
        for image_path, ground_truths, labeled_proposals in tqdm(load_ground_truths(testset), desc="Computing mAP"):
            img = Image.open(image_path).convert("RGB")
            gt_boxes = [gt[0] for gt in ground_truths if gt[1] == 1]
            gt_matched = [False] * len(gt_boxes)
            total_gt += len(gt_boxes)

            image_predictions = []
            for proposal, _ in labeled_proposals:
                x, y, w, h = map(int, proposal)
                prop_box = [x, y, x + w, y + h]

                patch = img.crop((x, y, x + w, y + h))
                patch = patch.resize((image_size, image_size), resample=Image.BICUBIC)
                patch_tensor = transform(patch).unsqueeze(0).to(device)

                logits = model(patch_tensor)
                probs = F.softmax(logits, dim=1)
                pothole_confidence = probs[0, 1].item()
                image_predictions.append((pothole_confidence, prop_box))

            image_predictions = nms(image_predictions, iou_threshold=nms_threshold)

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

                is_tp = best_iou >= iou_threshold and best_gt_idx >= 0
                if is_tp:
                    gt_matched[best_gt_idx] = True
                all_detections.append((confidence, is_tp))

    if total_gt == 0:
        return 0.0, [], []

    all_detections.sort(key=lambda x: x[0], reverse=True)

    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []

    for confidence, is_tp in all_detections:
        if is_tp:
            tp_cumsum += 1
        else:
            fp_cumsum += 1

        precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
        recalls.append(tp_cumsum / total_gt)

    ap = compute_ap(precisions, recalls)
    return ap, precisions, recalls, total_gt, tp_cumsum, fp_cumsum


# ============ Main Evaluation ============

def main():
    results_dir = Path("project4/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path("project4/checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameters (matching train.py)
    lr = 3e-4
    weight_decay = 1e-4
    batch_size = 64
    num_epochs = 10
    image_size = 224
    target_pos_fraction = 0.30

    # Initialize model
    model = PotholeResNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Data loaders
    train_loader, val_loader = make_pothole_proposal_loaders(
        processed_dir="project4/processed_data",
        trainset="train_selective_search_v2.json",
        valset="val_selective_search_v2.json",
        batch_size=batch_size,
        num_workers=4,
        image_size=image_size,
        target_pos_fraction=target_pos_fraction,
    )

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0
    best_epoch = 0

    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoints_dir / "best_model.pth")

    # Load best model
    print(f"\nLoading best model from epoch {best_epoch} (val_acc: {best_val_acc:.4f})")
    checkpoint = torch.load(checkpoints_dir / "best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation on validation set (for confusion matrix)
    val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader)

    # Confusion matrix
    tp = ((val_preds == 1) & (val_labels == 1)).sum()
    tn = ((val_preds == 0) & (val_labels == 0)).sum()
    fp = ((val_preds == 1) & (val_labels == 0)).sum()
    fn = ((val_preds == 0) & (val_labels == 1)).sum()

    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS (Validation Set)")
    print("="*60)
    print(f"Loss: {val_loss:.4f}")
    print(f"Accuracy: {val_acc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Neg    Pos")
    print(f"Actual Neg   {tn:5d}  {fp:5d}")
    print(f"       Pos   {fn:5d}  {tp:5d}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # mAP evaluation
    print("\n" + "="*60)
    print("DETECTION RESULTS (Test Set)")
    print("="*60)

    ap, precisions, recalls, total_gt, total_tp, total_fp = compute_map(
        model, iou_threshold=0.5, nms_threshold=0.5,
        image_size=image_size, testset="test_selective_search_v2.json"
    )

    print(f"\nmAP@0.5: {ap:.4f}")
    print(f"Total GT boxes: {total_gt}")
    print(f"True Positives: {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"Recall: {total_tp/total_gt:.4f}")

    # ============ Generate Plots ============

    # Learning curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, num_epochs + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "learning_curves.png", dpi=150)
    plt.close()
    print(f"\nSaved learning curves to {results_dir / 'learning_curves.png'}")

    # Precision-Recall curve
    if precisions and recalls:
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, 'b-', linewidth=2)
        plt.fill_between(recalls, precisions, alpha=0.2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (AP = {ap:.4f})')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / "precision_recall_curve.png", dpi=150)
        plt.close()
        print(f"Saved PR curve to {results_dir / 'precision_recall_curve.png'}")

    # ============ Generate LaTeX Table ============

    latex_table = f"""
% Classification Results Table
\\begin{{table}}[h]
\\centering
\\caption{{Proposal classification results on the validation set.}}
\\label{{tab:classification}}
\\begin{{tabular}}{{lc}}
\\toprule
Metric & Value \\\\
\\midrule
Validation Loss & {val_loss:.4f} \\\\
Validation Accuracy & {val_acc:.4f} \\\\
Precision & {precision:.4f} \\\\
Recall & {recall:.4f} \\\\
F1 Score & {f1:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

% Detection Results Table
\\begin{{table}}[h]
\\centering
\\caption{{Object detection results on the test set (IoU threshold = 0.5).}}
\\label{{tab:detection}}
\\begin{{tabular}}{{lc}}
\\toprule
Metric & Value \\\\
\\midrule
mAP@0.5 & {ap:.4f} \\\\
Total Ground Truth Boxes & {total_gt} \\\\
True Positives & {total_tp} \\\\
False Positives & {total_fp} \\\\
Detection Recall & {total_tp/total_gt:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

% Confusion Matrix Table
\\begin{{table}}[h]
\\centering
\\caption{{Confusion matrix for proposal classification on the validation set.}}
\\label{{tab:confusion}}
\\begin{{tabular}}{{lcc}}
\\toprule
& \\textbf{{Predicted Negative}} & \\textbf{{Predicted Positive}} \\\\
\\midrule
\\textbf{{Actual Negative}} & {tn} & {fp} \\\\
\\textbf{{Actual Positive}} & {fn} & {tp} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

% Training History (for plotting)
% Epoch, Train Loss, Train Acc, Val Loss, Val Acc
"""

    for i in range(num_epochs):
        latex_table += f"% {i+1}, {history['train_loss'][i]:.4f}, {history['train_acc'][i]:.4f}, {history['val_loss'][i]:.4f}, {history['val_acc'][i]:.4f}\n"

    with open(results_dir / "results_tables.tex", "w") as f:
        f.write(latex_table)

    print(f"Saved LaTeX tables to {results_dir / 'results_tables.tex'}")

    # Save history as JSON for future use
    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved training history to {results_dir / 'training_history.json'}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"Test mAP@0.5: {ap:.4f}")


if __name__ == "__main__":
    main()
