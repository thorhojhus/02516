import torch
def compute_training_click_metrics(logits, targets, threshold=0.5, eps=1e-6):
    """
    Computes metrics ONLY at the clicked pixels for a training batch.
    'targets' is the [B, 2, H, W] tensor.
    """
    target_mask = targets[:, 0:1, :, :]
    weight_mask = targets[:, 1:2, :, :]

    # Get predictions
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    dims = tuple(range(1, targets.dim()))

    # --- Calculate TP, TN at clicked pixels ---
    tp = (preds * target_mask).sum(dim=dims)
    tn = ((1 - preds) * (1 - target_mask) * weight_mask).sum(dim=dims)

    #Calculate click-based metrics
    total_clicks = weight_mask.sum(dim=dims)

    # Accuracy
    click_accuracy = (tp + tn + eps) / (total_clicks + eps)
    
    # Sensitivity
    total_positive_clicks = target_mask.sum(dim=dims)
    click_sensitivity = (tp + eps) / (total_positive_clicks + eps)

    # Specificity
    total_negative_clicks = total_clicks - total_positive_clicks
    click_specificity = (tn + eps) / (total_negative_clicks + eps)

    metrics = {
        'click_accuracy': click_accuracy.mean().item(),
        'click_sensitivity': click_sensitivity.mean().item(),
        'click_specificity': click_specificity.mean().item()
    }
    return metrics


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
