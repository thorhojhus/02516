import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for WSL/headless environments
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pandas as pd


def plot_training_curves(history, save_path, model_name='Model'):
    """
    Plot training and validation curves for loss and dice metric.
    Inspired by project2 plotting style.
    
    Args:
        history: dict with keys 'train_loss', 'val_loss', 'train_dice', 'val_dice', 
                 and 'test' (dict with test metrics)
        save_path: path to save the plot
        model_name: name for the plot title
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
    if 'test' in history:
        # Test loss as horizontal line (not in history from train_model, need to compute separately)
        pass
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Dice plot (primary segmentation metric)
    ax2.plot(epochs, history['train_dice'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs, history['val_dice'], 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
    if 'test' in history and 'dice' in history['test']:
        ax2.axhline(y=history['test']['dice'], color='g', linestyle='--', linewidth=2, label='Test')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Dice Score', fontsize=12)
    ax2.set_title(f'{model_name} - Dice Score', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {save_path}")
    plt.close()


def plot_metric_comparison(histories_dict, metric='dice', save_path='comparison.png', title='Model Comparison'):
    """
    Compare a specific metric across multiple experiments.
    
    Args:
        histories_dict: dict mapping experiment_name -> history_dict
        metric: which metric to plot (dice, iou, accuracy, etc.)
        save_path: path to save the plot
        title: plot title
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories_dict)))
    
    for idx, (exp_name, history) in enumerate(histories_dict.items()):
        epochs = range(1, len(history[f'train_{metric}']) + 1)
        
        # Training curves
        ax1.plot(epochs, history[f'train_{metric}'], 
                label=exp_name, linewidth=2, marker='o', markersize=3,
                color=colors[idx])
        
        # Validation curves
        ax2.plot(epochs, history[f'val_{metric}'], 
                label=exp_name, linewidth=2, marker='s', markersize=3,
                color=colors[idx])
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel(metric.capitalize(), fontsize=12)
    ax1.set_title(f'{title} - Training {metric.capitalize()}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel(metric.capitalize(), fontsize=12)
    ax2.set_title(f'{title} - Validation {metric.capitalize()}', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {save_path}")
    plt.close()


def plot_test_metrics_comparison(results_dict, save_path='test_metrics.png', title='Test Metrics Comparison'):
    """
    Create bar chart comparing test metrics across experiments.
    
    Args:
        results_dict: dict mapping experiment_name -> test_metrics_dict
        save_path: path to save the plot
        title: plot title
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    metrics = ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']
    experiments = list(results_dict.keys())
    
    # Prepare data
    data = {metric: [] for metric in metrics}
    for exp_name in experiments:
        test_results = results_dict[exp_name]
        for metric in metrics:
            data[metric].append(test_results.get(metric, 0))
    
    # Create grouped bar chart
    x = np.arange(len(experiments))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, metric in enumerate(metrics):
        offset = width * (i - 2)
        ax.bar(x + offset, data[metric], width, label=metric.capitalize())
    
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved test metrics comparison to {save_path}")
    plt.close()


def save_results_table(results_dict, save_path='results_summary'):
    """
    Save results as both CSV and Markdown table for easy report inclusion.
    
    Args:
        results_dict: dict mapping experiment_name -> test_metrics_dict
        save_path: base path (without extension) to save tables
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    metrics = ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']
    
    # Create DataFrame
    data = []
    for exp_name, test_results in results_dict.items():
        row = {'Experiment': exp_name}
        for metric in metrics:
            row[metric.capitalize()] = f"{test_results.get(metric, 0):.4f}"
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    csv_path = f"{save_path}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results table to {csv_path}")
    
    # Save as Markdown (manual formatting to avoid tabulate dependency)
    md_path = f"{save_path}.md"
    with open(md_path, 'w') as f:
        # Write header
        headers = ['Experiment'] + [m.capitalize() for m in metrics]
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('|' + '|'.join(['---' for _ in headers]) + '|\n')
        
        # Write data rows
        for exp_name, test_results in results_dict.items():
            values = [exp_name] + [f"{test_results.get(m, 0):.4f}" for m in metrics]
            f.write('| ' + ' | '.join(values) + ' |\n')
    
    print(f"Saved results table to {md_path}")
    
    return df


def load_history(history_path):
    """Load history from JSON file."""
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history


def save_history(history, save_path):
    """Save history to JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved history to {save_path}")

