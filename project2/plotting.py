import matplotlib.pyplot as plt
import numpy as np
import os

def plot_images(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(images[i])
        plt.title(labels[i])
        plt.axis('off')
    plt.show()


def plot_all_metrics(history, save_dir='project2/plots'):
    """Plot training, validation, and test metrics in a single figure."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_losses'], 'b-', label='Train', linewidth=2, marker='o')
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation', linewidth=2, marker='s')
    ax1.axhline(y=history['test_loss'], color='g', linestyle='--', linewidth=2, label='Test')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, history['train_accs'], 'b-', label='Train', linewidth=2, marker='o')
    ax2.plot(epochs, history['val_accs'], 'r-', label='Validation', linewidth=2, marker='s')
    ax2.axhline(y=history['test_acc'], color='g', linestyle='--', linewidth=2, label='Test')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    save_path = f'{save_dir}/training_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training results to {save_path}")
    plt.close()