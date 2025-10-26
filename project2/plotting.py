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


def plot_all_metrics(history, save_dir='project2/plots', plot_name: str = 'training_results'):
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
    save_path = f'{save_dir}/{plot_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training results to {save_path}")
    plt.close()

    # also save history as a numpy file
    np.savez(f'{save_dir}/{plot_name}_history.npz', **history)


if __name__ == '__main__':
    import glob
    # find all npz files in the plots directory
    npz_files = glob.glob('project2/plots/*.npz')

    history_dicts = []

    if not npz_files:
        print("No history files found in project2/plots")
    else:
        # collect all history files and plot them
        for npz_file in npz_files:
            model_name_with_path, _, _ = npz_file.partition("_")
            model_name = model_name_with_path.split('/')[-1]
            history_data = np.load(npz_file)
            history_dict = {key: history_data[key] for key in history_data.files}
            history_dicts.append((model_name, history_dict))
    
    # Do a shared plot showing only accuracies for all the models in the history_dicts:
    if history_dicts:
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(history_dicts[0][1]['train_accs']) + 1)

        for idx, (model_name, history) in enumerate(history_dicts):
            plt.plot(epochs, history['train_accs'], label=f'{model_name}', linestyle='-')
            # plt.plot(epochs, history['val_accs'], label=f'{model_name} Val', linestyle='--')
            # plt.axhline(y=history['test_acc'], linestyle=':', label=f'{model_name} Test')
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Model Accuracies', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        plt.tight_layout()
        shared_plot_path = 'project2/plots/all_models_accuracies.png'
        plt.savefig(shared_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved all models accuracies plot to {shared_plot_path}")
        plt.close()
    
    if history_dicts:
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(history_dicts[0][1]['train_accs']) + 1)

        for idx, (model_name, history) in enumerate(history_dicts):
            # plt.plot(epochs, history['train_accs'], label=f'{model_name}', linestyle='-')
            plt.plot(epochs, history['val_accs'], label=f'{model_name} Val', linestyle='-')
            # plt.axhline(y=history['test_acc'], linestyle=':', label=f'{model_name} Test')
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Model Accuracies (Validation)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        plt.xlim([1, 10])
        plt.xticks(epochs)
        plt.tight_layout()
        shared_plot_path = 'project2/plots/all_models_accuracies_val.png'
        plt.savefig(shared_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved all models accuracies plot to {shared_plot_path}")
        plt.close()