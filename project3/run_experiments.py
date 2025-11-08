import os
import sys
import json
import argparse
from pathlib import Path
import torch
import warnings

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from train import init_datasets, init_model, init_loss, train_one_epoch, evaluate, estimate_pos_weight, set_seed
from plotting import plot_training_curves, plot_metric_comparison, plot_test_metrics_comparison, save_results_table, save_history
from utils import set_seed

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_float32_matmul_precision('high')

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_experiment_name(model, dataset, loss):
    """Generate consistent experiment name."""
    return f"{model}_{dataset}_{loss}"


def run_single_experiment(config, results_dir, plots_dir):
    """
    Run a single training experiment.

    Args:
        config: dict with 'model', 'dataset', 'loss' keys
        results_dir: directory to save results
        plots_dir: directory to save plots

    Returns:
        dict with test metrics and training history
    """
    model_name = config['model']
    dataset_name = config['dataset']
    loss_name = config['loss']

    exp_name = get_experiment_name(model_name, dataset_name, loss_name)
    print(f"\n{'='*80}")
    print(f"Running Experiment: {exp_name}")
    print(f"{'='*80}\n")

    # Create experiment-specific directories
    exp_results_dir = results_dir / exp_name
    exp_results_dir.mkdir(parents=True, exist_ok=True)

    # Training parameters (can be adjusted)
    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 8)
    lr = config.get('lr', 1e-4)
    #lr = config.get('lr', 1e-3 if dataset_name == 'drive' else 1e-4)
    weight_decay = config.get('weight_decay', 1e-5)
    img_size = config.get('img_size', 512)
    base_channels = config.get('base_channels', 32)
    threshold = config.get('threshold', 0.5)

    # Initialize datasets
    print(f"Loading {dataset_name.upper()} dataset...")
    train_loader, val_loader, test_loader = init_datasets(
        dataset_name=dataset_name,
        img_size=img_size,
        batch_size=batch_size
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Initialize loss function
    pos_weight = None
    if loss_name == 'weighted_bce':
        print('Estimating positive class weight from training data...')
        pos_weight = estimate_pos_weight(train_loader)
        print(f'Using auto-computed pos_weight: {pos_weight.item():.2f}')

    loss_fn = init_loss(loss_name, pos_weight=pos_weight)

    # Initialize model
    print(f"Initializing {model_name.upper()} model...")
    model = init_model(model_name, base_channels=base_channels)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
        'train_iou': [],
        'val_iou': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_sensitivity': [],
        'val_sensitivity': [],
        'train_specificity': [],
        'val_specificity': [],
    }

    best_val_dice = -1.0
    best_epoch = 0
    best_state = None

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        print(f'\nEpoch [{epoch + 1}/{epochs}]')

        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, threshold
        )

        # Validate
        val_loss, val_metrics = evaluate(
            model, val_loader, loss_fn, threshold
        )

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        for metric in ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']:
            history[f'train_{metric}'].append(train_metrics[metric])
            history[f'val_{metric}'].append(val_metrics[metric])

        # Print metrics
        print(f'Train - Loss: {train_loss:.4f}, Dice: {train_metrics["dice"]:.4f}, '
              f'IoU: {train_metrics["iou"]:.4f}, Acc: {train_metrics["accuracy"]:.4f}')
        print(f'Val   - Loss: {val_loss:.4f}, Dice: {val_metrics["dice"]:.4f}, '
              f'IoU: {val_metrics["iou"]:.4f}, Acc: {val_metrics["accuracy"]:.4f}')

        # Save best model
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            best_epoch = epoch + 1
            best_state = model.state_dict()
            print(f'✓ New best model (val dice: {best_val_dice:.4f})')

    # Evaluate on test set with best model
    print(f'\n{"="*80}')
    print(f'Evaluating best model from epoch {best_epoch}')
    print(f'{"="*80}\n')

    if best_state:
        model.load_state_dict(best_state)

    test_loss, test_metrics = evaluate(model, test_loader, loss_fn, threshold)

    print(f'Test Results:')
    print(f'  Loss: {test_loss:.4f}')
    for key, value in test_metrics.items():
        print(f'  {key.capitalize()}: {value:.4f}')

    # Save results
    history['test'] = test_metrics
    history['test_loss'] = test_loss
    history['best_epoch'] = best_epoch
    history['config'] = config

    # Save history
    history_path = exp_results_dir / 'history.json'
    save_history(history, str(history_path))

    # Save test metrics separately for easy access
    metrics_path = exp_results_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'test_metrics': test_metrics,
            'test_loss': test_loss,
            'best_epoch': best_epoch,
            'best_val_dice': best_val_dice
        }, f, indent=2)

    # Save best model checkpoint
    checkpoint_path = exp_results_dir / 'model_best.pth'
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'val_dice': best_val_dice,
        'config': config
    }, checkpoint_path)
    print(f"\nSaved model checkpoint to {checkpoint_path}")

    # Generate individual training plot
    plot_path = plots_dir / f'{exp_name}_training.png'
    plot_training_curves(history, str(plot_path), model_name=exp_name.replace('_', ' ').upper())

    return {
        'history': history,
        'test_metrics': test_metrics,
        'exp_name': exp_name
    }


def generate_comparison_plots(all_results, plots_dir):
    """Generate comparison plots across experiments."""
    print(f"\n{'='*80}")
    print("Generating Comparison Plots")
    print(f"{'='*80}\n")

    # CNN comparison (both datasets)
    cnn_results = {
        res['exp_name']: res['history'] 
        for res in all_results 
        if res['exp_name'].startswith('cnn_')
    }
    if len(cnn_results) == 2:
        plot_metric_comparison(
            cnn_results,
            metric='dice',
            save_path=str(plots_dir / 'cnn_comparison.png'),
            title='CNN Comparison'
        )

    # UNet comparison (both datasets, BCE only)
    unet_bce_results = {
        res['exp_name']: res['history'] 
        for res in all_results 
        if res['exp_name'].startswith('unet_') and res['exp_name'].endswith('_bce')
    }
    if len(unet_bce_results) == 2:
        plot_metric_comparison(
            unet_bce_results,
            metric='dice',
            save_path=str(plots_dir / 'unet_comparison.png'),
            title='UNet Comparison'
        )

    # Ablation on PH2
    ph2_ablation = {
        res['exp_name']: res['history'] 
        for res in all_results 
        if 'ph2' in res['exp_name'] and res['exp_name'].startswith('unet_')
    }
    if len(ph2_ablation) >= 3:
        plot_metric_comparison(
            ph2_ablation,
            metric='dice',
            save_path=str(plots_dir / 'ablation_ph2.png'),
            title='Loss Function Ablation on PH2'
        )

    # Ablation on DRIVE
    drive_ablation = {
        res['exp_name']: res['history'] 
        for res in all_results 
        if 'drive' in res['exp_name'] and res['exp_name'].startswith('unet_')
    }
    if len(drive_ablation) >= 3:
        plot_metric_comparison(
            drive_ablation,
            metric='dice',
            save_path=str(plots_dir / 'ablation_drive.png'),
            title='Loss Function Ablation on DRIVE'
        )

    # All models comparison
    all_histories = {res['exp_name']: res['history'] for res in all_results}
    plot_metric_comparison(
        all_histories,
        metric='dice',
        save_path=str(plots_dir / 'all_models_dice_comparison.png'),
        title='All Experiments: Dice Score Comparison'
    )

    # Test metrics bar chart
    test_results = {res['exp_name']: res['test_metrics'] for res in all_results}
    plot_test_metrics_comparison(
        test_results,
        save_path=str(plots_dir / 'all_test_metrics.png'),
        title='Test Metrics Comparison'
    )

    # Save results table
    save_results_table(
        test_results,
        save_path=str(plots_dir / 'results_summary')
    )


def main():
    parser = argparse.ArgumentParser(description='Run all segmentation')
    parser.add_argument('--results-dir', type=str, default='project3/results',
                       help='Directory to save results')
    parser.add_argument('--plots-dir', type=str, default='project3/plots',
                       help='Directory to save plots')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip experiments that already have results')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Define all experiments
    experiments = [
        # CNN baseline
        {'model': 'cnn', 'dataset': 'ph2', 'loss': 'bce', 'epochs': args.epochs, 'batch_size': args.batch_size},
        {'model': 'cnn', 'dataset': 'drive', 'loss': 'bce', 'epochs': args.epochs, 'batch_size': args.batch_size},

        # UNet baseline
        {'model': 'unet', 'dataset': 'ph2', 'loss': 'bce', 'epochs': args.epochs, 'batch_size': args.batch_size},
        {'model': 'unet', 'dataset': 'drive', 'loss': 'bce', 'epochs': args.epochs, 'batch_size': args.batch_size},

        # Ablation study
        {'model': 'unet', 'dataset': 'ph2', 'loss': 'weighted_bce', 'epochs': args.epochs, 'batch_size': args.batch_size},
        {'model': 'unet', 'dataset': 'drive', 'loss': 'weighted_bce', 'epochs': args.epochs, 'batch_size': args.batch_size},
        {'model': 'unet', 'dataset': 'ph2', 'loss': 'focal', 'epochs': args.epochs, 'batch_size': args.batch_size},
        {'model': 'unet', 'dataset': 'drive', 'loss': 'focal', 'epochs': args.epochs, 'batch_size': args.batch_size},
    ]

    print(f"Total experiments: {len(experiments)}")
    print(f"Results directory: {results_dir}")
    print(f"Plots directory: {plots_dir}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    all_results = []

    for i, config in enumerate(experiments, 1):
        exp_name = get_experiment_name(config['model'], config['dataset'], config['loss'])

        # Check if experiment already completed
        if args.skip_existing:
            exp_results_dir = results_dir / exp_name
            if (exp_results_dir / 'history.json').exists():
                print(f"\n[{i}/{len(experiments)}] Skipping {exp_name} (already exists)")
                # Load existing results
                with open(exp_results_dir / 'history.json', 'r') as f:
                    history = json.load(f)
                all_results.append({
                    'history': history,
                    'test_metrics': history['test'],
                    'exp_name': exp_name
                })
                continue

        print(f"\n[{i}/{len(experiments)}] Starting experiment: {exp_name}")

        try:
            result = run_single_experiment(config, results_dir, plots_dir)
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Error in experiment {exp_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Generate comparison plots
    if all_results:
        generate_comparison_plots(all_results, plots_dir)

    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {results_dir}")
    print(f"Plots saved to: {plots_dir}")
    print(f"\nCompleted {len(all_results)}/{len(experiments)} experiments successfully.")


if __name__ == '__main__':
    main()

# usage: python project3/run_experiments.py --epochs 30 --batch-size 4