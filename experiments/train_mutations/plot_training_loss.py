import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import click
import os

def find_model_dirs(saved_data_dir):
    """Yield (program, loss_fn, job_id, model_dir) for each model directory containing train_accs.npy and train_losses.npy"""
    for test_results_file in Path(saved_data_dir).rglob("test_results.json"):
        model_dir = test_results_file.parent
        train_accs = model_dir / "train_accs.npy"
        train_losses = model_dir / "train_losses.npy"
        val_accs = model_dir / "val_accs.npy"
        val_losses = model_dir / "val_losses.npy"
        if train_accs.exists() and train_losses.exists() and val_accs.exists() and val_losses.exists():
            # Extract program, loss_fn, job_id from path
            try:
                job_id = model_dir.name
                loss_fn = model_dir.parent.name
                program = model_dir.parent.parent.name
                yield program, loss_fn, job_id, model_dir
            except Exception:
                continue

def plot_training_curves(train_accs, train_losses, output_file):
    training_steps = np.arange(1, len(train_accs) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot accuracy on ax1 (left y-axis)
    l1, = ax1.plot(training_steps, train_accs, label='Training Accuracy', linewidth=2)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0.0, 1.0)

    # Plot loss on ax2 (right y-axis)
    ax2 = ax1.twinx()
    l2, = ax2.plot(training_steps, train_losses, label='Training Loss', linewidth=2, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
    ax2.set_ylabel('Loss')
    ax2.set_ylim(np.min(train_losses), np.max(train_losses))

    # Combine legends from both axes
    lines = [l1, l2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=11, frameon=False)

    plt.title('Correctness Accuracy and Correctness Loss per Training Step')
    fig.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)

@click.command()
@click.option('--saved-data-dir', default='saved_data', help='Directory containing model results')
@click.option('--output-dir', default='training_loss_plots', help='Directory to save plots')
def main(saved_data_dir, output_dir):
    output_dir = Path(output_dir)
    for program, loss_fn, job_id, model_dir in find_model_dirs(saved_data_dir):
        train_accs = np.load(model_dir / 'train_accs.npy')
        train_losses = np.load(model_dir / 'train_losses.npy')
        out_path = output_dir / program / loss_fn
        out_path.mkdir(parents=True, exist_ok=True)
        plot_file = out_path / f'{job_id}.pdf'
        plot_training_curves(train_accs, train_losses, plot_file)
        print(f"Saved plot to {plot_file}")

if __name__ == '__main__':
    main() 