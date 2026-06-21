import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class Visualizer:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def plot_csi_heatmap(self, amplitude_matrix, title, filename):
        """Plots a heatmap of the CSI amplitude for a single sample."""
        plt.figure(figsize=(10, 6))
        # Plot subcarriers vs time
        sns.heatmap(amplitude_matrix, cmap='viridis', cbar_kws={'label': 'Amplitude'})
        plt.title(title)
        plt.xlabel("Time Step")
        plt.ylabel("Subcarrier Index")
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Saved plot: {filepath}")

    def plot_accuracy_vs_epsilon(self, epsilons, accuracies, baseline_acc, filename="accuracy_vs_epsilon.png"):
        """Plots classification accuracy as a function of privacy budget ε."""
        plt.figure(figsize=(8, 5))
        plt.plot(epsilons, accuracies, marker='o', linestyle='-', color='b', label='Mitigated Accuracy')
        plt.axhline(y=baseline_acc, color='r', linestyle='--', label=f'Baseline Acc ({baseline_acc:.2f})')
        
        # Add a line for random guess if known (e.g. 1/6 for 6 classes)
        plt.axhline(y=1/6.0, color='gray', linestyle=':', label='Random Guess (~0.16)')
        
        plt.title("Attacker Accuracy vs. Differential Privacy Budget (ε)")
        plt.xlabel("Privacy Budget ε (Lower is more private)")
        plt.ylabel("Classification Accuracy")
        plt.xscale('log') # Log scale is often better for epsilon
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Saved plot: {filepath}")

    def plot_privacy_utility_tradeoff(self, epsilons, accuracies, capacities, baseline_cap, filename="privacy_utility_tradeoff.png"):
        """Plots dual-axis chart: Accuracy (Privacy loss) vs Capacity (Utility)."""
        fig, ax1 = plt.subplots(figsize=(9, 6))

        color = 'tab:red'
        ax1.set_xlabel('Privacy Budget ε (log scale)')
        ax1.set_ylabel('Attacker Accuracy', color=color)
        ax1.plot(epsilons, accuracies, color=color, marker='o', label='Accuracy (Privacy Risk)')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xscale('log')

        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Channel Capacity Ratio', color=color)  
        
        # Normalize capacity relative to baseline
        cap_ratio = [c / baseline_cap for c in capacities]
        ax2.plot(epsilons, cap_ratio, color=color, marker='s', linestyle='--', label='Utility (Capacity)')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title("Privacy vs. Utility Tradeoff")
        fig.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Saved plot: {filepath}")

    def plot_confusion_matrix(self, cm, classes, title, filename):
        """Plots a confusion matrix."""
        plt.figure(figsize=(8, 6))
        # Normalize to show percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Saved plot: {filepath}")
        
    def plot_pca_scatter(self, features_pca, labels, classes, title, filename):
        """Plots a 2D scatter of the first two PCA components."""
        plt.figure(figsize=(10, 8))
        
        for idx, class_name in enumerate(classes):
            mask = (labels == idx)
            plt.scatter(features_pca[mask, 0], features_pca[mask, 1], label=class_name, alpha=0.7)
            
        plt.title(title)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Saved plot: {filepath}")
