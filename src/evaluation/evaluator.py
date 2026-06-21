import numpy as np
from src.core.csi_generator import CSIGenerator
from src.core.signal_processor import SignalProcessor
from src.core.privacy_risk_analyzer import PrivacyRiskAnalyzer
from src.core.mitigation_engine import MitigationEngine
from src.utils.metrics import compute_snr, compute_channel_capacity
from src.utils.visualization import Visualizer

class Evaluator:
    """
    Orchestrates the full experimental pipeline to evaluate privacy vs utility tradeoff.
    """
    def __init__(self):
        self.visualizer = Visualizer(output_dir="output")
        
    def run_full_evaluation(self, samples_per_class=100):
        print(f"--- Starting Full Evaluation Pipeline ---")
        
        # 1. Generate Data
        print("Generating synthetic CSI dataset...")
        generator = CSIGenerator(n_subcarriers=30, n_time_steps=500, fs=1000)
        X_complex, y, classes = generator.generate_dataset(samples_per_class=samples_per_class)
        print(f"Data shape: {X_complex.shape}, Labels: {len(np.unique(y))} classes")
        
        # Plot clean heatmap (first sample of 'walking' activity)
        # Find a walking sample
        walking_idx = classes.index('walking')
        sample_idx = np.where(y == walking_idx)[0][0]
        clean_amp = np.abs(X_complex[sample_idx])
        self.visualizer.plot_csi_heatmap(clean_amp, "Clean CSI Amplitude (Walking)", "csi_heatmap_clean.png")

        # 2. Baseline Attack (Clean Data)
        print("\n--- Running Baseline Attack (No Mitigation) ---")
        sig_proc = SignalProcessor(fs=generator.fs)
        features_pca, amp_filt, phase_san = sig_proc.process_pipeline(X_complex, fit_pca=True)
        
        # Plot PCA scatter before mitigation
        self.visualizer.plot_pca_scatter(features_pca, y, classes, 
                                         "PCA Feature Space (Before Mitigation)", 
                                         "feature_space_pca_before.png")
                                         
        analyzer = PrivacyRiskAnalyzer()
        baseline_results = analyzer.evaluate_cross_val(features_pca, y)
        baseline_acc = baseline_results['accuracy']
        
        self.visualizer.plot_confusion_matrix(baseline_results['confusion_matrix'], classes,
                                              "Confusion Matrix (Baseline)", "confusion_matrix_before.png")

        # Calculate baseline utility
        # Signal power without added noise
        # We'll treat the filtered signal as the "true" signal for SNR purposes in mitigation
        baseline_capacity = compute_channel_capacity(snr_db=30) # Assume 30dB baseline SNR
        
        # 3. Apply Mitigation & Re-evaluate
        epsilons = [10.0, 5.0, 2.0, 1.0, 0.5, 0.1]
        alpha = 0.5 # Constant phase obfuscation for this experiment
        
        accuracies = []
        capacities = []
        
        mitigator = MitigationEngine()
        
        print("\n--- Running Mitigation Experiments ---")
        for eps in epsilons:
            print(f"\nEvaluating with DP Epsilon = {eps}")
            
            # Apply mitigation
            X_mitigated = mitigator.mitigate_complex_csi(X_complex, epsilon=eps, alpha=alpha)
            
            # Reprocess
            mit_features_pca, mit_amp, mit_phase = sig_proc.process_pipeline(X_mitigated, fit_pca=False)
            
            # Re-evaluate Attacker
            analyzer_mit = PrivacyRiskAnalyzer()
            mit_results = analyzer_mit.evaluate_cross_val(mit_features_pca, y)
            acc = mit_results['accuracy']
            accuracies.append(acc)
            
            # Utility (SNR and Capacity)
            # Compare first sample for SNR approximation
            orig_amp = np.abs(X_complex[0])
            new_amp = np.abs(X_mitigated[0])
            noise = new_amp - orig_amp
            snr_db = compute_snr(orig_amp, noise)
            cap = compute_channel_capacity(snr_db)
            capacities.append(cap)
            
            print(f"Accuracy: {acc:.4f}, SNR: {snr_db:.2f} dB, Capacity: {cap/1e6:.2f} Mbps")
            
            # Save artifacts for the epsilon=1.0 case as a representative example
            if eps == 1.0:
                self.visualizer.plot_csi_heatmap(new_amp, "Mitigated CSI Amplitude (Epsilon=1.0)", "csi_heatmap_mitigated.png")
                self.visualizer.plot_confusion_matrix(mit_results['confusion_matrix'], classes,
                                                      "Confusion Matrix (Epsilon=1.0)", "confusion_matrix_after.png")
                self.visualizer.plot_pca_scatter(mit_features_pca, y, classes, 
                                         "PCA Feature Space (Epsilon=1.0)", 
                                         "feature_space_pca_after.png")

        # 4. Tradeoff Visualization
        print("\n--- Generating Tradeoff Visualizations ---")
        self.visualizer.plot_accuracy_vs_epsilon(epsilons, accuracies, baseline_acc)
        self.visualizer.plot_privacy_utility_tradeoff(epsilons, accuracies, capacities, baseline_capacity)
        
        print("Evaluation pipeline completed successfully. Plots saved to 'output/' directory.")

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.run_full_evaluation(samples_per_class=50)
