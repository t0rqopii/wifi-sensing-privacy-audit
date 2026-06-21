import numpy as np

class MitigationEngine:
    """
    Implements privacy-preserving algorithms for CSI data:
    1. Differential Privacy (DP) Noise Injection via Laplace Mechanism
    2. Phase Obfuscation
    """
    def __init__(self):
        pass

    def compute_sensitivity(self, data):
        """
        Computes the global sensitivity Δf of the CSI amplitude matrix.
        We estimate this as the maximum difference between any two samples in the batch.
        In a real DP scenario, this would be bounded by a theoretical maximum.
        """
        # Find the max and min across all samples, subcarriers, and time steps
        max_val = np.max(data)
        min_val = np.min(data)
        return np.abs(max_val - min_val)

    def apply_dp_noise(self, amplitude, epsilon=1.0):
        """
        Injects calibrated Laplace noise to satisfy ε-Differential Privacy.
        amplitude: shape (n_samples, n_subcarriers, n_time_steps)
        epsilon: Privacy budget. Lower ε -> More noise -> Higher privacy.
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be strictly positive (> 0)")
            
        sensitivity = self.compute_sensitivity(amplitude)
        
        # Scale of Laplace noise (b = Δf / ε)
        scale = sensitivity / epsilon
        
        # Generate Laplacian noise
        noise = np.random.laplace(loc=0.0, scale=scale, size=amplitude.shape)
        
        # Add noise to original amplitude
        mitigated_amplitude = amplitude + noise
        
        # Clip to ensure non-negative amplitude (optional but realistic for physical signals)
        mitigated_amplitude = np.clip(mitigated_amplitude, 0, None)
        
        return mitigated_amplitude

    def apply_phase_obfuscation(self, phase, alpha=0.5):
        """
        Injects uniformly random phase perturbations.
        phase: shape (n_samples, n_subcarriers, n_time_steps)
        alpha: Strength parameter in [0, 1]. α=1 means full random phase [-π, π].
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0")
            
        # Generate uniform random phase noise: U(-α*π, α*π)
        noise = np.random.uniform(low=-alpha * np.pi, high=alpha * np.pi, size=phase.shape)
        
        mitigated_phase = phase + noise
        
        # Wrap phase back to [-π, π]
        mitigated_phase = (mitigated_phase + np.pi) % (2 * np.pi) - np.pi
        
        return mitigated_phase

    def mitigate_complex_csi(self, X_complex, epsilon=1.0, alpha=0.5):
        """
        Applies both DP noise to amplitude and obfuscation to phase,
        reconstructing the complex CSI matrix.
        """
        amplitude = np.abs(X_complex)
        phase = np.angle(X_complex)
        
        mitigated_amp = self.apply_dp_noise(amplitude, epsilon)
        mitigated_phase = self.apply_phase_obfuscation(phase, alpha)
        
        # Reconstruct complex matrix: A * e^(j * theta)
        mitigated_complex = mitigated_amp * np.exp(1j * mitigated_phase)
        
        return mitigated_complex
