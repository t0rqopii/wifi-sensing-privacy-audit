import numpy as np

class CSIGenerator:
    """
    Generates synthetic CSI data simulating human activities affecting Wi-Fi channels,
    modeled after the Widar3.0 dataset format.
    """
    def __init__(self, n_subcarriers=30, n_time_steps=500, fs=1000):
        self.n_subcarriers = n_subcarriers
        self.n_time_steps = n_time_steps
        self.fs = fs  # sampling frequency
        
        # Activity classes
        self.activities = ['empty', 'walking', 'sitting_down', 'standing_up', 'waving', 'falling']
        
    def generate_base_channel(self):
        """Generates a base Rayleigh fading channel with multipath."""
        # Simple multipath model
        h_base = np.random.randn(self.n_subcarriers, self.n_time_steps) + \
                 1j * np.random.randn(self.n_subcarriers, self.n_time_steps)
        h_base *= 0.5  # Scale
        return h_base
        
    def add_activity_perturbation(self, h_base, activity):
        """Injects activity-specific amplitude and phase perturbations."""
        t = np.arange(self.n_time_steps) / self.fs
        perturbation = np.zeros_like(h_base, dtype=complex)
        
        if activity == 'empty':
            # Low frequency environmental noise
            freq = 0.5
            amp = 0.05
            for i in range(self.n_subcarriers):
                perturbation[i, :] = amp * np.exp(1j * 2 * np.pi * freq * t + np.random.rand() * 2 * np.pi)
                
        elif activity == 'walking':
            # Doppler shift roughly 10-30 Hz
            freq = np.random.uniform(10, 30)
            amp = 0.3
            for i in range(self.n_subcarriers):
                perturbation[i, :] = amp * np.exp(1j * 2 * np.pi * freq * t + np.random.rand() * 2 * np.pi)
                
        elif activity == 'sitting_down':
            # Transient lower frequency shift 5-15 Hz
            freq = np.random.uniform(5, 15)
            amp = 0.4
            envelope = np.exp(-((t - 0.25)**2) / 0.02)
            for i in range(self.n_subcarriers):
                perturbation[i, :] = amp * envelope * np.exp(1j * 2 * np.pi * freq * t + np.random.rand() * 2 * np.pi)
                
        elif activity == 'standing_up':
            # Transient frequency shift 5-20 Hz
            freq = np.random.uniform(5, 20)
            amp = 0.45
            envelope = np.exp(-((t - 0.25)**2) / 0.02)
            for i in range(self.n_subcarriers):
                perturbation[i, :] = amp * envelope * np.exp(1j * 2 * np.pi * freq * t + np.random.rand() * 2 * np.pi)
                
        elif activity == 'waving':
            # Periodic movement 2-5 Hz
            freq = np.random.uniform(2, 5)
            amp = 0.2
            for i in range(self.n_subcarriers):
                perturbation[i, :] = amp * np.sin(2 * np.pi * freq * t) * np.exp(1j * 2 * np.pi * freq * t + np.random.rand() * 2 * np.pi)
                
        elif activity == 'falling':
            # Sudden high frequency burst 20-40 Hz
            freq = np.random.uniform(20, 40)
            amp = 0.6
            envelope = np.exp(-((t - 0.25)**2) / 0.005)
            for i in range(self.n_subcarriers):
                perturbation[i, :] = amp * envelope * np.exp(1j * 2 * np.pi * freq * t + np.random.rand() * 2 * np.pi)
                
        return h_base + perturbation
        
    def generate_dataset(self, samples_per_class=50):
        """Generates a full synthetic dataset."""
        X_complex = []
        y = []
        
        for idx, activity in enumerate(self.activities):
            for _ in range(samples_per_class):
                h_base = self.generate_base_channel()
                h_perturbed = self.add_activity_perturbation(h_base, activity)
                
                # Reshape to (n_subcarriers, n_time_steps)
                X_complex.append(h_perturbed)
                y.append(idx)
                
        X_complex = np.array(X_complex)
        y = np.array(y)
        
        # We'll save the complex matrix directly or separate into amp/phase
        # Shape: (n_samples, n_subcarriers, n_time_steps)
        return X_complex, y, self.activities

if __name__ == "__main__":
    generator = CSIGenerator(n_subcarriers=30, n_time_steps=500, fs=1000)
    X, y, labels = generator.generate_dataset(samples_per_class=10)
    print(f"Generated data shape: {X.shape}, labels shape: {y.shape}")
