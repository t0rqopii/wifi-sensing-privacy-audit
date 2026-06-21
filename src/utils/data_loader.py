import numpy as np
import os

class DataLoader:
    """
    Handles data loading, saving, and parsing for CSI matrices.
    """
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
    def save_dataset(self, filename, X, y, activities):
        """Saves generated dataset to disk in .npz format."""
        filepath = os.path.join(self.data_dir, filename)
        np.savez_compressed(filepath, X=X, y=y, activities=activities)
        print(f"Dataset saved to {filepath}")
        
    def load_dataset(self, filename):
        """Loads dataset from disk."""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file {filepath} not found.")
            
        data = np.load(filepath, allow_pickle=True)
        X = data['X']
        y = data['y']
        activities = data['activities'].tolist()
        print(f"Loaded dataset from {filepath} with shape {X.shape}")
        return X, y, activities
        
    @staticmethod
    def get_amplitude_phase(X_complex):
        """Extracts amplitude and phase from complex CSI matrix."""
        amplitude = np.abs(X_complex)
        phase = np.angle(X_complex)
        return amplitude, phase

    @staticmethod
    def normalize_amplitude(amplitude):
        """Normalizes amplitude to [0, 1] range globally or per-sample."""
        # Normalize per sample
        min_vals = amplitude.min(axis=(1, 2), keepdims=True)
        max_vals = amplitude.max(axis=(1, 2), keepdims=True)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1e-10
        
        normalized = (amplitude - min_vals) / range_vals
        return normalized
