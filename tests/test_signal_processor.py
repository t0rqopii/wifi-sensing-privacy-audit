import pytest
import numpy as np
from src.core.signal_processor import SignalProcessor

def test_butter_bandpass_filter():
    processor = SignalProcessor(fs=1000)
    # Create a 5Hz sine wave (signal) + 100Hz sine wave (noise)
    t = np.arange(1000) / 1000
    signal = np.sin(2 * np.pi * 5 * t)
    noise = np.sin(2 * np.pi * 100 * t)
    data = (signal + noise).reshape(1, 1, 1000)
    
    filtered = processor.apply_bandpass_filter(data, lowcut=1.0, highcut=50.0)
    
    # 100Hz noise should be heavily attenuated
    assert np.var(filtered) < np.var(data)
    assert filtered.shape == data.shape

def test_extract_features():
    processor = SignalProcessor()
    amplitude = np.random.rand(5, 10, 100)
    phase = np.random.rand(5, 10, 100)
    
    features = processor.extract_features(amplitude, phase)
    # 10 subcarriers * 2 (mean, var) for amp + 10 * 2 for phase = 40 features
    assert features.shape == (5, 40)

def test_process_pipeline():
    processor = SignalProcessor(n_components=5)
    X_complex = np.random.randn(10, 15, 100) + 1j * np.random.randn(10, 15, 100)
    
    features_pca, amp_filt, phase_san = processor.process_pipeline(X_complex, fit_pca=True)
    
    assert features_pca.shape == (10, 5)
    assert amp_filt.shape == X_complex.shape
    assert phase_san.shape == X_complex.shape
    assert processor.is_pca_fitted
