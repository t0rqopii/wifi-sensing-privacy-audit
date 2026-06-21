import pytest
import numpy as np
from src.core.csi_generator import CSIGenerator

def test_csi_generator_initialization():
    gen = CSIGenerator(n_subcarriers=10, n_time_steps=100)
    assert gen.n_subcarriers == 10
    assert gen.n_time_steps == 100
    assert len(gen.activities) == 6

def test_generate_base_channel():
    gen = CSIGenerator(n_subcarriers=15, n_time_steps=50)
    h_base = gen.generate_base_channel()
    assert h_base.shape == (15, 50)
    assert np.iscomplexobj(h_base)

def test_generate_dataset():
    gen = CSIGenerator(n_subcarriers=10, n_time_steps=50)
    samples_per_class = 5
    X, y, classes = gen.generate_dataset(samples_per_class=samples_per_class)
    
    total_samples = 6 * samples_per_class
    assert X.shape == (total_samples, 10, 50)
    assert y.shape == (total_samples,)
    assert len(classes) == 6
    assert np.iscomplexobj(X)
