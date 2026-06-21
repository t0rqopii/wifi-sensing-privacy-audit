import pytest
import numpy as np
from src.core.mitigation_engine import MitigationEngine

def test_compute_sensitivity():
    mitigator = MitigationEngine()
    data = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    sensitivity = mitigator.compute_sensitivity(data)
    assert sensitivity == 3.0

def test_apply_dp_noise():
    mitigator = MitigationEngine()
    amplitude = np.random.rand(2, 5, 10)
    epsilon = 1.0
    
    mitigated = mitigator.apply_dp_noise(amplitude, epsilon)
    assert mitigated.shape == amplitude.shape
    assert not np.array_equal(mitigated, amplitude)
    # Ensure clipping works (all values >= 0)
    assert np.all(mitigated >= 0)

def test_apply_phase_obfuscation():
    mitigator = MitigationEngine()
    phase = np.zeros((2, 5, 10))
    alpha = 0.5
    
    mitigated = mitigator.apply_phase_obfuscation(phase, alpha)
    assert mitigated.shape == phase.shape
    assert not np.array_equal(mitigated, phase)
    # Phase must be wrapped between -pi and pi
    assert np.all(mitigated >= -np.pi)
    assert np.all(mitigated <= np.pi)

def test_mitigate_complex_csi():
    mitigator = MitigationEngine()
    # Complex array with amp 1.0 and phase 0.0
    X_complex = np.ones((2, 5, 10), dtype=complex)
    
    mitigated = mitigator.mitigate_complex_csi(X_complex, epsilon=1.0, alpha=0.5)
    assert mitigated.shape == X_complex.shape
    assert np.iscomplexobj(mitigated)
    assert not np.array_equal(mitigated, X_complex)
