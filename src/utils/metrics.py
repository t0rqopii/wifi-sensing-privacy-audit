import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_snr(signal, noise):
    """
    Computes Signal-to-Noise Ratio (SNR) in dB.
    signal: Original unmitigated CSI signal
    noise: The DP noise added (mitigated - original)
    """
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = np.mean(np.abs(noise)**2)
    
    if noise_power == 0:
        return float('inf')
        
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db

def compute_channel_capacity(snr_db, bandwidth=20e6):
    """
    Estimates Shannon channel capacity C = B * log2(1 + SNR_linear).
    Bandwidth default is 20 MHz (typical Wi-Fi channel).
    """
    if snr_db == float('inf'):
        return float('inf')
        
    snr_linear = 10 ** (snr_db / 10.0)
    capacity = bandwidth * np.log2(1 + snr_linear)
    return capacity

def evaluate_classification(y_true, y_pred):
    """
    Computes standard classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }
