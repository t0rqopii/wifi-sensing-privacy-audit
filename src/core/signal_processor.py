import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
import warnings

class SignalProcessor:
    """
    Handles preprocessing of CSI signals: filtering, sanitization, and feature extraction.
    """
    def __init__(self, fs=1000, n_components=20):
        self.fs = fs
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.is_pca_fitted = False

    def butter_bandpass(self, lowcut, highcut, order=3):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def apply_bandpass_filter(self, data, lowcut=1.0, highcut=50.0):
        """
        Applies a Butterworth bandpass filter to isolate human-motion frequencies.
        data shape: (n_samples, n_subcarriers, n_time_steps)
        """
        b, a = self.butter_bandpass(lowcut, highcut)
        # Apply filter along the time axis (-1)
        filtered_data = filtfilt(b, a, data, axis=-1)
        return filtered_data

    def hampel_filter(self, data, window_size=5, n_sigmas=3):
        """
        Applies a simplified Hampel filter for outlier removal along the time axis.
        """
        # For performance, we'll do a simple moving median threshold
        # on the last dimension
        filtered = data.copy()
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                series = data[i, j, :]
                # Use a fast approximation or scipy median_filter
                from scipy.ndimage import median_filter
                rolling_median = median_filter(series, size=window_size)
                # Calculate MAD
                mad = median_filter(np.abs(series - rolling_median), size=window_size)
                threshold = n_sigmas * 1.4826 * mad
                
                # Replace outliers with rolling median
                outliers = np.abs(series - rolling_median) > threshold
                filtered[i, j, outliers] = rolling_median[outliers]
                
        return filtered

    def sanitize_phase(self, phase_data):
        """
        Unwraps phase and applies linear regression to remove phase offset drift.
        phase_data shape: (n_samples, n_subcarriers, n_time_steps)
        """
        # Unwrap phase along the time axis
        unwrapped = np.unwrap(phase_data, axis=-1)
        
        sanitized = np.zeros_like(unwrapped)
        n_samples, n_sub, n_time = unwrapped.shape
        t = np.arange(n_time)
        
        for i in range(n_samples):
            for j in range(n_sub):
                series = unwrapped[i, j, :]
                # Simple linear detrending
                slope, intercept = np.polyfit(t, series, 1)
                sanitized[i, j, :] = series - (slope * t + intercept)
                
        return sanitized

    def extract_features(self, amplitude, phase=None):
        """
        Extracts statistical features from amplitude (and optionally phase).
        Returns a flat feature vector per sample.
        shape out: (n_samples, n_features)
        """
        n_samples = amplitude.shape[0]
        features = []
        
        for i in range(n_samples):
            sample_features = []
            
            # Amplitude statistics per subcarrier
            amp_mean = np.mean(amplitude[i], axis=-1)
            amp_var = np.var(amplitude[i], axis=-1)
            
            sample_features.extend(amp_mean)
            sample_features.extend(amp_var)
            
            if phase is not None:
                phase_mean = np.mean(phase[i], axis=-1)
                phase_var = np.var(phase[i], axis=-1)
                sample_features.extend(phase_mean)
                sample_features.extend(phase_var)
                
            features.append(sample_features)
            
        return np.array(features)

    def fit_transform_pca(self, features):
        """Fits PCA and transforms features."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reduced = self.pca.fit_transform(features)
        self.is_pca_fitted = True
        return reduced

    def transform_pca(self, features):
        """Transforms features using already fitted PCA."""
        if not self.is_pca_fitted:
            raise ValueError("PCA has not been fitted yet.")
        return self.pca.transform(features)

    def process_pipeline(self, X_complex, fit_pca=True):
        """
        Runs the full preprocessing pipeline on raw complex CSI data.
        """
        amplitude = np.abs(X_complex)
        phase = np.angle(X_complex)
        
        # 1. Filter amplitude
        amp_filtered = self.apply_bandpass_filter(amplitude)
        amp_filtered = self.hampel_filter(amp_filtered)
        
        # 2. Sanitize phase
        phase_sanitized = self.sanitize_phase(phase)
        
        # 3. Extract features
        features = self.extract_features(amp_filtered, phase_sanitized)
        
        # 4. PCA
        if fit_pca:
            features_pca = self.fit_transform_pca(features)
        else:
            features_pca = self.transform_pca(features)
            
        return features_pca, amp_filtered, phase_sanitized
