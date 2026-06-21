import pytest
import numpy as np
from src.core.privacy_risk_analyzer import PrivacyRiskAnalyzer

def test_analyzer_training():
    analyzer = PrivacyRiskAnalyzer()
    X_train = np.random.rand(20, 10)
    y_train = np.random.randint(0, 3, 20)
    
    analyzer.train(X_train, y_train)
    assert analyzer.is_fitted

def test_analyzer_prediction():
    analyzer = PrivacyRiskAnalyzer()
    X_train = np.random.rand(20, 10)
    y_train = np.random.randint(0, 3, 20)
    
    analyzer.train(X_train, y_train)
    
    X_test = np.random.rand(5, 10)
    y_pred = analyzer.predict(X_test)
    
    assert y_pred.shape == (5,)
    assert set(y_pred).issubset({0, 1, 2})

def test_evaluate_cross_val():
    analyzer = PrivacyRiskAnalyzer()
    # Need enough samples for 3-fold CV (changed to 3 for fast test)
    X = np.random.rand(30, 5)
    y = np.array([0]*10 + [1]*10 + [2]*10)
    
    results = analyzer.evaluate_cross_val(X, y, n_splits=3)
    
    assert 'accuracy' in results
    assert 'confusion_matrix' in results
    assert 0.0 <= results['accuracy'] <= 1.0
