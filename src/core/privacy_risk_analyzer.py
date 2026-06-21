import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

class PrivacyRiskAnalyzer:
    """
    Simulates the unauthorized eavesdropper (attacker) using ML to extract activities from CSI.
    Uses an SVM classifier on extracted PCA features.
    """
    def __init__(self, C=1.0, kernel='rbf'):
        self.classifier = SVC(C=C, kernel=kernel, probability=True, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def train(self, X_train, y_train):
        """
        Trains the SVM classifier.
        X_train: shape (n_samples, n_features)
        y_train: shape (n_samples,)
        """
        X_scaled = self.scaler.fit_transform(X_train)
        self.classifier.fit(X_scaled, y_train)
        self.is_fitted = True

    def predict(self, X_test):
        """Predicts labels for test data."""
        if not self.is_fitted:
            raise ValueError("Analyzer is not fitted yet.")
        X_scaled = self.scaler.transform(X_test)
        return self.classifier.predict(X_scaled)

    def evaluate_cross_val(self, X, y, n_splits=5):
        """
        Evaluates the attacker's capability using Stratified K-Fold CV.
        Returns average accuracy and aggregated true/pred labels for confusion matrix.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        accuracies = []
        y_true_all = []
        y_pred_all = []
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale per fold to avoid data leakage
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            clf = SVC(kernel='rbf', probability=True, random_state=42)
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
            
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            
        mean_acc = np.mean(accuracies)
        print(f"Cross-Validation Mean Accuracy: {mean_acc:.4f} (+/- {np.std(accuracies):.4f})")
        
        report = classification_report(y_true_all, y_pred_all, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true_all, y_pred_all)
        
        return {
            'accuracy': mean_acc,
            'report': report,
            'confusion_matrix': cm,
            'y_true': y_true_all,
            'y_pred': y_pred_all
        }
