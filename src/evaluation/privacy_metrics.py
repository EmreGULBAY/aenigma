
"""
Privacy evaluation metrics for synthetic data.
Includes discriminative score, TSTR, TRTS, and distance-based metrics.
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class DiscriminativeModel(nn.Module):
    """
    Simple discriminative model to distinguish real from synthetic data.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(DiscriminativeModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


class PrivacyMetrics:
    """
    Evaluate privacy of synthetic data using various metrics.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize privacy metrics evaluator.
        
        Args:
            device: Device to run on
        """
        self.device = device
    
    def discriminative_score(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        epochs: int = 100,
        batch_size: int = 128
    ) -> Dict[str, float]:
        """
        Train a discriminator to distinguish real from synthetic data.
        Score close to 0.5 indicates good privacy (indistinguishable).
        
        Args:
            real_data: Real sequences (n_samples, seq_len, n_features)
            synthetic_data: Synthetic sequences (n_samples, seq_len, n_features)
            epochs: Training epochs
            batch_size: Batch size
            
        Returns:
            Dictionary with accuracy, precision, recall, f1, auc
        """
        logger.info("Computing discriminative score...")
        
        real_flat = real_data.reshape(len(real_data), -1)
        synthetic_flat = synthetic_data.reshape(len(synthetic_data), -1)
        
        X = np.vstack([real_flat, synthetic_flat])
        y = np.hstack([np.ones(len(real_flat)), np.zeros(len(synthetic_flat))])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(self.device)
        
        input_dim = X_train.shape[1]
        model = DiscriminativeModel(input_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        model.train()
        for epoch in range(epochs):
            indices = torch.randperm(len(X_train_tensor))
            
            for i in range(0, len(X_train_tensor), batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = X_train_tensor[batch_indices]
                y_batch = y_train_tensor[batch_indices]
                
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            y_pred_proba = model(X_test_tensor).cpu().numpy()
            y_pred = (y_pred_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', zero_division=0
        )
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.5
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        logger.info(f"Discriminative Score - Accuracy: {accuracy:.4f} (target: ~0.5)")
        logger.info(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        return results
    
    def tstr_evaluation(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        real_labels: np.ndarray,
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Train on Synthetic, Test on Real (TSTR).
        Measures utility: how well a model trained on synthetic data
        performs on real data.
        
        Args:
            real_data: Real sequences (n_samples, seq_len, n_features)
            synthetic_data: Synthetic sequences (n_samples, seq_len, n_features)
            real_labels: Labels for real data (e.g., hotel_id or occupancy_category)
            test_size: Proportion of real data to use for testing
            
        Returns:
            Dictionary with accuracy and other metrics
        """
        logger.info("Computing TSTR (Train on Synthetic, Test on Real)...")
        
        real_flat = real_data.reshape(len(real_data), -1)
        synthetic_flat = synthetic_data.reshape(len(synthetic_data), -1)
        
        _, X_test_real, _, y_test_real = train_test_split(
            real_flat, real_labels, test_size=test_size, random_state=42
        )
        
        unique_labels, counts = np.unique(real_labels, return_counts=True)
        synthetic_labels = np.random.choice(
            unique_labels,
            size=len(synthetic_data),
            p=counts / counts.sum()
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(synthetic_flat, synthetic_labels)
        
        y_pred = model.predict(X_test_real)
        
        accuracy = accuracy_score(y_test_real, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_real, y_pred, average='weighted', zero_division=0
        )
        
        results = {
            'tstr_accuracy': accuracy,
            'tstr_precision': precision,
            'tstr_recall': recall,
            'tstr_f1': f1
        }
        
        logger.info(f"TSTR - Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return results
    
    def trts_evaluation(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        real_labels: np.ndarray,
        train_size: float = 0.8
    ) -> Dict[str, float]:
        """
        Train on Real, Test on Synthetic (TRTS).
        Measures how well synthetic data represents real data distribution.
        
        Args:
            real_data: Real sequences (n_samples, seq_len, n_features)
            synthetic_data: Synthetic sequences (n_samples, seq_len, n_features)
            real_labels: Labels for real data
            train_size: Proportion of real data to use for training
            
        Returns:
            Dictionary with accuracy and other metrics
        """
        logger.info("Computing TRTS (Train on Real, Test on Synthetic)...")
        
        real_flat = real_data.reshape(len(real_data), -1)
        synthetic_flat = synthetic_data.reshape(len(synthetic_data), -1)
        
        X_train_real, _, y_train_real, _ = train_test_split(
            real_flat, real_labels, train_size=train_size, random_state=42
        )
        
        unique_labels, counts = np.unique(real_labels, return_counts=True)
        synthetic_labels = np.random.choice(
            unique_labels,
            size=len(synthetic_data),
            p=counts / counts.sum()
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train_real, y_train_real)
        
        y_pred = model.predict(synthetic_flat)
        
        accuracy = accuracy_score(synthetic_labels, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            synthetic_labels, y_pred, average='weighted', zero_division=0
        )
        
        results = {
            'trts_accuracy': accuracy,
            'trts_precision': precision,
            'trts_recall': recall,
            'trts_f1': f1
        }
        
        logger.info(f"TRTS - Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return results
    
    def tstr_trts_comparison(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        real_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compare TSTR and TRTS scores.
        Ideally, TSTR ≈ TRTS indicates good synthetic data quality.
        
        Args:
            real_data: Real sequences
            synthetic_data: Synthetic sequences
            real_labels: Labels for real data
            
        Returns:
            Combined dictionary with both TSTR and TRTS metrics
        """
        tstr_results = self.tstr_evaluation(real_data, synthetic_data, real_labels)
        trts_results = self.trts_evaluation(real_data, synthetic_data, real_labels)
        
        diff = abs(tstr_results['tstr_accuracy'] - trts_results['trts_accuracy'])
        
        results = {**tstr_results, **trts_results}
        results['tstr_trts_diff'] = diff
        
        logger.info(f"TSTR-TRTS Difference: {diff:.4f} (lower is better)")
        
        return results
    
    def distance_to_closest_record(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        n_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Compute Distance to Closest Record (DCR).
        Measures privacy risk: if synthetic data is too close to real data,
        it might reveal sensitive information.
        
        Args:
            real_data: Real sequences (n_samples, seq_len, n_features)
            synthetic_data: Synthetic sequences (n_samples, seq_len, n_features)
            n_samples: Number of samples to evaluate (for efficiency)
            
        Returns:
            Dictionary with mean, median, min DCR
        """
        logger.info("Computing Distance to Closest Record (DCR)...")
        
        real_flat = real_data.reshape(len(real_data), -1)
        synthetic_flat = synthetic_data.reshape(len(synthetic_data), -1)
        
        if len(synthetic_flat) > n_samples:
            indices = np.random.choice(len(synthetic_flat), n_samples, replace=False)
            synthetic_sample = synthetic_flat[indices]
        else:
            synthetic_sample = synthetic_flat
        
        distances = []
        for syn_record in synthetic_sample:
            dists = np.linalg.norm(real_flat - syn_record, axis=1)
            min_dist = np.min(dists)
            distances.append(min_dist)
        
        distances = np.array(distances)
        
        results = {
            'dcr_mean': np.mean(distances),
            'dcr_median': np.median(distances),
            'dcr_min': np.min(distances),
            'dcr_max': np.max(distances),
            'dcr_std': np.std(distances)
        }
        
        logger.info(f"DCR - Mean: {results['dcr_mean']:.4f}, Median: {results['dcr_median']:.4f}")
        logger.info(f"  Min: {results['dcr_min']:.4f}, Max: {results['dcr_max']:.4f}")
        
        return results
    
    def evaluate_all(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        real_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Run all privacy evaluations.
        
        Args:
            real_data: Real sequences
            synthetic_data: Synthetic sequences
            real_labels: Labels for real data
            
        Returns:
            Combined dictionary with all metrics
        """
        logger.info("=" * 50)
        logger.info("Running Privacy Evaluation")
        logger.info("=" * 50)
        
        results = {}
        
        disc_results = self.discriminative_score(real_data, synthetic_data)
        results.update(disc_results)
        
        tstr_trts_results = self.tstr_trts_comparison(
            real_data, synthetic_data, real_labels
        )
        results.update(tstr_trts_results)
        
        dcr_results = self.distance_to_closest_record(real_data, synthetic_data)
        results.update(dcr_results)
        
        logger.info("=" * 50)
        logger.info("Privacy Evaluation Complete")
        logger.info("=" * 50)
        
        return results