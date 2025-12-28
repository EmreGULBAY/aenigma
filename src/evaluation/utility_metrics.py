
"""
Utility evaluation metrics for synthetic data.
Measures statistical similarity and usefulness of synthetic data.
"""
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class UtilityMetrics:
    """
    Evaluate utility of synthetic data through statistical tests.
    """
    
    def __init__(self):
        """Initialize utility metrics evaluator."""
        pass
    
    def basic_statistics(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        feature_names: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare basic statistics (mean, std, min, max) between real and synthetic.
        
        Args:
            real_data: Real sequences (n_samples, seq_len, n_features)
            synthetic_data: Synthetic sequences (n_samples, seq_len, n_features)
            feature_names: Names of features
            
        Returns:
            Dictionary with statistics for each feature
        """
        logger.info("Computing basic statistics...")
        
        n_features = real_data.shape[2]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        results = {}
        
        for i, feature_name in enumerate(feature_names):
            real_feature = real_data[:, :, i].flatten()
            synthetic_feature = synthetic_data[:, :, i].flatten()
            
            results[feature_name] = {
                'real_mean': np.mean(real_feature),
                'synthetic_mean': np.mean(synthetic_feature),
                'real_std': np.std(real_feature),
                'synthetic_std': np.std(synthetic_feature),
                'real_min': np.min(real_feature),
                'synthetic_min': np.min(synthetic_feature),
                'real_max': np.max(real_feature),
                'synthetic_max': np.max(synthetic_feature),
                'mean_diff': abs(np.mean(real_feature) - np.mean(synthetic_feature)),
                'std_diff': abs(np.std(real_feature) - np.std(synthetic_feature))
            }
        
        mean_diffs = [v['mean_diff'] for v in results.values()]
        std_diffs = [v['std_diff'] for v in results.values()]
        
        logger.info(f"Average mean difference: {np.mean(mean_diffs):.4f}")
        logger.info(f"Average std difference: {np.mean(std_diffs):.4f}")
        
        return results
    
    def kolmogorov_smirnov_test(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        feature_names: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Kolmogorov-Smirnov test for distribution similarity.
        p-value > 0.05 indicates distributions are similar.
        
        Args:
            real_data: Real sequences
            synthetic_data: Synthetic sequences
            feature_names: Names of features
            
        Returns:
            Dictionary with KS statistics and p-values
        """
        logger.info("Computing Kolmogorov-Smirnov test...")
        
        n_features = real_data.shape[2]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        results = {}
        
        for i, feature_name in enumerate(feature_names):
            real_feature = real_data[:, :, i].flatten()
            synthetic_feature = synthetic_data[:, :, i].flatten()
            
            statistic, pvalue = stats.ks_2samp(real_feature, synthetic_feature)
            
            results[feature_name] = {
                'ks_statistic': statistic,
                'ks_pvalue': pvalue,
                'similar': pvalue > 0.05
            }
        
        similar_count = sum(1 for v in results.values() if v['similar'])
        logger.info(f"Features with similar distributions: {similar_count}/{n_features}")
        
        return results
    
    def wasserstein_distance(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        feature_names: List[str] = None
    ) -> Dict[str, float]:
        """
        Compute Wasserstein distance (Earth Mover's Distance).
        Lower values indicate more similar distributions.
        
        Args:
            real_data: Real sequences
            synthetic_data: Synthetic sequences
            feature_names: Names of features
            
        Returns:
            Dictionary with Wasserstein distances
        """
        logger.info("Computing Wasserstein distance...")
        
        n_features = real_data.shape[2]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        results = {}
        
        for i, feature_name in enumerate(feature_names):
            real_feature = real_data[:, :, i].flatten()
            synthetic_feature = synthetic_data[:, :, i].flatten()
            
            distance = stats.wasserstein_distance(real_feature, synthetic_feature)
            results[feature_name] = distance
        
        avg_distance = np.mean(list(results.values()))
        logger.info(f"Average Wasserstein distance: {avg_distance:.4f}")
        
        return results
    
    def correlation_matrix_comparison(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray
    ) -> Dict[str, float]:
        """
        Compare correlation matrices between real and synthetic data.
        
        Args:
            real_data: Real sequences
            synthetic_data: Synthetic sequences
            
        Returns:
            Dictionary with correlation metrics
        """
        logger.info("Comparing correlation matrices...")
        
        real_flat = real_data.reshape(-1, real_data.shape[2])
        synthetic_flat = synthetic_data.reshape(-1, synthetic_data.shape[2])
        
        real_corr = np.corrcoef(real_flat.T)
        synthetic_corr = np.corrcoef(synthetic_flat.T)
        
        corr_diff = np.abs(real_corr - synthetic_corr)
        
        results = {
            'corr_mean_diff': np.mean(corr_diff),
            'corr_max_diff': np.max(corr_diff),
            'corr_frobenius_norm': np.linalg.norm(corr_diff, 'fro')
        }
        
        logger.info(f"Correlation mean difference: {results['corr_mean_diff']:.4f}")
        logger.info(f"Correlation max difference: {results['corr_max_diff']:.4f}")
        
        return results
    
    def autocorrelation_comparison(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        max_lag: int = 10
    ) -> Dict[str, float]:
        """
        Compare autocorrelation for temporal consistency.
        
        Args:
            real_data: Real sequences (n_samples, seq_len, n_features)
            synthetic_data: Synthetic sequences
            max_lag: Maximum lag to compute
            
        Returns:
            Dictionary with autocorrelation metrics
        """
        logger.info("Comparing autocorrelation...")
        
        n_features = real_data.shape[2]
        
        autocorr_diffs = []
        
        for i in range(n_features):
            real_feature = real_data[:, :, i].flatten()
            synthetic_feature = synthetic_data[:, :, i].flatten()
            
            for lag in range(1, max_lag + 1):
                if len(real_feature) > lag:
                    real_autocorr = np.corrcoef(
                        real_feature[:-lag], real_feature[lag:]
                    )[0, 1]
                    synthetic_autocorr = np.corrcoef(
                        synthetic_feature[:-lag], synthetic_feature[lag:]
                    )[0, 1]
                    
                    if not (np.isnan(real_autocorr) or np.isnan(synthetic_autocorr)):
                        autocorr_diffs.append(abs(real_autocorr - synthetic_autocorr))
        
        results = {
            'autocorr_mean_diff': np.mean(autocorr_diffs) if autocorr_diffs else 0,
            'autocorr_max_diff': np.max(autocorr_diffs) if autocorr_diffs else 0
        }
        
        logger.info(f"Autocorrelation mean difference: {results['autocorr_mean_diff']:.4f}")
        
        return results
    
    def evaluate_all(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        feature_names: List[str] = None
    ) -> Dict:
        """
        Run all utility evaluations.
        
        Args:
            real_data: Real sequences
            synthetic_data: Synthetic sequences
            feature_names: Names of features
            
        Returns:
            Combined dictionary with all metrics
        """
        logger.info("=" * 50)
        logger.info("Running Utility Evaluation")
        logger.info("=" * 50)
        
        results = {}
        
        results['basic_stats'] = self.basic_statistics(
            real_data, synthetic_data, feature_names
        )
        
        results['ks_test'] = self.kolmogorov_smirnov_test(
            real_data, synthetic_data, feature_names
        )
        
        results['wasserstein'] = self.wasserstein_distance(
            real_data, synthetic_data, feature_names
        )
        
        results['correlation'] = self.correlation_matrix_comparison(
            real_data, synthetic_data
        )
        
        results['autocorrelation'] = self.autocorrelation_comparison(
            real_data, synthetic_data
        )
        
        logger.info("=" * 50)
        logger.info("Utility Evaluation Complete")
        logger.info("=" * 50)
        
        return results