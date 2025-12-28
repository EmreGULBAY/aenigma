"""
Data preprocessing: normalization, sequence creation, train/test split.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process hotel data for TimeGAN training."""
    
    def __init__(self, sequence_length: int, stride: int = 1):
        """
        Initialize data processor.
        
        Args:
            sequence_length: Length of sequences to create
            stride: Stride for sliding window
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.scalers = {}
        self.feature_names = None
        
    def normalize_data(
        self, 
        df: pd.DataFrame, 
        feature_columns: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize features using MinMaxScaler.
        
        Args:
            df: Input DataFrame
            feature_columns: Columns to normalize
            fit: Whether to fit scaler (True for train, False for test)
            
        Returns:
            DataFrame with normalized features
        """
        df_normalized = df.copy()
        
        for col in feature_columns:
            if col not in df.columns:
                continue
                
            if fit:
                scaler = MinMaxScaler(feature_range=(0, 1))
                df_normalized[col] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
            else:
                if col in self.scalers:
                    df_normalized[col] = self.scalers[col].transform(df[[col]])
        
        return df_normalized
    
    def denormalize_data(
        self, 
        data: np.ndarray, 
        feature_columns: List[str]
    ) -> np.ndarray:
        """
        Denormalize data back to original scale.
        
        Args:
            data: Normalized data array (samples, seq_len, features)
            feature_columns: List of feature names
            
        Returns:
            Denormalized data array
        """
        denormalized = data.copy()
        
        for idx, col in enumerate(feature_columns):
            if col in self.scalers:
                original_shape = denormalized[:, :, idx].shape
                flat_data = denormalized[:, :, idx].reshape(-1, 1)
                denormalized[:, :, idx] = self.scalers[col].inverse_transform(
                    flat_data
                ).reshape(original_shape)
        
        return denormalized
    
    def create_sequences(
        self, 
        df: pd.DataFrame, 
        feature_columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences using sliding window per hotel.
        
        Args:
            df: Input DataFrame with hotel_id column
            feature_columns: Features to include in sequences
            
        Returns:
            Tuple of (sequences, hotel_ids)
            - sequences: (num_sequences, seq_length, num_features)
            - hotel_ids: (num_sequences,)
        """
        self.feature_names = feature_columns
        sequences = []
        hotel_ids = []
        
        for hotel_id in df['hotel_id'].unique():
            hotel_df = df[df['hotel_id'] == hotel_id].reset_index(drop=True)
            hotel_data = hotel_df[feature_columns].values
            
            num_sequences = (len(hotel_data) - self.sequence_length) // self.stride + 1
            
            for i in range(num_sequences):
                start_idx = i * self.stride
                end_idx = start_idx + self.sequence_length
                
                if end_idx <= len(hotel_data):
                    sequences.append(hotel_data[start_idx:end_idx])
                    hotel_ids.append(hotel_id)
            
            logger.info(
                f"Hotel {hotel_id}: Created {num_sequences} sequences "
                f"from {len(hotel_data)} records"
            )
        
        sequences = np.array(sequences, dtype=np.float32)
        hotel_ids = np.array(hotel_ids, dtype=np.int64)
        
        logger.info(f"Total sequences created: {len(sequences)}")
        logger.info(f"Sequence shape: {sequences.shape}")
        
        return sequences, hotel_ids
    
    def train_test_split(
        self,
        sequences: np.ndarray,
        hotel_ids: np.ndarray,
        train_ratio: float = 0.8,
        random_seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets.
        
        Args:
            sequences: Sequence data
            hotel_ids: Hotel identifiers
            train_ratio: Ratio of training data
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_sequences, train_hotel_ids, test_sequences, test_hotel_ids)
        """
        np.random.seed(random_seed)
        
        indices = np.arange(len(sequences))
        np.random.shuffle(indices)
        
        split_idx = int(len(sequences) * train_ratio)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        train_sequences = sequences[train_indices]
        train_hotel_ids = hotel_ids[train_indices]
        test_sequences = sequences[test_indices]
        test_hotel_ids = hotel_ids[test_indices]
        
        logger.info(f"Train sequences: {len(train_sequences)}")
        logger.info(f"Test sequences: {len(test_sequences)}")
        
        return train_sequences, train_hotel_ids, test_sequences, test_hotel_ids
    
    def get_scaler_params(self) -> Dict:
        """Return scaler parameters for saving/loading."""
        return {
            col: {
                'min': scaler.data_min_[0],
                'max': scaler.data_max_[0],
                'scale': scaler.scale_[0]
            }
            for col, scaler in self.scalers.items()
        }