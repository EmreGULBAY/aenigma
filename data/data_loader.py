"""
Data loading and preprocessing for hotel consumption data.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class HotelDataLoader:
    """Load and preprocess hotel consumption data from Excel."""
    
    def __init__(self, file_path: str, sheet_names: List[str]):
        """
        Initialize data loader.
        
        Args:
            file_path: Path to Excel file
            sheet_names: List of sheet names to load
        """
        self.file_path = Path(file_path)
        self.sheet_names = sheet_names
        self.hotel_metadata = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from all hotel sheets and combine.
        
        Returns:
            Combined DataFrame with hotel_id column
        """
        logger.info(f"Loading data from {self.file_path}")
        
        all_data = []
        
        for idx, sheet_name in enumerate(self.sheet_names):
            logger.info(f"Loading {sheet_name}...")
            
            df = pd.read_excel(self.file_path, sheet_name=sheet_name)
            
            hotel_id = idx + 1
            df['hotel_id'] = hotel_id
            
            if 'occupied_rooms' in df.columns:
                capacity = df['occupied_rooms'].max()
                self.hotel_metadata[hotel_id] = {
                    'name': sheet_name,
                    'capacity': capacity,
                    'num_records': len(df)
                }
                logger.info(f"  {sheet_name}: {len(df)} records, capacity: {capacity} rooms")
            
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        if 'date' in combined_df.columns:
            combined_df['date'] = pd.to_datetime(combined_df['date'])
            combined_df = combined_df.sort_values(['hotel_id', 'date']).reset_index(drop=True)
        
        logger.info(f"Total records loaded: {len(combined_df)}")
        logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        
        return combined_df
    
    def get_hotel_metadata(self) -> Dict:
        """Return hotel metadata dictionary."""
        return self.hotel_metadata
    
    def validate_data(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate that DataFrame contains required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        missing_cols = set(required_columns) - set(df.columns)
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            logger.warning(f"Found missing values:\n{null_counts[null_counts > 0]}")
        
        return True