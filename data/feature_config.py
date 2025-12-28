"""
Feature configuration and metadata for hotel consumption data.
"""
from dataclasses import dataclass
from typing import List, Dict
import yaml


@dataclass
class FeatureConfig:
    """Configuration for feature groups and metadata."""
    
    temporal: List[str]
    consumption: List[str]
    occupancy: List[str]
    weather: List[str]
    categorical: List[str]
    exclude: List[str]
    
    @property
    def all_features(self) -> List[str]:
        """Get all feature names (excluding date and metadata columns)."""
        return (
            self.temporal + 
            self.consumption + 
            self.occupancy + 
            self.weather + 
            self.categorical
        )
    
    @property
    def num_features(self) -> int:
        """Total number of features."""
        return len(self.all_features)
    
    @property
    def continuous_features(self) -> List[str]:
        """Features that should be normalized."""
        return self.consumption + self.occupancy + self.weather + [
            "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos"
        ]
    
    @property
    def discrete_features(self) -> List[str]:
        """Binary/categorical features."""
        return self.categorical + ["day_of_week", "month"]
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Return feature groups as dictionary."""
        return {
            "temporal": self.temporal,
            "consumption": self.consumption,
            "occupancy": self.occupancy,
            "weather": self.weather,
            "categorical": self.categorical
        }
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'FeatureConfig':
        """Load feature configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config['features'])