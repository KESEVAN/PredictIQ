import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import numpy as np
from typing import Dict, List, Tuple

class DataPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def generate_sample_data(self) -> pd.DataFrame:
        """Generate synthetic enterprise metrics data"""
        dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='H')
        data = {
            'timestamp': dates,
            'cpu_usage': np.random.normal(60, 15, len(dates)),
            'memory_usage': np.random.normal(70, 10, len(dates)),
            'response_time': np.random.exponential(2, len(dates)),
            'error_rate': np.random.poisson(0.5, len(dates)),
            'throughput': np.random.normal(1000, 200, len(dates))
        }
        return pd.DataFrame(data)
    
    def preprocess_data(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Preprocess data for model training"""
        # Create time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Scale numerical features
        numerical_cols = ['cpu_usage', 'memory_usage', 'response_time', 
                         'error_rate', 'throughput']
        scaled_data = self.scaler.fit_transform(df[numerical_cols])
        
        return {
            'features': scaled_data,
            'timestamps': df['timestamp'].values,
            'raw_data': df[numerical_cols].values
        }
