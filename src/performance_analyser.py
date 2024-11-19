from models.model_trainer import ModelTrainer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class PerformanceAnalyzer:
    def __init__(self, trainer):
        self.trainer = trainer
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)

    def detect_anomalies(self, features: np.ndarray) -> pd.DataFrame:
        """
        Detect anomalies in preprocessed data.

        Parameters:
        processed_data (dict): Dictionary containing:
            - 'features': Preprocessed features (numpy array)
            - 'timestamps': Timestamps corresponding to features
            - 'raw_data': Original raw data (optional, numpy array)

        Returns:
        pd.DataFrame: DataFrame with timestamps, features, and an 'is_anomaly' column.
        """
        
        
        # Fit the anomaly detection model
        self.anomaly_detector.fit(features)

        # Predict anomalies (-1 = anomaly, 1 = normal)
        predictions = self.anomaly_detector.predict(features)

        # # Create a DataFrame to combine timestamps, features, and anomaly labels
        # df = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(features.shape[1])])
        # df['timestamp'] = pd.to_datetime(timestamps)
        # df['is_anomaly'] = np.where(predictions == -1, True, False)

        return predictions
    
    def make_predictions(self, data: np.ndarray) -> np.ndarray:
        """Make predictions using the LSTM model"""
        sequences = []
        for i in range(len(data) - self.trainer.sequence_length):
            sequences.append(data[i:(i + self.trainer.sequence_length)])
        sequences = np.array(sequences)
        return self.trainer.lstm_model.predict(sequences)
    
    def visualize_results(self, raw_data: pd.DataFrame, predictions: np.ndarray, anomalies: np.ndarray):
        """Create interactive visualizations using plotly"""
        metrics = ['cpu_usage', 'memory_usage', 'response_time', 'error_rate', 'throughput']
        
        for i, metric in enumerate(metrics):
            fig = go.Figure()
            
            # Plot actual values
            fig.add_trace(go.Scatter(
                x=raw_data['timestamp'],
                y=raw_data[metric],
                name='Actual',
                line=dict(color='blue')
            ))
            
            # Plot predictions
            fig.add_trace(go.Scatter(
                x=raw_data['timestamp'][self.trainer.sequence_length:],
                y=predictions[:, i],
                name='Predicted',
                line=dict(color='green', dash='dash')
            ))
            
            # Highlight anomalies
            anomaly_indices = np.where(anomalies == -1)[0]
            fig.add_trace(go.Scatter(
                x=raw_data['timestamp'][anomaly_indices],
                y=raw_data[metric].iloc[anomaly_indices],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10)
            ))
            
            fig.update_layout(
                title=f'{metric.replace("_", " ").title()} Over Time',
                xaxis_title='Timestamp',
                yaxis_title=metric.replace('_', ' ').title(),
                template='plotly_white'
            )
            
            fig.show()


