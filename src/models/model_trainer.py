from venv import logger
import numpy as np
import tensorflow as tf # type: ignore
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple


class ModelTrainer:
    def __init__(self, sequence_length: int = 24):
        self.sequence_length = sequence_length
        self.lstm_model = self._build_lstm()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
    def _build_lstm(self) -> tf.keras.Model:
        """Build LSTM model for time series prediction"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(self.sequence_length, 5), return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(5)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def train_models(self, data: Dict[str, np.ndarray], validation_split: float = 0.2):
        """Train both LSTM and anomaly detection models"""
        # Prepare sequences for LSTM
        X, y = self.prepare_sequences(data['features'])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        
        # Train LSTM
        logger.info("Training LSTM model...")
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            verbose=1
        )
        
        # Train anomaly detector
        logger.info("Training anomaly detector...")
        self.anomaly_detector.fit(data['features'])
        
        return history
