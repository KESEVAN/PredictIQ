from venv import logger
import numpy as np
import tensorflow as tf # type: ignore
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler 
from tensorflow.keras.callbacks import EarlyStopping

class ModelTrainer:
    def __init__(self, sequence_length: int = 24):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()  # Initialize scaler
        self.lstm_model = self._build_lstm()
        self.anomaly_detector = IsolationForest(contamination='auto', random_state=42) 
    def _build_lstms(self) -> tf.keras.Model:
        """Build LSTM model for time series prediction"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(self.sequence_length, 5), return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation='linear'),
            tf.keras.layers.Dense(5)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def _build_lstm(self) -> tf.keras.Model:
        """Build a more complex LSTM model with bidirectional layers."""
        model = tf.keras.Sequential([
            # Bidirectional LSTM to capture patterns in both directions
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(128, return_sequences=True),
                input_shape=(self.sequence_length, 5)
            ),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=True)
            ),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32)
            ),
            
            # Wider dense layers with different activations
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(5, activation='linear')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust to outliers than MSE
            metrics=['mae', 'mse']
        )
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
         # ***Scaling inside ModelTrainer - CRUCIAL***
        self.scaler.fit(y_train)  # Fit on the training output
        y_train_scaled = self.scaler.transform(y_train.copy())  # Transform training output
        y_val_scaled = self.scaler.transform(y_val.copy()) # Transform validation output
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train LSTM
        logger.info("Training LSTM model...")
        history = self.lstm_model.fit(
            X_train, y_train_scaled,
            validation_data=(X_val, y_val_scaled),
            epochs=100,
            batch_size=32,
            verbose=1,
            callbacks=[early_stopping]
        )
        
        # Train anomaly detector
        logger.info("Training anomaly detector...")
        self.anomaly_detector.fit(data['features'])
        
        return history
