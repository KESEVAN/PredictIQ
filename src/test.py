import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt # For plotting\
from sklearn.model_selection import train_test_split


class DataPipeline:  # Simplified for sine wave
    def __init__(self):
        self.scaler = MinMaxScaler()

    def generate_sine_wave(self, num_points=1000):
        time = np.arange(0, num_points, 0.1)
        data = np.sin(time)
        df = pd.DataFrame({'timestamp': time, 'value': data})
        return df
    
    def preprocess_data(self, df, fit_scaler=True):
         if fit_scaler:
            self.scaler.fit(df[['value']])  # Fit only on training data

         scaled_data = self.scaler.transform(df[['value']].copy())

         return {
            'features': scaled_data,
            'timestamps': df['timestamp'].values,
            'scaler': self.scaler
        }

class ModelTrainer:
    def __init__(self, sequence_length=24, scaler=None):
        self.sequence_length = sequence_length
        self.scaler = scaler # Pass scaler from DataPipeline

        self.lstm_model = self._build_lstm()

    def _build_lstm(self): # Simplified LSTM for testing
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(self.sequence_length, 1), return_sequences=True),  # 1 Feature
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1, activation='linear') # Output is 1 value
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    def prepare_sequences(self, data):
        sequences = []
        targets = []
        
        # Convert data to correct shape if needed
        if isinstance(data, np.ndarray) and len(data.shape) == 2:
            data = data.flatten()
            
        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:(i + self.sequence_length)])
            targets.append(data[i + self.sequence_length])
            
        return np.array(sequences).reshape(-1, self.sequence_length, 1), np.array(targets)

    def train_models(self, data, epochs=100, batch_size=32, validation_split=0.2): # Increased epochs
        X, y = self.prepare_sequences(data['features']) # Use the scaled data from pipeline
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)

        # No scaling here, as it's done in the pipeline

        history = self.lstm_model.fit(X_train, y_train, 
                                      validation_data=(X_val, y_val),
                                      epochs=epochs, batch_size=batch_size, verbose=1)
        return history
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        # Ensure data is in correct shape
        if len(data.shape) == 2:
            data = data.flatten()
            
        sequences = []
        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:(i + self.sequence_length)])
        
        sequences = np.array(sequences).reshape(-1, self.sequence_length, 1)
        predictions = self.lstm_model.predict(sequences)
        
        return predictions



# Simplified Test Script
if __name__ == "__main__":
    pipeline = DataPipeline()
    sine_df = pipeline.generate_sine_wave()
    processed_sine_data = pipeline.preprocess_data(sine_df)
    scaler = processed_sine_data['scaler']  # Store scaler

    trainer = ModelTrainer(sequence_length=20, scaler=scaler)  # Pass scaler, adjusted sequence length
    history = trainer.train_models(processed_sine_data, epochs=20, batch_size=32)

    X_test, y_test = trainer.prepare_sequences(processed_sine_data['features'])
    predictions = trainer.predict(X_test)  # No scaling needed here

    # Plotting
    plt.plot(sine_df['timestamp'][trainer.sequence_length:], y_test.flatten(), label='Actual')
    plt.plot(sine_df['timestamp'][trainer.sequence_length:], predictions.flatten(), label='Predicted') # Correct plotting slice
    plt.legend()
    plt.show()