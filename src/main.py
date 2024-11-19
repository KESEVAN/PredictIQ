import pandas as pd
import numpy as np
import tensorflow as tf # type: ignore
from performance_analyser import PerformanceAnalyzer
from models.model_trainer import ModelTrainer
from pipeline.new_data_pipeline import DataPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Initialize pipeline and generate data
    logger.info("Initializing data pipeline...")
    pipeline = DataPipeline()
    raw_df = pipeline.generate_sample_data()
    processed_data = pipeline.preprocess_data(raw_df)
    scaler = processed_data['scaler']  # Retrieve the scaler
    # Train models
    logger.info("Initializing and training models...")
    trainer = ModelTrainer()
    history = trainer.train_models(processed_data)
    
    # Analyze performance
    logger.info("Analyzing performance...")
    analyzer = PerformanceAnalyzer(trainer)
    # processed_data['features'] = scaler.transform(processed_data['features'])  # Scale features
    predictions = analyzer.make_predictions(processed_data['features'])
    predictions = scaler.inverse_transform(predictions)
    print(predictions)
    anomalies = analyzer.detect_anomalies(processed_data['features'])
    
    # Visualize results
    logger.info("Generating visualizations...")
    analyzer.visualize_results(raw_df, predictions, anomalies)