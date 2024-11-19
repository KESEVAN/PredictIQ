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
import os
import json

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

    # # Analyze anomalies with LLM
    # logger.info("Performing LLM analysis of anomalies...")
    # anomaly_analysis = analyzer.analyze_anomalies(raw_df, anomalies)
    
    # # Generate performance report
    # logger.info("Generating performance report...")
    # performance_report = analyzer.generate_report(raw_df, anomalies)
    
    # # Save analysis results
    # output_dir = 'reports'
    # os.makedirs(output_dir, exist_ok=True)
    
    # # Save anomaly analysis
    # with open(os.path.join(output_dir, 'anomaly_analysis.json'), 'w') as f:
    #     json.dump(anomaly_analysis, f, indent=2, default=str)
    
    # # Save performance report
    # with open(os.path.join(output_dir, 'performance_report.md'), 'w') as f:
    #     f.write(performance_report)
    