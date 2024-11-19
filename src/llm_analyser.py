from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class LLMAnalyzer:
    def __init__(self, api_key: str = None):
        """Initialize LLM analyzer with OpenAI API key"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4-turbo-preview",
            openai_api_key=self.api_key
        )
        
        # Define output schemas for structured responses
        self.response_schemas = [
            ResponseSchema(name="root_cause", description="The likely root cause of the anomaly"),
            ResponseSchema(name="severity", description="Severity level (LOW, MEDIUM, HIGH)"),
            ResponseSchema(name="recommendations", description="List of recommended actions"),
            ResponseSchema(name="impact", description="Potential business impact of the anomaly")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)
        
    def analyze_anomaly(self, 
                       metrics_data: pd.DataFrame, 
                       anomaly_timestamp: datetime,
                       window_size: int = 24) -> Dict[str, Any]:
        """Analyze anomaly using LLM for root cause analysis"""
        # Get context window around anomaly
        start_time = anomaly_timestamp - timedelta(hours=window_size//2)
        end_time = anomaly_timestamp + timedelta(hours=window_size//2)
        context_data = metrics_data[
            (metrics_data['timestamp'] >= start_time) & 
            (metrics_data['timestamp'] <= end_time)
        ]
        
        # Prepare metrics summary
        metrics_summary = self._prepare_metrics_summary(context_data, anomaly_timestamp)
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_template("""
        Analyze the following enterprise system metrics around an detected anomaly:
        
        Timestamp of Anomaly: {timestamp}
        
        Metrics Summary:
        {metrics_summary}
        
        Previous 12-hour trends:
        {trends}
        
        Based on this information, provide:
        1. Most likely root cause
        2. Severity level (LOW, MEDIUM, HIGH)
        3. Recommended actions
        4. Potential business impact
        
        {format_instructions}
        """)
        
        # Format prompt with data
        messages = prompt.format_messages(
            timestamp=anomaly_timestamp,
            metrics_summary=metrics_summary,
            trends=self._get_trend_description(context_data),
            format_instructions=self.output_parser.get_format_instructions()
        )
        
        # Get LLM response
        response = self.llm.invoke(messages)
        
        # Parse structured output
        return self.output_parser.parse(response.content)
    
    def generate_performance_report(self, 
                                 metrics_data: pd.DataFrame,
                                 anomalies: List[datetime],
                                 period: str = "daily") -> str:
        """Generate a comprehensive performance report"""
        prompt = ChatPromptTemplate.from_template("""
        Generate a detailed {period} performance report based on the following metrics:
        
        System Metrics Summary:
        {metrics_summary}
        
        Number of Anomalies: {anomaly_count}
        Anomaly Timestamps: {anomaly_times}
        
        Please provide:
        1. Overall system health assessment
        2. Key performance trends
        3. Critical incidents summary
        4. Recommendations for improvement
        
        Format the response in markdown.
        """)
        
        messages = prompt.format_messages(
            period=period,
            metrics_summary=self._prepare_metrics_summary(metrics_data),
            anomaly_count=len(anomalies),
            anomaly_times=", ".join([str(t) for t in anomalies])
        )
        
        response = self.llm.invoke(messages)
        return response.content
    
    def _prepare_metrics_summary(self, 
                               df: pd.DataFrame, 
                               timestamp: datetime = None) -> str:
        """Prepare a summary of metrics for LLM analysis"""
        if timestamp:
            # Get metrics at anomaly time
            current = df[df['timestamp'] == timestamp].iloc[0]
            summary = f"""
            Current Metrics (at {timestamp}):
            - CPU Usage: {current['cpu_usage']:.2f}%
            - Memory Usage: {current['memory_usage']:.2f}%
            - Response Time: {current['response_time']:.2f}ms
            - Error Rate: {current['error_rate']:.2f}
            - Throughput: {current['throughput']:.2f} req/s
            """
        else:
            # Get overall statistics
            summary = f"""
            Metric Ranges:
            - CPU Usage: {df['cpu_usage'].min():.2f}% - {df['cpu_usage'].max():.2f}%
            - Memory Usage: {df['memory_usage'].min():.2f}% - {df['memory_usage'].max():.2f}%
            - Response Time: {df['response_time'].min():.2f}ms - {df['response_time'].max():.2f}ms
            - Error Rate: {df['error_rate'].min():.2f} - {df['error_rate'].max():.2f}
            - Throughput: {df['throughput'].min():.2f} - {df['throughput'].max():.2f} req/s
            """
        return summary
    
    def _get_trend_description(self, df: pd.DataFrame) -> str:
        """Generate trend descriptions for metrics"""
        trends = []
        for col in ['cpu_usage', 'memory_usage', 'response_time', 'error_rate', 'throughput']:
            # Calculate simple trend
            start_val = df[col].iloc[0]
            end_val = df[col].iloc[-1]
            pct_change = ((end_val - start_val) / start_val) * 100
            
            trend = "increasing" if pct_change > 5 else "decreasing" if pct_change < -5 else "stable"
            trends.append(f"{col.replace('_', ' ').title()}: {trend} ({pct_change:.1f}% change)")
        
        return "\n".join(trends)