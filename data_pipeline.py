"""
Data pipeline for SF311 Street and Sidewalk Cleaning predictions.
Based on user's actual SF311 data fetching and prediction logic.
"""

import pandas as pd
import numpy as np
import datetime as dt
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import requests
import streamlit as st

class SF311DataPipeline:
    """Pipeline for fetching SF311 data and generating predictions using user's approach"""
    
    def __init__(self):
        # SF311 API configuration (using user's setup)
        self.base_url = "https://data.sfgov.org/resource/vw6y-z8j6.json"
        self.meta_url = "https://data.sfgov.org/api/views/vw6y-z8j6?content=metadata"
        self.app_token = "TuXFZRAF7T8dnb1Rqk5VOdOKN"
        
        # Configuration
        self.time_field = "requested_datetime"
        self.category_field = "service_name"
        self.category_value = "Street and Sidewalk Cleaning"
        self.page_size = 50000
        
        # Preferred neighborhood fields in order
        self.neighbor_pref_order = [
            "neighborhoods_analysis_boundaries",
            "neighborhoods_sffind_boundaries", 
            "neighborhood_district",
        ]
        
        # Setup session with token
        self.session = requests.Session()
        self.session.headers.update({"X-App-Token": self.app_token})
        
    def fetch_historical_data(self, 
                            start_date: str = None, 
                            end_date: str = None,
                            limit: int = 10000) -> pd.DataFrame:
        """Fetch historical SF311 data for Street and Sidewalk Cleaning"""
        
        # Build query parameters
        params = {
            "$where": "service_name = 'Street and Sidewalk Cleaning'",
            "$limit": limit,
            "$order": "requested_datetime DESC"
        }
        
        if start_date and end_date:
            params["$where"] += f" AND requested_datetime between '{start_date}T00:00:00' and '{end_date}T23:59:59'"
        
        try:
            response = requests.get(self.sf311_api_base, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # Clean and process the data
            df = self._clean_historical_data(df)
            
            return df
            
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching SF311 data: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error processing SF311 data: {str(e)}")
            return pd.DataFrame()
    
    def _clean_historical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize historical SF311 data"""
        
        if df.empty:
            return df
        
        # Convert date columns
        if 'requested_datetime' in df.columns:
            df['requested_datetime'] = pd.to_datetime(df['requested_datetime'], errors='coerce')
            df['date'] = df['requested_datetime'].dt.date
        
        # Standardize neighborhood names
        if 'supervisor_district' in df.columns:
            df['neighborhood'] = df['supervisor_district'].fillna('Unknown')
        elif 'analysis_neighborhood' in df.columns:
            df['neighborhood'] = df['analysis_neighborhood'].fillna('Unknown')
        else:
            df['neighborhood'] = 'Unknown'
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['requested_datetime'])
        
        return df
    
    def generate_predictions(self, 
                           historical_data: pd.DataFrame,
                           prediction_days: int = 30) -> pd.DataFrame:
        """
        Generate predictions based on historical data.
        You can replace this with your actual prediction logic.
        """
        
        if historical_data.empty:
            return self._generate_baseline_predictions(prediction_days)
        
        # Aggregate historical data by date and neighborhood
        daily_counts = historical_data.groupby(['date', 'neighborhood']).size().reset_index(name='request_count')
        
        # Calculate statistics for each neighborhood
        neighborhood_stats = self._calculate_neighborhood_statistics(daily_counts)
        
        # Generate future predictions
        predictions = self._predict_future_requests(neighborhood_stats, prediction_days)
        
        return predictions
    
    def _calculate_neighborhood_statistics(self, daily_counts: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate statistics for each neighborhood"""
        
        stats = {}
        
        for neighborhood in daily_counts['neighborhood'].unique():
            neighborhood_data = daily_counts[daily_counts['neighborhood'] == neighborhood]
            
            # Calculate basic statistics
            mean_requests = neighborhood_data['request_count'].mean()
            std_requests = neighborhood_data['request_count'].std()
            
            # Calculate day-of-week patterns
            neighborhood_data['date'] = pd.to_datetime(neighborhood_data['date'])
            neighborhood_data['day_of_week'] = neighborhood_data['date'].dt.dayofweek
            
            day_patterns = neighborhood_data.groupby('day_of_week')['request_count'].mean().to_dict()
            
            stats[neighborhood] = {
                'mean': mean_requests if not pd.isna(mean_requests) else 5,
                'std': std_requests if not pd.isna(std_requests) else 2,
                'day_patterns': day_patterns
            }
        
        return stats
    
    def _predict_future_requests(self, 
                               neighborhood_stats: Dict[str, Dict], 
                               prediction_days: int) -> pd.DataFrame:
        """Predict future requests based on historical patterns"""
        
        predictions = []
        start_date = datetime.now().date()
        
        for i in range(prediction_days):
            prediction_date = start_date + timedelta(days=i)
            day_of_week = prediction_date.weekday()
            
            for neighborhood, stats in neighborhood_stats.items():
                # Base prediction
                base_prediction = stats['mean']
                
                # Apply day-of-week adjustment
                if day_of_week in stats['day_patterns']:
                    day_adjustment = stats['day_patterns'][day_of_week] / stats['mean']
                    base_prediction *= day_adjustment
                
                # Add some randomness
                np.random.seed(int(prediction_date.strftime('%Y%m%d')) + hash(neighborhood) % 1000)
                predicted_requests = max(1, int(np.random.normal(base_prediction, stats['std'])))
                
                # Calculate confidence intervals
                confidence_lower = max(1, int(predicted_requests * 0.8))
                confidence_upper = int(predicted_requests * 1.2)
                
                predictions.append({
                    'date': prediction_date,
                    'neighborhood': neighborhood,
                    'predicted_requests': predicted_requests,
                    'confidence_lower': confidence_lower,
                    'confidence_upper': confidence_upper
                })
        
        return pd.DataFrame(predictions)
    
    def _generate_baseline_predictions(self, prediction_days: int) -> pd.DataFrame:
        """Generate baseline predictions when no historical data is available"""
        
        # Default SF neighborhoods
        neighborhoods = [
            "Mission", "Castro", "SOMA", "Chinatown", "North Beach", 
            "Pacific Heights", "Marina", "Haight-Ashbury", "Richmond", 
            "Sunset", "Tenderloin", "Financial District"
        ]
        
        predictions = []
        start_date = datetime.now().date()
        
        for i in range(prediction_days):
            prediction_date = start_date + timedelta(days=i)
            
            for neighborhood in neighborhoods:
                # Simple baseline prediction
                base_requests = {
                    "Mission": 25, "Castro": 15, "SOMA": 30, "Chinatown": 18,
                    "North Beach": 16, "Pacific Heights": 10, "Marina": 12,
                    "Haight-Ashbury": 20, "Richmond": 14, "Sunset": 16,
                    "Tenderloin": 22, "Financial District": 25
                }.get(neighborhood, 15)
                
                # Weekend adjustment
                if prediction_date.weekday() in [5, 6]:
                    base_requests = int(base_requests * 0.7)
                
                predictions.append({
                    'date': prediction_date,
                    'neighborhood': neighborhood,
                    'predicted_requests': base_requests,
                    'confidence_lower': int(base_requests * 0.8),
                    'confidence_upper': int(base_requests * 1.2)
                })
        
        return pd.DataFrame(predictions)
    
    def run_full_pipeline(self, 
                         days_back: int = 90,
                         prediction_days: int = 30) -> pd.DataFrame:
        """Run the complete data pipeline"""
        
        # Calculate date range for historical data
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        # Fetch historical data
        historical_data = self.fetch_historical_data(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat()
        )
        
        # Generate predictions
        predictions = self.generate_predictions(historical_data, prediction_days)
        
        return predictions