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
        
    def get_field_names(self) -> set:
        """Get available field names from SF311 API metadata"""
        try:
            r = self.session.get(self.meta_url, timeout=60)
            r.raise_for_status()
            meta = r.json()
            return {c["fieldName"] for c in meta.get("columns", [])}
        except Exception as e:
            st.warning(f"Could not fetch field metadata: {e}")
            return set()

    def pick_neighborhood_field(self, field_names: set) -> str:
        """Pick the best available neighborhood field"""
        for f in self.neighbor_pref_order:
            if f in field_names:
                return f
        return self.neighbor_pref_order[0]

    def month_windows(self, start_date: dt.date, end_date: dt.date) -> List[tuple]:
        """Generate month windows for data fetching"""
        cur = dt.date(start_date.year, start_date.month, 1)
        end_month = dt.date(end_date.year, end_date.month, 1)
        while cur <= end_month:
            next_month = (cur.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
            win_start = dt.datetime.combine(cur, dt.time.min)
            hard_end_date = end_date + dt.timedelta(days=1)
            win_end_date = min(next_month, hard_end_date)
            win_end = dt.datetime.combine(win_end_date, dt.time.min)
            yield win_start.isoformat(), win_end.isoformat()
            cur = next_month

    def fetch_month(self, neighborhood_field: str, win_start_iso: str, win_end_iso: str) -> pd.DataFrame:
        """Fetch one month of data with paging"""
        frames = []
        offset = 0
        retries = 0
        
        while True:
            params = {
                "$select": f"{self.time_field}, {neighborhood_field}",
                "$where": (
                    f"{self.category_field} = '{self.category_value}' AND "
                    f"{self.time_field} >= '{win_start_iso}' AND {self.time_field} < '{win_end_iso}'"
                ),
                "$order": f"{self.time_field} ASC",
                "$limit": self.page_size,
                "$offset": offset,
            }
            
            try:
                r = self.session.get(self.base_url, params=params, timeout=120)
            except requests.exceptions.RequestException as e:
                if retries < 5:
                    time.sleep(2 ** retries)
                    retries += 1
                    continue
                raise RuntimeError(f"Network error after retries: {e}") from e

            if r.status_code == 429:  # rate limited
                time.sleep(2 + retries)
                retries = min(retries + 1, 5)
                continue

            if r.status_code != 200:
                raise RuntimeError(f"Socrata error {r.status_code}: {r.text[:1000]}")

            rows = r.json()
            if not rows:
                break
                
            frames.append(pd.DataFrame(rows))
            if len(rows) < self.page_size:
                break
            offset += self.page_size

        if not frames:
            return pd.DataFrame(columns=[self.time_field, neighborhood_field])
        return pd.concat(frames, ignore_index=True)

    def fetch_historical_data(self, start_days: int = 365) -> pd.DataFrame:
        """Fetch historical SF311 data using your exact approach"""
        today = dt.date.today()
        start_date = today - dt.timedelta(days=start_days)
        
        try:
            # Get field names and pick neighborhood field
            fields = self.get_field_names()
            nbhd_field = self.pick_neighborhood_field(fields)
            
            # Fetch data by month windows
            all_frames = []
            for win_start, win_end in self.month_windows(start_date, today):
                df_month = self.fetch_month(nbhd_field, win_start, win_end)
                if not df_month.empty:
                    all_frames.append(df_month)
            
            if not all_frames:
                return pd.DataFrame()
                
            # Combine and process
            raw = pd.concat(all_frames, ignore_index=True)
            
            # Clean and aggregate
            raw[self.time_field] = pd.to_datetime(raw[self.time_field], errors="coerce", utc=True)
            raw["date"] = raw[self.time_field].dt.tz_convert("US/Pacific").dt.date
            raw["neighborhood"] = raw[nbhd_field].fillna("Unknown").astype(str)
            
            # Aggregate to daily counts
            daily = (
                raw.groupby(["date", "neighborhood"], as_index=False)
                   .size()
                   .rename(columns={"size": "cases"})
                   .sort_values(["date", "neighborhood"])
            )
            
            return daily
            
        except Exception as e:
            st.error(f"Error fetching SF311 data: {str(e)}")
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
        Using simple time series approach - replace with your ML model.
        """
        
        if historical_data.empty:
            return self._generate_baseline_predictions(prediction_days)
        
        # Use the 'cases' column from your data structure
        daily_counts = historical_data.copy()
        daily_counts = daily_counts.rename(columns={'cases': 'request_count'})
        
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
                         days_back: int = 180,
                         prediction_days: int = 30) -> pd.DataFrame:
        """Run the complete data pipeline using your approach"""
        
        # Fetch historical data using your method
        historical_data = self.fetch_historical_data(start_days=days_back)
        
        # Generate predictions
        predictions = self.generate_predictions(historical_data, prediction_days)
        
        return predictions