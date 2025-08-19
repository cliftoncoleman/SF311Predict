import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

class DataProcessor:
    """Class for processing and transforming SF311 prediction data"""
    
    def __init__(self):
        pass
    
    def process_for_visualization(self, data: pd.DataFrame, aggregation_level: str = "daily") -> pd.DataFrame:
        """Process data for visualization based on aggregation level"""
        try:
            if data.empty:
                return data
            
            # Ensure date column is datetime
            data = data.copy()
            data['date'] = pd.to_datetime(data['date'])
            
            if aggregation_level == "weekly":
                return self._aggregate_weekly(data)
            elif aggregation_level == "monthly":
                return self._aggregate_monthly(data)
            else:  # daily
                return self._aggregate_daily(data)
                
        except Exception as e:
            print(f"Error processing data for visualization: {str(e)}")
            return data
    
    def _aggregate_daily(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data by day"""
        try:
            aggregated = data.groupby(['date', 'neighborhood']).agg({
                'predicted_requests': 'sum',
                'confidence_lower': 'sum',
                'confidence_upper': 'sum'
            }).reset_index()
            
            return aggregated.sort_values(['date', 'neighborhood'])
            
        except Exception as e:
            print(f"Error in daily aggregation: {str(e)}")
            return data
    
    def _aggregate_weekly(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data by week with proper confidence interval calculation - only complete weeks"""
        try:
            data_copy = data.copy()
            
            # Create ISO week starting on Monday
            data_copy['week_start'] = data_copy['date'].dt.to_period('W-MON').dt.start_time
            
            # Count days per week per neighborhood to filter out partial weeks
            day_counts = data_copy.groupby(['week_start', 'neighborhood']).size().reset_index(name='day_count')
            
            # Only keep weeks with 7 complete days
            complete_weeks = day_counts[day_counts['day_count'] == 7][['week_start', 'neighborhood']]
            
            # Filter original data to only include complete weeks
            data_filtered = data_copy.merge(complete_weeks, on=['week_start', 'neighborhood'], how='inner')
            
            # Group and aggregate predictions for complete weeks only
            grouped = data_filtered.groupby(['week_start', 'neighborhood'])
            
            # Sum the predicted requests
            predicted_sums = grouped['predicted_requests'].sum()
            
            # For confidence intervals, use statistical combination
            # Assuming independence, variance adds when summing random variables
            # CI width ≈ (upper - lower), so we use root sum of squares for combining uncertainties
            lower_diffs = grouped.apply(lambda x: ((x['predicted_requests'] - x['confidence_lower']) ** 2).sum() ** 0.5)
            upper_diffs = grouped.apply(lambda x: ((x['confidence_upper'] - x['predicted_requests']) ** 2).sum() ** 0.5)
            
            # Create final aggregated dataframe
            aggregated = pd.DataFrame({
                'date': predicted_sums.index.get_level_values('week_start'),
                'neighborhood': predicted_sums.index.get_level_values('neighborhood'),
                'predicted_requests': predicted_sums.values,
                'confidence_lower': predicted_sums.values - lower_diffs.values,
                'confidence_upper': predicted_sums.values + upper_diffs.values
            })
            
            # Ensure confidence bounds are reasonable
            aggregated['confidence_lower'] = aggregated['confidence_lower'].clip(lower=0)
            
            return aggregated.sort_values(['date', 'neighborhood'])
            
        except Exception as e:
            print(f"Error in weekly aggregation: {str(e)}")
            return data
    
    def _aggregate_monthly(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data by month with proper confidence interval calculation"""
        try:
            data_copy = data.copy()
            data_copy['month'] = data_copy['date'].dt.to_period('M').dt.start_time
            
            # Group and aggregate predictions
            grouped = data_copy.groupby(['month', 'neighborhood'])
            
            # Sum the predicted requests
            predicted_sums = grouped['predicted_requests'].sum()
            
            # For confidence intervals, use statistical combination
            # Assuming independence, variance adds when summing random variables
            lower_diffs = grouped.apply(lambda x: ((x['predicted_requests'] - x['confidence_lower']) ** 2).sum() ** 0.5)
            upper_diffs = grouped.apply(lambda x: ((x['confidence_upper'] - x['predicted_requests']) ** 2).sum() ** 0.5)
            
            # Create final aggregated dataframe
            aggregated = pd.DataFrame({
                'date': predicted_sums.index.get_level_values('month'),
                'neighborhood': predicted_sums.index.get_level_values('neighborhood'),
                'predicted_requests': predicted_sums.values,
                'confidence_lower': predicted_sums.values - lower_diffs.values,
                'confidence_upper': predicted_sums.values + upper_diffs.values
            })
            
            # Ensure confidence bounds are reasonable
            aggregated['confidence_lower'] = aggregated['confidence_lower'].clip(lower=0)
            
            return aggregated.sort_values(['date', 'neighborhood'])
            
        except Exception as e:
            print(f"Error in monthly aggregation: {str(e)}")
            return data
    
    def calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for the data"""
        try:
            if data.empty:
                return {}
            
            stats = {
                'total_requests': data['predicted_requests'].sum(),
                'average_daily': data.groupby('date')['predicted_requests'].sum().mean(),
                'peak_day': data.groupby('date')['predicted_requests'].sum().idxmax(),
                'peak_value': data.groupby('date')['predicted_requests'].sum().max(),
                'total_neighborhoods': data['neighborhood'].nunique(),
                'date_range': {
                    'start': data['date'].min(),
                    'end': data['date'].max()
                },
                'top_neighborhoods': data.groupby('neighborhood')['predicted_requests'].sum().nlargest(5).to_dict()
            }
            
            return stats
            
        except Exception as e:
            print(f"Error calculating statistics: {str(e)}")
            return {}
    
    def prepare_heatmap_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for heatmap visualization"""
        try:
            if data.empty:
                return data
            
            # Create pivot table for heatmap
            heatmap_data = data.pivot_table(
                index='neighborhood',
                columns='date',
                values='predicted_requests',
                aggfunc='sum',
                fill_value=0
            )
            
            return heatmap_data
            
        except Exception as e:
            print(f"Error preparing heatmap data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trend information for the data"""
        try:
            if data.empty or len(data) < 2:
                return {}
            
            # Calculate overall trend
            daily_totals = data.groupby('date')['predicted_requests'].sum().reset_index()
            daily_totals = daily_totals.sort_values('date')
            
            if len(daily_totals) < 2:
                return {}
            
            # Calculate percentage change from first to last period
            first_value = daily_totals['predicted_requests'].iloc[0]
            last_value = daily_totals['predicted_requests'].iloc[-1]
            
            if first_value > 0:
                overall_change = ((last_value - first_value) / first_value) * 100
            else:
                overall_change = 0
            
            # Calculate moving average
            if len(daily_totals) >= 7:
                daily_totals['moving_avg'] = daily_totals['predicted_requests'].rolling(window=7).mean()
            
            trends = {
                'overall_change_percent': overall_change,
                'trend_direction': 'increasing' if overall_change > 5 else 'decreasing' if overall_change < -5 else 'stable',
                'daily_totals': daily_totals,
                'volatility': daily_totals['predicted_requests'].std() if len(daily_totals) > 1 else 0
            }
            
            return trends
            
        except Exception as e:
            print(f"Error calculating trends: {str(e)}")
            return {}
    
    def filter_by_confidence(self, data: pd.DataFrame, min_confidence: float = 0.8) -> pd.DataFrame:
        """Filter data based on prediction confidence"""
        try:
            if data.empty:
                return data
            
            # Calculate confidence as the ratio of prediction to upper bound
            data_copy = data.copy()
            data_copy['confidence_ratio'] = (
                data_copy['predicted_requests'] / data_copy['confidence_upper']
            ).fillna(0)
            
            # Filter by minimum confidence
            filtered_data = data_copy[data_copy['confidence_ratio'] >= min_confidence]
            
            return filtered_data
            
        except Exception as e:
            print(f"Error filtering by confidence: {str(e)}")
            return data
    
    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in prediction data"""
        try:
            if data.empty:
                return data
            
            data_copy = data.copy()
            
            # Calculate z-scores for each neighborhood
            for neighborhood in data_copy['neighborhood'].unique():
                mask = data_copy['neighborhood'] == neighborhood
                neighborhood_data = data_copy.loc[mask, 'predicted_requests']
                
                if len(neighborhood_data) > 1:
                    mean_val = neighborhood_data.mean()
                    std_val = neighborhood_data.std()
                    
                    if std_val > 0:
                        z_scores = np.abs((neighborhood_data - mean_val) / std_val)
                        data_copy.loc[mask, 'is_anomaly'] = z_scores > 2.5
                    else:
                        data_copy.loc[mask, 'is_anomaly'] = False
                else:
                    data_copy.loc[mask, 'is_anomaly'] = False
            
            return data_copy
            
        except Exception as e:
            print(f"Error detecting anomalies: {str(e)}")
            return data
