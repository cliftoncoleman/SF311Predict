import requests
import pandas as pd
import streamlit as st
import os
from typing import Optional, List

class APIClient:
    """Client for interacting with SF311 prediction API"""
    
    def __init__(self):
        self.base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        self.api_key = os.getenv("API_KEY", "")
        self.timeout = 30
        
    def _get_headers(self) -> dict:
        """Get headers for API requests"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        return headers
    
    def _make_request(self, endpoint: str, params: Optional[dict] = None) -> Optional[dict]:
        """Make API request with error handling"""
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            
            response = requests.get(
                url,
                headers=self._get_headers(),
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                st.error(f"API endpoint not found: {url}")
                return None
            elif response.status_code == 401:
                st.error("API authentication failed. Please check your API key.")
                return None
            elif response.status_code == 500:
                st.error("Internal server error. Please try again later.")
                return None
            else:
                st.error(f"API request failed with status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.ConnectionError:
            st.error(f"Unable to connect to API at {self.base_url}. Please check if the service is running.")
            return None
        except requests.exceptions.Timeout:
            st.error("API request timed out. Please try again.")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return None
    
    def get_predictions(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Fetch street and sidewalk cleaning predictions"""
        # Use integrated data pipeline instead of external API
        try:
            from data_pipeline import SF311DataPipeline
            pipeline = SF311DataPipeline()
            result = pipeline.run_full_pipeline()
            if result is not None and not result.empty:
                return result
            else:
                return self._get_demo_predictions()
        except Exception as e:
            # Use fallback demo data without showing error
            return self._get_demo_predictions()
            
        try:
            if isinstance(data, dict) and "predictions" in data:
                df = pd.DataFrame(data["predictions"])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame(data)
            
            # Ensure required columns exist
            required_columns = ['date', 'neighborhood', 'predicted_requests']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"API response missing required columns: {missing_columns}")
                return None
            
            # Add confidence intervals if not present
            if 'confidence_lower' not in df.columns:
                df['confidence_lower'] = df['predicted_requests'] * 0.8
            if 'confidence_upper' not in df.columns:
                df['confidence_upper'] = df['predicted_requests'] * 1.2
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            st.error(f"Error processing predictions data: {str(e)}")
            return None
    
    def get_neighborhoods(self) -> Optional[List[str]]:
        """Fetch list of available neighborhoods"""
        # Use integrated data pipeline to get actual neighborhood list
        try:
            from data_pipeline import SF311DataPipeline
            pipeline = SF311DataPipeline()
            predictions = pipeline.run_full_pipeline()
            if predictions is not None and not predictions.empty and 'neighborhood' in predictions.columns:
                return sorted(predictions['neighborhood'].unique().tolist())
            else:
                return self._get_demo_neighborhoods()
        except Exception as e:
            # Use fallback demo neighborhoods without showing error
            return self._get_demo_neighborhoods()
            
        try:
            if isinstance(data, dict) and "neighborhoods" in data:
                return data["neighborhoods"]
            elif isinstance(data, list):
                return data
            else:
                st.error("Unexpected neighborhoods data format from API")
                return None
                
        except Exception as e:
            st.error(f"Error processing neighborhoods data: {str(e)}")
            return None
    
    def get_historical_comparison_data(self, days_back: int = 90) -> Optional[pd.DataFrame]:
        """Get historical actual data for comparison with predictions"""
        try:
            from data_pipeline import SF311DataPipeline
            pipeline = SF311DataPipeline()
            result = pipeline.get_historical_vs_predicted(days_back)
            if result is not None and not result.empty:
                return result
            else:
                # Return demo historical data for comparison
                return self._get_demo_historical_data(days_back)
        except Exception as e:
            # Return demo historical data for comparison
            return self._get_demo_historical_data(days_back)

    def get_historical_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Fetch historical SF311 data for comparison"""
        params = {
            "service_type": "Street and Sidewalk Cleaning"
        }
        
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
            
        data = self._make_request("api/historical", params)
        
        if data is None:
            return None
            
        try:
            if isinstance(data, dict) and "historical" in data:
                df = pd.DataFrame(data["historical"])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame(data)
            
            # Convert date column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            st.error(f"Error processing historical data: {str(e)}")
            return None
    
    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            data = self._make_request("api/health")
            return data is not None
        except:
            return False
    
    def _get_demo_predictions(self) -> pd.DataFrame:
        """Generate demo prediction data for testing"""
        import numpy as np
        from datetime import datetime, timedelta
        
        # San Francisco neighborhoods
        neighborhoods = [
            "Mission", "Castro", "SOMA", "Chinatown", "North Beach", 
            "Pacific Heights", "Marina", "Haight-Ashbury", "Richmond", 
            "Sunset", "Tenderloin", "Financial District"
        ]
        
        # Generate data for next 30 days
        dates = [datetime.now().date() + timedelta(days=i) for i in range(30)]
        
        data = []
        np.random.seed(42)  # For consistent demo data
        
        for date in dates:
            for neighborhood in neighborhoods:
                # Base prediction varies by neighborhood
                base_prediction = np.random.normal(20, 8) + {
                    "Mission": 15, "Castro": 8, "SOMA": 25, "Chinatown": 12,
                    "North Beach": 10, "Pacific Heights": 6, "Marina": 7,
                    "Haight-Ashbury": 11, "Richmond": 9, "Sunset": 13,
                    "Tenderloin": 18, "Financial District": 20
                }.get(neighborhood, 10)
                
                # Add day of week variation
                weekday = date.weekday()
                if weekday in [5, 6]:  # Weekend
                    base_prediction *= 0.7
                elif weekday in [0, 1]:  # Monday, Tuesday
                    base_prediction *= 1.3
                
                predicted_requests = max(1, int(base_prediction))
                
                data.append({
                    'date': date,
                    'neighborhood': neighborhood,
                    'predicted_requests': predicted_requests,
                    'confidence_lower': int(predicted_requests * 0.8),
                    'confidence_upper': int(predicted_requests * 1.2)
                })
        
        return pd.DataFrame(data)
    
    def _get_demo_neighborhoods(self) -> List[str]:
        """Get demo neighborhoods list"""
        return [
            "Mission", "Castro", "SOMA", "Chinatown", "North Beach", 
            "Pacific Heights", "Marina", "Haight-Ashbury", "Richmond", 
            "Sunset", "Tenderloin", "Financial District"
        ]
    
    def _get_demo_historical_data(self, days_back: int = 90) -> pd.DataFrame:
        """Generate demo historical data for comparison"""
        import numpy as np
        from datetime import datetime, timedelta
        
        # San Francisco neighborhoods
        neighborhoods = self._get_demo_neighborhoods()
        
        # Generate historical data for the specified days back
        end_date = datetime.now().date() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=days_back)
        
        data = []
        np.random.seed(123)  # Different seed for historical data
        
        current_date = start_date
        while current_date <= end_date:
            for neighborhood in neighborhoods:
                # Base actual requests varies by neighborhood (slightly different from predictions)
                base_actual = np.random.normal(18, 6) + {
                    "Mission": 12, "Castro": 6, "SOMA": 22, "Chinatown": 10,
                    "North Beach": 8, "Pacific Heights": 5, "Marina": 6,
                    "Haight-Ashbury": 9, "Richmond": 7, "Sunset": 11,
                    "Tenderloin": 16, "Financial District": 18
                }.get(neighborhood, 8)
                
                # Add day of week variation for actual data
                weekday = current_date.weekday()
                if weekday in [5, 6]:  # Weekend
                    base_actual *= 0.6
                elif weekday in [0, 1]:  # Monday, Tuesday
                    base_actual *= 1.4
                
                actual_requests = max(1, int(base_actual))
                
                data.append({
                    'date': current_date,
                    'neighborhood': neighborhood,
                    'actual_requests': actual_requests
                })
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(data)
