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
    
    def _make_request(self, endpoint: str, params: dict = None) -> Optional[dict]:
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
    
    def get_predictions(self, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """Fetch street and sidewalk cleaning predictions"""
        params = {
            "service_type": "Street and Sidewalk Cleaning"
        }
        
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
            
        data = self._make_request("api/predictions", params)
        
        if data is None:
            return None
            
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
        data = self._make_request("api/neighborhoods")
        
        if data is None:
            return None
            
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
    
    def get_historical_data(self, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
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
