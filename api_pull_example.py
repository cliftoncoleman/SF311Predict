#!/usr/bin/env python3
"""
SF311 API Data Pull Example
Direct API access for fetching SF311 Street & Sidewalk Cleaning data
"""

import pandas as pd
import requests
import datetime as dt
from fixed_pipeline import FixedSF311Pipeline

def main():
    """Example of direct API data pull"""
    
    print("SF311 API Data Pull Example")
    print("=" * 40)
    
    # Initialize the pipeline
    pipeline = FixedSF311Pipeline()
    
    print(f"API Base URL: {pipeline.base_url}")
    print(f"App Token: {pipeline.app_token}")
    print(f"Category: {pipeline.category_value}")
    print()
    
    # Example 1: Get field names
    print("1. Getting available field names...")
    try:
        fields = pipeline.get_field_names()
        print(f"Available fields: {len(fields)}")
        neighborhood_field = pipeline.pick_neighborhood_field(fields)
        print(f"Selected neighborhood field: {neighborhood_field}")
    except Exception as e:
        print(f"Error getting fields: {e}")
    
    print()
    
    # Example 2: Fetch recent data (last 30 days)
    print("2. Fetching recent SF311 data...")
    try:
        recent_data = pipeline.fetch_historical_data(start_days=30)
        
        if not recent_data.empty:
            print(f"Records fetched: {len(recent_data)}")
            print(f"Date range: {recent_data['date'].min()} to {recent_data['date'].max()}")
            print(f"Neighborhoods: {recent_data['neighborhood'].nunique()}")
            
            # Show sample data
            print("\nSample data:")
            print(recent_data.head())
            
            # Show top neighborhoods by case count
            print("\nTop neighborhoods by case count:")
            top_neighborhoods = recent_data.groupby('neighborhood')['cases'].sum().sort_values(ascending=False).head(10)
            for neighborhood, cases in top_neighborhoods.items():
                print(f"  {neighborhood}: {cases} cases")
                
        else:
            print("No data retrieved")
            
    except Exception as e:
        print(f"Error fetching data: {e}")
    
    print()
    
    # Example 3: Direct API call
    print("3. Making direct API call...")
    try:
        # Get data for the last 7 days
        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=7)
        
        params = {
            "$select": f"{pipeline.time_field}, neighborhoods_analysis_boundaries",
            "$where": (
                f"{pipeline.category_field} = '{pipeline.category_value}' AND "
                f"{pipeline.time_field} >= '{start_date.isoformat()}T00:00:00.000'"
            ),
            "$order": f"{pipeline.time_field} DESC",
            "$limit": 100,
        }
        
        response = pipeline.session.get(pipeline.base_url, params=params, timeout=30)
        
        if response.status_code == 200:
            raw_data = response.json()
            print(f"Direct API call successful: {len(raw_data)} records")
            
            if raw_data:
                print("Sample raw record:")
                print(raw_data[0])
        else:
            print(f"API call failed: {response.status_code}")
            
    except Exception as e:
        print(f"Error with direct API call: {e}")

if __name__ == "__main__":
    main()