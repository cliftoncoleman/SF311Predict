#!/usr/bin/env python3
"""Debug historical data functionality"""

import pandas as pd
from datetime import datetime, timedelta
from fixed_pipeline import FixedSF311Pipeline

def debug_historical_data():
    """Debug historical data fetching and processing"""
    print("=== Debugging Historical Data Functionality ===\n")
    
    pipeline = FixedSF311Pipeline()
    
    # Test 1: Historical data fetching
    print("1. Testing historical data fetching...")
    historical = pipeline.fetch_historical_data(start_days=365)
    
    if historical.empty:
        print("❌ No historical data retrieved")
        return
    
    print(f"✓ Retrieved {len(historical)} historical records")
    print(f"  Date range: {historical['date'].min()} to {historical['date'].max()}")
    print(f"  Neighborhoods: {historical['neighborhood'].nunique()}")
    print(f"  Columns: {list(historical.columns)}")
    
    # Test 2: Data quality checks
    print("\n2. Checking data quality...")
    print(f"  Missing values: {historical.isnull().sum().sum()}")
    print(f"  Duplicate records: {historical.duplicated().sum()}")
    print(f"  Cases range: {historical['cases'].min():.1f} to {historical['cases'].max():.1f}")
    print(f"  Average cases per day: {historical['cases'].mean():.1f}")
    
    # Test 3: Neighborhood distribution
    print("\n3. Neighborhood distribution...")
    neighborhood_counts = historical['neighborhood'].value_counts().head(10)
    print("  Top neighborhoods by record count:")
    for name, count in neighborhood_counts.items():
        print(f"    {name}: {count} records")
    
    # Test 4: Time series continuity
    print("\n4. Time series continuity...")
    historical['date'] = pd.to_datetime(historical['date'])
    historical_sorted = historical.sort_values('date')
    date_gaps = historical_sorted['date'].diff().dt.days
    max_gap = date_gaps.max()
    print(f"  Maximum gap between consecutive records: {max_gap} days")
    
    # Test 5: Recent vs prediction comparison
    print("\n5. Testing historical vs predicted comparison...")
    recent_data = pipeline.get_historical_vs_predicted(days_back=30)
    
    if not recent_data.empty:
        print(f"✓ Retrieved {len(recent_data)} recent records for comparison")
        print(f"  Date range: {recent_data['date'].min()} to {recent_data['date'].max()}")
        print(f"  Average actual requests: {recent_data['actual_requests'].mean():.1f}")
    else:
        print("❌ No recent comparison data retrieved")
    
    # Test 6: Sample neighborhood forecasting
    print("\n6. Testing sample neighborhood forecasting...")
    sample_neighborhood = historical['neighborhood'].value_counts().index[0]
    nbhd_data = historical[historical['neighborhood'] == sample_neighborhood].copy()
    nbhd_data = nbhd_data.sort_values('date').reset_index(drop=True)
    
    print(f"  Sample neighborhood: {sample_neighborhood}")
    print(f"  Historical records: {len(nbhd_data)}")
    print(f"  Date range: {nbhd_data['date'].min()} to {nbhd_data['date'].max()}")
    
    if len(nbhd_data) >= 60:
        print(f"  ✓ Sufficient data for advanced modeling")
        # Show last few values
        recent_values = nbhd_data['cases'].tail(7).values
        print(f"  Last 7 days: {recent_values}")
    else:
        print(f"  ⚠ Limited data - will use simple forecasting")
    
    # Test 7: Full pipeline test
    print("\n7. Testing full pipeline...")
    try:
        predictions = pipeline.run_full_fixed_pipeline(
            days_back=365,
            prediction_days=7  # Small test
        )
        
        if not predictions.empty:
            print(f"✓ Generated {len(predictions)} predictions")
            print(f"  Neighborhoods: {predictions['neighborhood'].nunique()}")
            print(f"  Date range: {predictions['date'].min()} to {predictions['date'].max()}")
            
            # Check prediction quality
            pred_stats = predictions.groupby('neighborhood').agg({
                'predicted_requests': ['count', 'mean', 'std'],
                'confidence_lower': 'mean',
                'confidence_upper': 'mean'
            }).round(2)
            
            print("  Sample predictions by neighborhood:")
            for neighborhood in predictions['neighborhood'].unique()[:3]:
                nbhd_preds = predictions[predictions['neighborhood'] == neighborhood]
                avg_pred = nbhd_preds['predicted_requests'].mean()
                avg_uncertainty = (nbhd_preds['confidence_upper'] - nbhd_preds['confidence_lower']).mean()
                print(f"    {neighborhood}: avg={avg_pred:.1f}, uncertainty=±{avg_uncertainty/2:.1f}")
        else:
            print("❌ No predictions generated")
            
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
    
    print("\n=== Debug Complete ===")

if __name__ == "__main__":
    debug_historical_data()