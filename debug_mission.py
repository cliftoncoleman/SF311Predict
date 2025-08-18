#!/usr/bin/env python3
"""
Debug Mission neighborhood prediction issues
"""

import pandas as pd
import numpy as np
from fixed_pipeline import FixedSF311Pipeline

def debug_mission_predictions():
    """Debug why Mission predictions are wrong"""
    print("=== DEBUGGING MISSION PREDICTIONS ===")
    
    # Create pipeline
    pipeline = FixedSF311Pipeline()
    
    # Get historical data
    print("Fetching historical data...")
    historical_data = pipeline.fetch_historical_data(start_days=180)  # 6 months
    
    if historical_data.empty:
        print("ERROR: No historical data found")
        return
    
    print(f"Total historical records: {len(historical_data)}")
    print(f"Date range: {historical_data['date'].min()} to {historical_data['date'].max()}")
    print(f"Neighborhoods: {sorted(historical_data['neighborhood'].unique())}")
    
    # Check Mission specifically
    mission_data = historical_data[historical_data['neighborhood'] == 'MISSION'].copy()
    print(f"\n=== MISSION DATA ===")
    print(f"Mission records: {len(mission_data)}")
    
    if len(mission_data) == 0:
        # Try other case variations
        mission_variants = ['Mission', 'mission', 'MISSION', 'Mission District', 'Mission Bay']
        for variant in mission_variants:
            test_data = historical_data[historical_data['neighborhood'] == variant]
            if len(test_data) > 0:
                print(f"Found Mission data under variant: '{variant}' ({len(test_data)} records)")
                mission_data = test_data.copy()
                break
    
    if len(mission_data) == 0:
        print("ERROR: No Mission data found in any variant")
        # Show available neighborhoods
        print("Available neighborhoods:")
        for nbhd in sorted(historical_data['neighborhood'].unique()):
            count = len(historical_data[historical_data['neighborhood'] == nbhd])
            print(f"  {nbhd}: {count} records")
        return
    
    mission_data = mission_data.sort_values('date').reset_index(drop=True)
    print(f"Mission date range: {mission_data['date'].min()} to {mission_data['date'].max()}")
    print("Mission data sample:")
    print(mission_data.head(10))
    print("Mission recent data:")
    print(mission_data.tail(10))
    
    # Ensure continuous days
    mission_data = pipeline._ensure_continuous_days(mission_data)
    print(f"After ensuring continuous days: {len(mission_data)} records")
    
    # Check daily pattern
    mission_data['dow'] = pd.to_datetime(mission_data['date']).dt.dayofweek
    dow_avg = mission_data.groupby('dow')['cases'].mean()
    print("\nDay of week averages (0=Monday, 6=Sunday):")
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i, day in enumerate(days):
        print(f"  {day}: {dow_avg.get(i, 0):.1f}")
    
    # Test model selection
    print("\n=== MODEL SELECTION TEST ===")
    try:
        best_model = pipeline.backtest_and_select_model(mission_data)
        print(f"Selected model: {best_model['model_type']}")
        print(f"MASE score: {best_model.get('mase_score', 'N/A'):.3f}")
        print(f"Weekly repeat score: {best_model.get('weekly_repeat_score', 'N/A'):.3f}")
        print(f"Penalized MASE: {best_model.get('penalized_mase', 'N/A'):.3f}")
        
        # Show all candidates if available
        if 'all_candidates' in best_model:
            print("\nAll candidates:")
            for i, candidate in enumerate(best_model['all_candidates']):
                print(f"  {candidate['model_type']}: MASE={candidate.get('mase_score', 'N/A'):.3f}, "
                      f"Weekly_repeat={candidate.get('weekly_repeat_score', 'N/A'):.3f}, "
                      f"Penalized_MASE={candidate.get('penalized_mase', 'N/A'):.3f}")
        
    except Exception as e:
        print(f"ERROR in model selection: {e}")
        import traceback
        traceback.print_exc()
    
    # Test prediction generation
    print("\n=== PREDICTION TEST ===")
    try:
        forecast = pipeline._generate_forecast_from_model(
            best_model, mission_data, 'MISSION', 14  # 2 weeks
        )
        print("Generated forecast sample:")
        print(forecast.head(10))
        
        # Check for repetitive pattern
        predictions = forecast['predicted_requests'].values
        if len(predictions) >= 14:
            week1 = predictions[:7]
            week2 = predictions[7:14]
            diff = np.abs(week1 - week2)
            avg_diff = np.mean(diff)
            print(f"\nWeek 1 vs Week 2 average difference: {avg_diff:.2f}")
            if avg_diff < 1.0:
                print("WARNING: Predictions appear to be weekly repetitive!")
                print(f"Week 1: {week1}")
                print(f"Week 2: {week2}")
            else:
                print("Predictions appear to have variation between weeks")
        
    except Exception as e:
        print(f"ERROR in prediction generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_mission_predictions()