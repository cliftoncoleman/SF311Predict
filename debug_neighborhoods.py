#!/usr/bin/env python3

import sys
sys.path.append('.')
from fixed_pipeline import SF311Pipeline
from neighborhood_coalescer import apply_neighborhood_coalescing
from datetime import datetime, timedelta
import pandas as pd

def check_neighborhoods():
    print("*** CHECKING NEIGHBORHOOD NAMES ***")
    
    # Get recent data to check neighborhood names
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    try:
        pipeline = SF311Pipeline()
        data = pipeline.fetch_historical_data(30)
        
        neighborhoods = sorted(data['neighborhood'].unique())
        print(f'\nTotal neighborhoods: {len(neighborhoods)}')
        print('\nAll neighborhoods:')
        for i, n in enumerate(neighborhoods):
            print(f'{i+1:2d}. "{n}"')
        
        print('\nLooking for variations of "South of Market":')
        soma_variations = [n for n in neighborhoods if 'south' in n.lower() or 'soma' in n.lower() or 'market' in n.lower()]
        if soma_variations:
            for n in soma_variations:
                print(f'  - "{n}"')
        else:
            print('  - No matches found')
            
        print('\nChecking priority neighborhoods:')
        priority_neighborhoods = [
            "South of Market", "Tenderloin", "Hayes Valley", 
            "Mission", "Bayview Hunters Point"
        ]
        
        for priority in priority_neighborhoods:
            if priority in neighborhoods:
                print(f'  ✓ "{priority}" - FOUND')
            else:
                print(f'  ✗ "{priority}" - MISSING')
                # Look for similar names
                similar = [n for n in neighborhoods if any(word.lower() in n.lower() for word in priority.split())]
                if similar:
                    print(f'    Similar: {similar}')
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_neighborhoods()