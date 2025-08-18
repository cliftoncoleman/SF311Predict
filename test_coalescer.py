#!/usr/bin/env python3
"""
Test the neighborhood coalescer with real SF311 data
"""

import pandas as pd
from neighborhood_coalescer import apply_neighborhood_coalescing

# Create test data similar to what we see from the API
test_data = pd.DataFrame({
    'requested_datetime': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05', '2025-01-06'],
    'analysis_neighborhood': ['Mission', 'Mission', 'Bayview Hunters Point', None, 'Tenderloin', 'Castro/Upper Market'],
    'neighborhoods_sffind_boundaries': ['Mission', 'Mission Dolores', 'Hunters Point', 'Panhandle', 'Tenderloin', 'Dolores Heights'],
    'request_count': [5, 3, 8, 2, 4, 6]
})

print("=== TEST DATA ===")
print(test_data)
print()

print("=== APPLYING NEIGHBORHOOD COALESCING ===")
result_df, diagnostics = apply_neighborhood_coalescing(test_data, verbose=True)

print("\n=== RESULT ===")
print(result_df[['analysis_neighborhood', 'neighborhoods_sffind_boundaries', 'neighborhood']])

print(f"\n=== DIAGNOSTICS ===")
for key, value in diagnostics.items():
    print(f"{key}: {value}")

print(f"\n=== FINAL NEIGHBORHOOD COUNTS ===")
neighborhood_counts = result_df['neighborhood'].value_counts()
print(neighborhood_counts)