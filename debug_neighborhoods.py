#!/usr/bin/env python3
"""
Debug script to check actual neighborhood column names and data in SF311 API
"""

import requests
import json

def check_sf311_columns():
    """Check what neighborhood columns are actually available"""
    
    app_token = "TuXFZRAF7T8dnb1Rqk5VOdOKN"
    meta_url = "https://data.sfgov.org/api/views/vw6y-z8j6?content=metadata"
    
    try:
        response = requests.get(meta_url, headers={"X-App-Token": app_token}, timeout=30)
        response.raise_for_status()
        
        meta = response.json()
        
        # Find neighborhood-related columns
        neighborhood_columns = []
        for col in meta.get("columns", []):
            field_name = col.get("fieldName", "")
            if "neigh" in field_name.lower():
                neighborhood_columns.append({
                    "fieldName": field_name,
                    "name": col.get("name", ""),
                    "description": col.get("description", "")
                })
        
        print("=== NEIGHBORHOOD COLUMNS FOUND ===")
        for col in neighborhood_columns:
            print(f"Field: {col['fieldName']}")
            print(f"Name: {col['name']}")
            print(f"Description: {col['description']}")
            print("-" * 50)
        
        return [col['fieldName'] for col in neighborhood_columns]
        
    except Exception as e:
        print(f"Error fetching metadata: {e}")
        return []

def check_sample_data(neighborhood_fields):
    """Check sample data for neighborhood fields"""
    
    if not neighborhood_fields:
        print("No neighborhood fields to check")
        return
    
    app_token = "TuXFZRAF7T8dnb1Rqk5VOdOKN"
    base_url = "https://data.sfgov.org/resource/vw6y-z8j6.json"
    
    # Build select clause
    select_fields = ",".join(neighborhood_fields)
    
    params = {
        "$select": select_fields,
        "$where": "service_name = 'Street and Sidewalk Cleaning'",
        "$limit": 20
    }
    
    try:
        response = requests.get(base_url, params=params, headers={"X-App-Token": app_token}, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        print(f"\n=== SAMPLE DATA ({len(data)} records) ===")
        for i, record in enumerate(data):
            print(f"Record {i+1}:")
            for field in neighborhood_fields:
                value = record.get(field, "NOT_PRESENT")
                print(f"  {field}: {value}")
            print("-" * 30)
            
        # Count unique values per field
        print("\n=== UNIQUE VALUES COUNT ===")
        for field in neighborhood_fields:
            values = set()
            for record in data:
                val = record.get(field)
                if val:
                    values.add(val)
            print(f"{field}: {len(values)} unique values")
            if len(values) <= 10:
                print(f"  Values: {sorted(values)}")
            print()
        
    except Exception as e:
        print(f"Error fetching sample data: {e}")

if __name__ == "__main__":
    print("SF311 Neighborhood Debug Script")
    print("=" * 50)
    
    neighborhood_fields = check_sf311_columns()
    
    if neighborhood_fields:
        check_sample_data(neighborhood_fields)
    else:
        print("No neighborhood fields found - checking if API is accessible")
        
        # Try a basic query
        try:
            response = requests.get("https://data.sfgov.org/resource/vw6y-z8j6.json?$limit=1", timeout=30)
            print(f"API Status: {response.status_code}")
            if response.status_code == 200:
                print("API is accessible, but metadata fetch failed")
            else:
                print(f"API error: {response.text[:200]}")
        except Exception as e:
            print(f"API connection failed: {e}")