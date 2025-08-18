"""
Neighborhood coalescer for SF311 data.
Ensures all data uses Analysis Boundaries vocabulary (broader neighborhoods like 'Tenderloin')
rather than SFFind boundaries (micro-neighborhoods like 'Civic Center').
"""

import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, Optional, Tuple

# Standardized column names
ANALYSIS = "neighborhoods_analysis"  # Primary - broad neighborhoods
SFFIND = "neighborhoods_sffind_boundaries"      # Secondary - micro neighborhoods

def _normalize_text(s):
    """Normalize neighborhood text values"""
    if pd.isna(s):
        return None
    return " ".join(str(s).strip().split()).title()

def learn_sffind_to_analysis_mapping(df: pd.DataFrame, 
                                   analysis_col: str = ANALYSIS,
                                   sffind_col: str = SFFIND,
                                   min_votes: int = 1) -> Dict[str, str]:
    """
    Learn mapping from micro-neighborhoods (SFFind) to broad neighborhoods (Analysis)
    using co-occurrence in rows where both columns exist.
    
    For each SFFind value, pick the Analysis parent with highest co-occurrence count.
    """
    
    # Check if both columns exist
    required_cols = [analysis_col, sffind_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return {}
    
    # Get rows where both columns have valid data
    both_present = df[[analysis_col, sffind_col]].dropna()
    if both_present.empty:
        return {}
    
    # Count co-occurrences
    pairs = Counter(zip(
        both_present[sffind_col].map(_normalize_text),
        both_present[analysis_col].map(_normalize_text)
    ))
    
    # Group by sffind neighborhood and find most common analysis parent
    by_micro = defaultdict(lambda: Counter())
    for (micro, parent), count in pairs.items():
        by_micro[micro][parent] += count
    
    # Build mapping: micro -> most common parent
    mapping = {}
    for micro, parent_counts in by_micro.items():
        if parent_counts:
            parent, votes = parent_counts.most_common(1)[0]
            if votes >= min_votes:
                mapping[micro] = parent
    
    return mapping

def coalesce_to_analysis_boundaries(df: pd.DataFrame,
                                   analysis_col: str = ANALYSIS,
                                   sffind_col: str = SFFIND,
                                   external_mapping: Optional[Dict[str, str]] = None,
                                   fill_blanks: str = "Unknown") -> Tuple[pd.Series, Dict]:
    """
    Produce a single 'neighborhood' column using Analysis Boundaries vocabulary only.
    
    Priority order:
    1. If analysis_col exists & is not blank -> use it
    2. If sffind_col exists -> map to analysis via learned crosswalk
    3. Otherwise -> handle blanks per fill_blanks setting
    
    Parameters:
    -----------
    df : DataFrame with neighborhood columns
    analysis_col : str, column name for analysis boundaries (target vocab)
    sffind_col : str, column name for sffind boundaries (source for mapping)
    external_mapping : dict, optional pre-built SFFind->Analysis mapping
    fill_blanks : str, how to handle remaining blanks
        - "Unknown": fill with "Unknown" (safe for groupby)
        - None: leave as NaN
        - "drop": will be handled by caller
    
    Returns:
    --------
    neighborhood_series : pd.Series with standardized neighborhoods
    diagnostics : dict with processing information
    """
    
    # Check available columns
    available_cols = [col for col in [analysis_col, sffind_col] if col in df.columns]
    if not available_cols:
        raise ValueError(f"Need at least one of {analysis_col} or {sffind_col} columns")
    
    work_df = df.copy()
    
    # Normalize text in available columns
    for col in available_cols:
        work_df[col] = work_df[col].map(_normalize_text)
    
    # Start with analysis column if available
    if analysis_col in work_df.columns:
        result = work_df[analysis_col].copy()
    else:
        # Create empty series of same length
        result = pd.Series([None] * len(work_df), index=work_df.index, dtype=object)
    
    # Learn or use external mapping for SFFind -> Analysis
    if sffind_col in work_df.columns:
        if external_mapping:
            sffind_mapping = external_mapping
        else:
            sffind_mapping = learn_sffind_to_analysis_mapping(work_df, analysis_col, sffind_col)
        
        # Apply mapping to fill missing analysis values
        missing_mask = result.isna()
        if missing_mask.any() and sffind_mapping:
            sffind_values = work_df.loc[missing_mask, sffind_col]
            mapped_values = sffind_values.map(sffind_mapping)
            result.loc[mapped_values.index] = mapped_values
    else:
        sffind_mapping = {}
    
    # Handle remaining blanks
    remaining_blanks = result.isna()
    if remaining_blanks.any():
        if fill_blanks == "Unknown":
            result.loc[remaining_blanks] = "Unknown"
        elif fill_blanks is None:
            pass  # leave as NaN
        # "drop" case handled by caller
    
    # Compile diagnostics
    total_records = len(df)
    filled_from_sffind = 0
    if analysis_col in df.columns and sffind_col in df.columns and sffind_mapping:
        original_analysis = df[analysis_col].map(_normalize_text)
        filled_from_sffind = ((original_analysis != result) & original_analysis.isna()).sum()
    
    diagnostics = {
        "primary_column": analysis_col,
        "secondary_column": sffind_col if sffind_col in df.columns else None,
        "total_records": total_records,
        "had_analysis": (df[analysis_col].notna().sum() if analysis_col in df.columns else 0),
        "filled_from_sffind": int(filled_from_sffind),
        "still_missing": int(result.isna().sum()),
        "unique_neighborhoods": int(result.dropna().nunique()),
        "sffind_mapping_size": len(sffind_mapping),
        "coverage_percent": float((1 - result.isna().mean()) * 100)
    }
    
    return result.rename("neighborhood"), diagnostics

def apply_neighborhood_coalescing(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply neighborhood coalescing to a DataFrame.
    
    This is the main entry point for the coalescing process.
    """
    
    if df.empty:
        return df, {"error": "Empty DataFrame provided"}
    
    try:
        # Apply coalescing
        neighborhood_series, diagnostics = coalesce_to_analysis_boundaries(df)
        
        # Create result DataFrame
        result_df = df.copy()
        result_df["neighborhood"] = neighborhood_series
        
        if verbose:
            print(f"Neighborhood coalescing complete:")
            print(f"  - Primary column: {diagnostics['primary_column']}")
            print(f"  - Total records: {diagnostics['total_records']:,}")
            print(f"  - Had analysis data: {diagnostics['had_analysis']:,}")
            print(f"  - Filled from SFFind: {diagnostics['filled_from_sffind']:,}")
            print(f"  - Final coverage: {diagnostics['coverage_percent']:.1f}%")
            print(f"  - Unique neighborhoods: {diagnostics['unique_neighborhoods']}")
        
        return result_df, diagnostics
        
    except Exception as e:
        error_info = {"error": str(e)}
        if verbose:
            print(f"Error in neighborhood coalescing: {e}")
        return df, error_info

# Example usage function for testing
def example_usage():
    """Example of how to use the neighborhood coalescer"""
    
    # Sample data with mixed neighborhood columns
    sample_data = pd.DataFrame({
        'requested_datetime': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04'],
        'neighborhoods_analysis_boundaries': ['Tenderloin', None, 'Mission', None],
        'neighborhoods_sffind_boundaries': ['Civic Center', 'Lower Nob Hill', 'Mission Dolores', 'Castro District'],
        'request_count': [5, 3, 8, 2]
    })
    
    print("Original data:")
    print(sample_data)
    print("\nApplying neighborhood coalescing...")
    
    # Apply coalescing
    result_df, diagnostics = apply_neighborhood_coalescing(sample_data)
    
    print("\nResult:")
    print(result_df[['neighborhoods_analysis_boundaries', 'neighborhoods_sffind_boundaries', 'neighborhood']])
    
    return result_df, diagnostics

if __name__ == "__main__":
    example_usage()