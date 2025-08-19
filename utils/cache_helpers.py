"""
Cache helpers for SF311 pipeline
"""
import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Set, List

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_field_names_cached(meta_url: str, app_token: str) -> Set[str]:
    """Get field names from SF311 API metadata - cached version"""
    try:
        session = requests.Session()
        session.headers.update({"X-App-Token": app_token})
        r = session.get(meta_url, timeout=60)
        r.raise_for_status()
        meta = r.json()
        return {c["fieldName"] for c in meta.get("columns", [])}
    except Exception:
        return set()

# Removed fetch_historical_data_cached due to circular import issues
# Caching will be handled differently at the session level