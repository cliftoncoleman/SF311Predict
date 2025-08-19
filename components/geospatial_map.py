# components/geospatial_map.py
import os
import json
import urllib.request
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

SF_CENTER = {"lat": 37.7749, "lon": -122.4194}
DEFAULT_LOCAL_GEOJSON = "data/sf_neighborhoods.geojson"

# Primary: Socrata Geospatial Export (works when enabled)
REMOTE_SF_GEOJSON_PRIMARY = (
    "https://data.sfgov.org/api/geospatial/6ia5-2f8k?method=export&format=GeoJSON"
)
# Fallback A: GitHub (Click That Hood — stable, permissive)
REMOTE_SF_GEOJSON_FALLBACK_A = (
    "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/san-francisco.geojson"
)
# Fallback B: GitHub (blackmad/neighborhoods)
REMOTE_SF_GEOJSON_FALLBACK_B = (
    "https://raw.githubusercontent.com/blackmad/neighborhoods/master/sf.geojson"
)


# -------- Cached loader (works for local OR remote) --------
@st.cache_data(show_spinner=False, ttl=24 * 3600)
def _load_geojson_cached(local_path: str) -> Tuple[dict, str]:
    """
    Return (geojson, source_str). Prefer local file; then try a sequence of remotes.
    """
    # 0) Local file first (offline-friendly)
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f), f"(local) {local_path}"

    # 1) Remote attempts in order
    urls = [
        REMOTE_SF_GEOJSON_PRIMARY,
        REMOTE_SF_GEOJSON_FALLBACK_A,
        REMOTE_SF_GEOJSON_FALLBACK_B,
    ]

    headers = {"User-Agent": "Mozilla/5.0"}
    app_token = os.environ.get("SOCRATA_APP_TOKEN")
    if app_token:
        headers["X-App-Token"] = app_token

    last_err = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=25) as resp:
                text = resp.read().decode("utf-8")
                gj = json.loads(text)
                # quick sanity check
                if isinstance(gj, dict) and "features" in gj:
                    return gj, f"(remote) {url}"
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        "Could not load SF neighborhoods GeoJSON from remote sources.\n"
        f"Tried: {urls}\n"
        f"Last error: {last_err}\n"
        f"Fix: download once and save to '{local_path}' (create the 'data' folder)."
    )



class GeospatialMapComponent:
    """
    Plotly choropleth mapbox for SF neighborhoods.

    Expected df columns:
      - 'date' (datetime or parseable)
      - 'neighborhood' (string)
      - value_field (default 'predicted_requests')

    Call:
      GeospatialMapComponent().render_map_component(df, title="...", value_field="predicted_requests")
    """

    def __init__(self, local_geojson_path: str = DEFAULT_LOCAL_GEOJSON):
        self.local_geojson_path = local_geojson_path

    # ---------- internal helpers ----------
    def _featureidkey_for(self, gj: dict) -> str:
        """Pick the best properties key for neighborhood name."""
        props = gj["features"][0]["properties"]
        candidates = ["name", "neighborhood", "Neighborhood", "nhood", "NHOOD"]
        for c in candidates:
            if c in props:
                return f"properties.{c}"
        # Fallback to first property
        return f"properties.{list(props.keys())[0]}"

    def _prop_key_from_featureidkey(self, featureidkey: str) -> str:
        return featureidkey.split(".", 1)[1]  # e.g. "properties.name" -> "name"

    def _norm(self, s: str) -> str:
        s = str(s or "").lower().strip()
        s = s.replace("&", " and ")
        for ch in "/-.'":
            s = s.replace(ch, " ")
        s = " ".join(s.split())
        return s

    def _alias_map(self) -> Dict[str, str]:
        """Common alt names → canonical SFFind names."""
        return {
            self._norm("South Of Market"): "South of Market",
            self._norm("Castro"): "Castro/Upper Market",
            self._norm("Financial District"): "Financial District/South Beach",
            self._norm("South Beach"): "Financial District/South Beach",
            self._norm("Bayview Hunters Point"): "Bayview",
            self._norm("Haight-Ashbury"): "Haight Ashbury",
            self._norm("Western Addition NOPA"): "Western Addition",
            # Add more aliases here if you see unmatched names in the expander
        }

    def _to_geo_name(self, s: str, geo_map: Dict[str, str]) -> str | None:
        """Normalize a neighborhood name and map it to a canonical GeoJSON name."""
        n = self._norm(s)
        # try direct
        if n in geo_map:
            return geo_map[n]
        # try alias
        alias = self._alias_map().get(n)
        if alias:
            n2 = self._norm(alias)
            return geo_map.get(n2)
        return None

    def _build_geo_map(self, gj: dict, prop_key: str) -> Dict[str, str]:
        """normalized_name -> canonical_name in GeoJSON."""
        out = {}
        for feat in gj["features"]:
            name = str(feat["properties"][prop_key])
            out[self._norm(name)] = name
        return out

    # ---------- public ----------
    def render_map_component(
        self,
        df: pd.DataFrame,
        title: str = "Daily Geospatial Heatmap - All Neighborhoods",
        value_field: str = "predicted_requests",
        color_scale: str = "YlOrRd",
        key: str = "geo_map",    
    ):
        # Basic guards
        if df is None or df.empty:
            st.info("No prediction data available for the map.")
            return

        needed = {"date", "neighborhood", value_field}
        missing = needed - set(df.columns)
        if missing:
            st.error(f"Map requires these columns in the DataFrame: {sorted(needed)}. Missing: {sorted(missing)}")
            return

        # Standardize types
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df[value_field] = pd.to_numeric(df[value_field], errors="coerce").fillna(0)

        # Load/calc GeoJSON and mapping
        try:
            with st.spinner("Loading neighborhood boundaries..."):
                gj, src = _load_geojson_cached(self.local_geojson_path)
        except Exception as e:
            st.error("Neighborhood boundaries failed to load.")
            with st.expander("Details"):
                st.write(str(e))
            st.info(
                "Fix: download a GeoJSON once and save it to "
                f"**{self.local_geojson_path}** (create the `data` folder). "
                "For example, you can use:\n"
                "- Click-That-Hood: "
                "`https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/san-francisco.geojson`\n"
                "- Or blackmad/neighborhoods: "
                "`https://raw.githubusercontent.com/blackmad/neighborhoods/master/sf.geojson`"
            )
            return
        featureidkey = self._featureidkey_for(gj)
        prop_key = self._prop_key_from_featureidkey(featureidkey)
        geo_map = self._build_geo_map(gj, prop_key)
        all_geo_names = [str(f["properties"][prop_key]) for f in gj["features"]]

        # ---------- UI Header ----------
        st.subheader(f"🗺️ {title}")
    

        # ----- build date list -----
        dates = sorted(df["date"].dt.date.dropna().unique().tolist())
        if not dates:
            st.info("No valid dates present for mapping.")
            return

        # namespaced state keys
        k_date = f"{key}__date"
        k_prev = f"{key}__prev"
        k_next = f"{key}__next"

        # init / clamp
        if k_date not in st.session_state or st.session_state[k_date] not in dates:
            st.session_state[k_date] = dates[0]

        cur_idx = dates.index(st.session_state[k_date])

        # layout: Prev | Select | Next
        col_prev, col_sel, col_next = st.columns([1, 3, 1])

        with col_prev:
            prev_clicked = st.button("◀ Prev Day", key=k_prev, use_container_width=True)
        with col_next:
            next_clicked = st.button("Next Day ▶", key=k_next, use_container_width=True)

        # apply clicks (no st.rerun needed)
        if prev_clicked:
            cur_idx = max(0, cur_idx - 1)
            st.session_state[k_date] = dates[cur_idx]
        elif next_clicked:
            cur_idx = min(len(dates) - 1, cur_idx + 1)
            st.session_state[k_date] = dates[cur_idx]

        # recompute index after potential change
        cur_idx = dates.index(st.session_state[k_date])

        with col_sel:
            st.selectbox(
                "Select Date:",
                options=dates,
                index=cur_idx,
                key=k_date,  # namespaced
                format_func=lambda d: f"{d} ({pd.Timestamp(d).day_name()})",
            )

        # use the chosen day
        day = pd.to_datetime(st.session_state[k_date])
        day_df = df[df["date"].dt.date == day.date()].copy()


        # Aggregate to one row per neighborhood
        day_df = (
            day_df.groupby("neighborhood", as_index=False)[value_field]
            .sum()
            .rename(columns={"neighborhood": "nbhd"})
        )

        # Map names to GeoJSON canonical
        day_df["geo_name"] = day_df["nbhd"].apply(lambda s: self._to_geo_name(s, geo_map))

        # Show unmatched for quick fixes
        unmatched = day_df[day_df["geo_name"].isna()]
        if not unmatched.empty:
            with st.expander("Unmatched neighborhoods (add aliases to fix)", expanded=False):
                st.dataframe(unmatched[["nbhd"]], use_container_width=True, hide_index=True)

        # Build full frame so all polygons render (zeros where missing)
        base = pd.DataFrame({"geo_name": all_geo_names})
        plot_df = base.merge(day_df[["geo_name", value_field]], on="geo_name", how="left")
        plot_df[value_field] = plot_df[value_field].fillna(0)

        # Metrics
        daily_total = float(plot_df[value_field].sum())
        active_n = int((plot_df[value_field] > 0).sum())
        avg_per = float(plot_df[value_field].mean())
        m1, m2, m3 = st.columns(3)
        m1.metric("Daily Total", f"{daily_total:,.1f}")
        m2.metric("Active Neighborhoods", f"{active_n}")
        m3.metric("Avg per Neighborhood", f"{avg_per:.1f}")

        # Choropleth
        vmax = max(1.0, float(plot_df[value_field].max()))
        fig = px.choropleth_mapbox(
            plot_df,
            geojson=gj,
            locations="geo_name",
            featureidkey=featureidkey,           # e.g., "properties.name"
            color=value_field,
            color_continuous_scale=color_scale,  # Plotly scale (no callables)
            range_color=(0, vmax),
            mapbox_style="carto-positron",       # no token needed
            center=SF_CENTER,
            zoom=10,
            opacity=0.70,
            hover_name="geo_name",
            hover_data={value_field: ":.1f"},
            labels={value_field: "Predicted"},
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_colorbar=dict(title="Predicted"),
        )

        st.plotly_chart(fig, use_container_width=True)

