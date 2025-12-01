# pages/components/stdExpansion.py

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go  # ← add this


def apply_std_expansion(
    df: pd.DataFrame,
    window: int = 9,
    anchor_lookback: int = 5,
) -> pd.DataFrame:
    """
    F% STD Expansion on Mike axis (F_numeric):

    - F%_STD      = rolling std of F_numeric over `window`
    - STD_Anchor  = F%_STD shifted by `anchor_lookback` bars
    - STD_Ratio   = F%_STD / STD_Anchor
    - STD_Alert   = '🐦‍🔥' when STD_Ratio >= 2 (double or triple expansion)
    - STD_Mike    = normalized STD: (STD_Ratio - 1) * 100  (Mike-style scale)
    """

    if "F_numeric" not in df.columns or df.empty:
        df["F%_STD"] = np.nan
        df["STD_Anchor"] = np.nan
        df["STD_Ratio"] = np.nan
        df["STD_Alert"] = ""
        df["STD_Level"] = np.nan
        df["STD_Mike"] = np.nan
        return df

    # 1) rolling std of F_numeric
    df["F%_STD"] = df["F_numeric"].rolling(window=window, min_periods=1).std()

    # 2) anchor and ratio
    df["STD_Anchor"] = df["F%_STD"].shift(anchor_lookback)
    df["STD_Ratio"] = df["F%_STD"] / df["STD_Anchor"]

    # 3) STD Expansion Levels
    df["STD_Alert"] = ""
    df["STD_Level"] = np.nan  # numeric magnitude (2x, 3x, …)

    valid = df["STD_Ratio"].replace([np.inf, -np.inf], np.nan).notna()
    exp_mask = valid & (df["STD_Ratio"] >= 2)

    df.loc[exp_mask, "STD_Alert"] = "🐦‍🔥"
    df.loc[exp_mask, "STD_Level"] = df["STD_Ratio"]

    # 4) Normalized STD-Mike: baseline at 0 when ratio == 1
    #    Example:
    #      ratio 1.0 -> 0
    #      ratio 1.5 -> +50
    #      ratio 2.0 -> +100
    df["STD_Mike"] = np.nan
    df.loc[valid, "STD_Mike"] = (df.loc[valid, "STD_Ratio"] - 1.0) * 100.0

    return df






def render_std_expander(intraday_df, mike_col="Mike"):
    """
    STD / Sigma lab:
    - Rolling STD of Mike
    - Sigma = ΔMike / rolling STD
    - Simple spike-end detector
    """
    with st.expander("Mike STD Engine (Volatility Lab)", expanded=False):
        # 1) Sensitivity knob (window = how many bars define the environment)
        window = st.slider("Rolling STD window (bars)", 5, 60, 20, step=5)

        df = intraday_df.copy()
        df["mike"] = df[mike_col]
        df["mike_delta"] = df["mike"].diff()
        df["mike_std"] = df["mike"].rolling(window).std()
        df["mike_sigma"] = df["mike_delta"] / df["mike_std"]

        # 2) Mike vs STD (trend body vs volatility heartbeat)
        st.markdown("**Mike vs Rolling STD**")
        st.line_chart(df[["mike", "mike_std"]])

        # 3) Sigma histogram (spikes in standard deviations)
        st.markdown("**Sigma (ΔMike / Rolling STD)**")
        st.bar_chart(df["mike_sigma"])

        # 4) First-pass “trend ending” spike-collapse marker
        spike_level = st.select_slider(
            "Spike threshold (σ) for exhaustion",
            options=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            value=3.0,
        )

        df["SpikeEnd"] = (
            (df["mike_sigma"].abs() < spike_level) &
            (df["mike_sigma"].shift(1).abs() >= spike_level)
        )

        # Just to see it working for now; later you can plot emojis/markers on main chart
        spike_indices = list(df.index[df["SpikeEnd"]])
        if spike_indices:
            st.caption(f"Potential volatility exhaustion bars (SpikeEnd): {spike_indices[:15]}")

        return df
