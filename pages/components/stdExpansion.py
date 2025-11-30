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




def render_std_component(df: pd.DataFrame, ticker: str):
    """
    Minimal STD panel turned into:
    - Manual Implied Volatility selector
    - Numeric input + slider
    """

    st.subheader(f"📊 {ticker} — Manual IV Input")

    with st.expander("🎚️ Implied Volatility (Manual Entry)", expanded=False):

        st.markdown(
            """
            Enter the **live IV you see on Robinhood** for this ticker.
            This value becomes your intraday reference for energy, risk and momentum.
            """
        )

        # Numeric input + slider (linked)
        iv_value = st.number_input(
            "IV (from Robinhood)",
            min_value=0.0,
            max_value=300.0,
            value=30.0,
            step=1.0,
            help="Type the exact IV shown in Robinhood (e.g., 28.5, 34, 42).",
            key=f"iv_input_{ticker}"
        )

        iv_value = st.slider(
            "Adjust IV",
            min_value=0.0,
            max_value=300.0,
            value=float(iv_value),
            step=1.0,
            key=f"iv_slider_{ticker}"
        )

        st.info(f"**Current IV for {ticker}: {iv_value}%**")

        # Store for later use by other components
        st.session_state[f"{ticker}_iv"] = iv_value
