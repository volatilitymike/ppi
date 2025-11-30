# pages/components/stdExpansion.py

import pandas as pd
import numpy as np
import streamlit as st


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
    """STD snapshot + histogram + normalized STD-Mike line plot."""
    if df.empty or "STD_Ratio" not in df.columns:
        st.info("No STD data available.")
        return

    st.subheader(f"📊 {ticker} — STD Expansion Overview")

    # --- Snapshot (last values) ---
    last_std = float(df["F%_STD"].iloc[-1]) if "F%_STD" in df.columns else np.nan
    last_anchor = float(df["STD_Anchor"].iloc[-1]) if "STD_Anchor" in df.columns else np.nan
    last_ratio = float(df["STD_Ratio"].iloc[-1]) if "STD_Ratio" in df.columns else np.nan
    last_level = df["STD_Level"].iloc[-1] if "STD_Level" in df.columns else np.nan
    last_alert = df["STD_Alert"].iloc[-1] if "STD_Alert" in df.columns else ""

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("STD Now", f"{last_std:.2f}" if not np.isnan(last_std) else "—")
    c2.metric("Anchor", f"{last_anchor:.2f}" if not np.isnan(last_anchor) else "—")
    c3.metric("Ratio", f"{last_ratio:.2f}" if not np.isnan(last_ratio) else "—")
    c4.metric(
        "Level",
        f"{last_level:.2f}x" if not np.isnan(last_level) else "—",
        delta=last_alert,
    )

    st.divider()

    # --- Histogram of STD_Ratio ---
    clean_ratio = (
        df["STD_Ratio"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        if "STD_Ratio" in df.columns
        else pd.Series(dtype=float)
    )

    if not clean_ratio.empty:
        hist_vals, bins = np.histogram(clean_ratio, bins=10)

        hist_df = pd.DataFrame(
            {
                "Bin": [f"{bins[i]:.2f} → {bins[i + 1]:.2f}" for i in range(len(hist_vals))],
                "Count": hist_vals,
            }
        )

        st.bar_chart(hist_df, x="Bin", y="Count", height=220)
    else:
        st.info("Not enough data to build STD ratio histogram.")

    st.divider()

    # --- Normalized STD-Mike line plot ---
    if "STD_Mike" not in df.columns:
        st.info("No normalized STD-Mike series available.")
        return

    norm_series = (
        df["STD_Mike"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if norm_series.empty:
        st.info("Not enough data to plot normalized STD-Mike.")
        return

    st.markdown("**Normalized STD-Mike (Volatility Path)**")
    # Use index as x-axis (time-ordered), values as y
    st.line_chart(norm_series, height=200)
