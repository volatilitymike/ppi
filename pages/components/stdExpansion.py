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


    # 5) Bollinger Bands on STD-Mike
    bb_window = 20      # you can tweak this
    bb_mult = 2.0       # standard 2σ bands

    roll_mean = df["STD_Mike"].rolling(window=bb_window, min_periods=5).mean()
    roll_std  = df["STD_Mike"].rolling(window=bb_window, min_periods=5).std()

    df["STD_BB_MA"]     = roll_mean
    df["STD_BB_Upper"]  = roll_mean + bb_mult * roll_std
    df["STD_BB_Lower"]  = roll_mean - bb_mult * roll_std
    # 5) Tenkan & Kijun on STD-Mike
    df["STD_Tenkan"] = np.nan
    df["STD_Kijun"] = np.nan

    if df["STD_Mike"].notna().any():
        # Tenkan = midpoint of highest/lowest STD_Mike over last 9 bars
        high_9 = df["STD_Mike"].rolling(window=9, min_periods=1).max()
        low_9 = df["STD_Mike"].rolling(window=9, min_periods=1).min()
        df["STD_Tenkan"] = (high_9 + low_9) / 2.0

        # Kijun = midpoint of highest/lowest STD_Mike over last 26 bars
        high_26 = df["STD_Mike"].rolling(window=26, min_periods=1).max()
        low_26 = df["STD_Mike"].rolling(window=26, min_periods=1).min()
        df["STD_Kijun"] = (high_26 + low_26) / 2.0

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

        # --- Normalized STD-Mike line plot (Time on hover via Plotly) ---
    if "STD_Mike" not in df.columns or "Time" not in df.columns:
        st.info("No normalized STD-Mike series available.")
        return

    plot_df = df[["Time", "STD_Mike", "STD_Tenkan", "STD_Kijun"]].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)

    # Drop rows where STD_Mike is NaN
    plot_df = plot_df.dropna(subset=["STD_Mike"])

    if plot_df.empty:
        st.info("Not enough data to plot normalized STD-Mike.")
        return

    st.markdown("**Normalized STD-Mike (Volatility Path)**")

    fig = go.Figure()

    # Main STD-Mike line
    fig.add_trace(
        go.Scatter(
            x=plot_df["Time"],
            y=plot_df["STD_Mike"],
            mode="lines",
            name="STD-Mike",
            hovertemplate=(
                "Time: %{x}<br>"
                "STD-Mike: %{y:.2f}"
                "<extra></extra>"
            ),
        )
    )
 # Bollinger on STD-Mike (if available)
    if {"STD_BB_MA", "STD_BB_Upper", "STD_BB_Lower"}.issubset(df.columns):
        bb_df = df[["Time", "STD_BB_MA", "STD_BB_Upper", "STD_BB_Lower"]].copy()
        bb_df = bb_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["STD_BB_MA"])

        if not bb_df.empty:
            # Upper band
            fig.add_trace(
                go.Scatter(
                    x=bb_df["Time"],
                    y=bb_df["STD_BB_Upper"],
                    mode="lines",
                    name="STD BB Upper",
                    line=dict(width=1, dash="line", color="gray"),
                    hovertemplate="Time: %{x}<br>STD BB Upper: %{y:.2f}<extra></extra>",
                )
            )

            # Lower band
            fig.add_trace(
                go.Scatter(
                    x=bb_df["Time"],
                    y=bb_df["STD_BB_Lower"],
                    mode="lines",
                    name="STD BB Lower",
                    line=dict(width=1, dash="line",color="gray"),
                    hovertemplate="Time: %{x}<br>STD BB Lower: %{y:.2f}<extra></extra>",
                )
            )

            # Middle band (MA)
            fig.add_trace(
                go.Scatter(
                    x=bb_df["Time"],
                    y=bb_df["STD_BB_MA"],
                    mode="lines",
                    name="STD BB MA",
                    line=dict(width=1.5, dash="dash",color="gray"),
                    hovertemplate="Time: %{x}<br>STD BB MA: %{y:.2f}<extra></extra>",
                )
            )

    # Tenkan on STD-Mike (if present)
    if "STD_Tenkan" in plot_df.columns:
        tenkan_df = plot_df.dropna(subset=["STD_Tenkan"])
        if not tenkan_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=tenkan_df["Time"],
                    y=tenkan_df["STD_Tenkan"],
                    mode="lines",
                    name="STD Tenkan",
                    line=dict(width=1, dash="dot", color="red"),
                    hovertemplate=(
                        "Time: %{x}<br>"
                        "STD Tenkan: %{y:.2f}"
                        "<extra></extra>"
                    ),
                )
            )

    # Kijun on STD-Mike (if present)
    if "STD_Kijun" in plot_df.columns:
        kijun_df = plot_df.dropna(subset=["STD_Kijun"])
        if not kijun_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=kijun_df["Time"],
                    y=kijun_df["STD_Kijun"],
                    mode="lines",
                    name="STD Kijun",
                    line=dict(width=1, dash="dash", color="green"),
                    hovertemplate=(
                        "Time: %{x}<br>"
                        "STD Kijun: %{y:.2f}"
                        "<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=20, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)
