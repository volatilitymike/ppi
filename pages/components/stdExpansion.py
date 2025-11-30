import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go


# ============================================================
#  STD EXPANSION + NORMALIZED "VOLATILITY PRICE" (STD-Mike)
# ============================================================

def apply_std_expansion(
    df: pd.DataFrame,
    window: int = 9,
    anchor_lookback: int = 5,
) -> pd.DataFrame:
    """
    Convert STD movement of Mike into a normalized "volatility price".

    Outputs:
    - F%_STD         → rolling std of F_numeric
    - STD_Ratio      → current STD vs anchor STD
    - STD_Mike       → normalized volatility price: (ratio - 1) * 100
    - STD_Mike_MA    → Bollinger MA on STD-Mike
    - STD_Mike_Upper → upper BB
    - STD_Mike_Lower → lower BB
    - STD_Mike_BBW   → BBW on STD-Mike (volatility of volatility)
    - STD_Tenkan     → intraday baseline of volatility
    - STD_Kijun      → structure line of volatility
    """

    if df.empty or "F_numeric" not in df.columns:
        df["F%_STD"] = df["STD_Anchor"] = df["STD_Ratio"] = np.nan
        df["STD_Alert"] = df["STD_Level"] = np.nan
        df["STD_Mike"] = df["STD_Mike_MA"] = np.nan
        df["STD_Mike_Upper"] = df["STD_Mike_Lower"] = np.nan
        df["STD_Mike_BBW"] = df["STD_Tenkan"] = df["STD_Kijun"] = np.nan
        return df

    # ------------------------
    # 1) Rolling std of Mike
    # ------------------------
    df["F%_STD"] = df["F_numeric"].rolling(window=window, min_periods=1).std()

    # ------------------------
    # 2) Anchor & ratio
    # ------------------------
    df["STD_Anchor"] = df["F%_STD"].shift(anchor_lookback)
    df["STD_Ratio"] = df["F%_STD"] / df["STD_Anchor"]

    ratio = df["STD_Ratio"].replace([np.inf, -np.inf], np.nan)

    # ------------------------
    # 3) Expansion alerts
    # ------------------------
    df["STD_Alert"] = ""
    df["STD_Level"] = np.nan

    exp_mask = ratio >= 2
    df.loc[exp_mask, "STD_Alert"] = "🐦‍🔥"
    df.loc[exp_mask, "STD_Level"] = ratio

    # ------------------------
    # 4) STD-Mike (Volatility Price)
    # ------------------------
    df["STD_Mike"] = (ratio - 1) * 100  # normalized so 0 = no expansion

    # ------------------------
    # 5) Bollinger Bands on STD-Mike
    # ------------------------
    std_mike = df["STD_Mike"].replace([np.inf, -np.inf], np.nan)

    df["STD_Mike_MA"] = std_mike.rolling(window=window, min_periods=1).mean()
    std_dev = std_mike.rolling(window=window, min_periods=1).std()

    df["STD_Mike_Upper"] = df["STD_Mike_MA"] + 2 * std_dev
    df["STD_Mike_Lower"] = df["STD_Mike_MA"] - 2 * std_dev

    # Bandwidth (volatility OF volatility)
    df["STD_Mike_BBW"] = (
        (df["STD_Mike_Upper"] - df["STD_Mike_Lower"]) / df["STD_Mike_MA"]
    ).replace([np.inf, -np.inf], np.nan)

    # ------------------------
    # 6) Tenkan / Kijun on STD-Mike
    # ------------------------
    df["STD_Tenkan"] = (
        df["STD_Mike"].rolling(9, min_periods=1).max() +
        df["STD_Mike"].rolling(9, min_periods=1).min()
    ) / 2

    df["STD_Kijun"] = (
        df["STD_Mike"].rolling(26, min_periods=1).max() +
        df["STD_Mike"].rolling(26, min_periods=1).min()
    ) / 2

    return df


# ============================================================
#  VISUAL COMPONENT
# ============================================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go


# ============================================================
#  STD EXPANSION + NORMALIZED "VOLATILITY PRICE" (STD-Mike)
# ============================================================

def apply_std_expansion(
    df: pd.DataFrame,
    window: int = 9,
    anchor_lookback: int = 5,
) -> pd.DataFrame:
    """
    Convert STD movement of Mike into a normalized "volatility price".

    Outputs:
    - F%_STD         → rolling std of F_numeric
    - STD_Ratio      → current STD vs anchor STD
    - STD_Mike       → normalized volatility price: (ratio - 1) * 100
    - STD_Mike_MA    → Bollinger MA on STD-Mike
    - STD_Mike_Upper → upper BB
    - STD_Mike_Lower → lower BB
    - STD_Mike_BBW   → BBW on STD-Mike (volatility of volatility)
    - STD_Tenkan     → intraday baseline of volatility
    - STD_Kijun      → structure line of volatility
    """

    if df.empty or "F_numeric" not in df.columns:
        df["F%_STD"] = df["STD_Anchor"] = df["STD_Ratio"] = np.nan
        df["STD_Alert"] = df["STD_Level"] = np.nan
        df["STD_Mike"] = df["STD_Mike_MA"] = np.nan
        df["STD_Mike_Upper"] = df["STD_Mike_Lower"] = np.nan
        df["STD_Mike_BBW"] = df["STD_Tenkan"] = df["STD_Kijun"] = np.nan
        return df

    # ------------------------
    # 1) Rolling std of Mike
    # ------------------------
    df["F%_STD"] = df["F_numeric"].rolling(window=window, min_periods=1).std()

    # ------------------------
    # 2) Anchor & ratio
    # ------------------------
    df["STD_Anchor"] = df["F%_STD"].shift(anchor_lookback)
    df["STD_Ratio"] = df["F%_STD"] / df["STD_Anchor"]

    ratio = df["STD_Ratio"].replace([np.inf, -np.inf], np.nan)

    # ------------------------
    # 3) Expansion alerts
    # ------------------------
    df["STD_Alert"] = ""
    df["STD_Level"] = np.nan

    exp_mask = ratio >= 2
    df.loc[exp_mask, "STD_Alert"] = "🐦‍🔥"
    df.loc[exp_mask, "STD_Level"] = ratio

    # ------------------------
    # 4) STD-Mike (Volatility Price)
    # ------------------------
    df["STD_Mike"] = (ratio - 1) * 100  # normalized so 0 = no expansion

    # ------------------------
    # 5) Bollinger Bands on STD-Mike
    # ------------------------
    std_mike = df["STD_Mike"].replace([np.inf, -np.inf], np.nan)

    df["STD_Mike_MA"] = std_mike.rolling(window=window, min_periods=1).mean()
    std_dev = std_mike.rolling(window=window, min_periods=1).std()

    df["STD_Mike_Upper"] = df["STD_Mike_MA"] + 2 * std_dev
    df["STD_Mike_Lower"] = df["STD_Mike_MA"] - 2 * std_dev

    # Bandwidth (volatility OF volatility)
    df["STD_Mike_BBW"] = (
        (df["STD_Mike_Upper"] - df["STD_Mike_Lower"]) / df["STD_Mike_MA"]
    ).replace([np.inf, -np.inf], np.nan)

    # ------------------------
    # 6) Tenkan / Kijun on STD-Mike
    # ------------------------
    df["STD_Tenkan"] = (
        df["STD_Mike"].rolling(9, min_periods=1).max() +
        df["STD_Mike"].rolling(9, min_periods=1).min()
    ) / 2

    df["STD_Kijun"] = (
        df["STD_Mike"].rolling(26, min_periods=1).max() +
        df["STD_Mike"].rolling(26, min_periods=1).min()
    ) / 2

    return df


# ============================================================
#  VISUAL COMPONENT
# ============================================================

def render_std_component(df: pd.DataFrame, ticker: str):
    """Snapshot + STD-Mike Volatility Chart (BB, Tenkan, Kijun, Hover Time)."""

    if df.empty or "STD_Mike" not in df.columns:
        st.info("No STD-Mike data available.")
        return

    st.subheader(f"📊 {ticker} — STD-Mike Volatility Overview")

    # ------------------------
    # Snapshot
    # ------------------------
    last_std = df["F%_STD"].iloc[-1]
    last_ratio = df["STD_Ratio"].iloc[-1]
    last_lvl = df["STD_Level"].iloc[-1]
    last_alert = df["STD_Alert"].iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("STD Now", f"{last_std:.2f}")
    c2.metric("Ratio", f"{last_ratio:.2f}")
    c3.metric("Expansion", f"{last_lvl:.2f}x" if not pd.isna(last_lvl) else "—",
              delta=last_alert)
    c4.metric("Volatility Price (STD-Mike)", f"{df['STD_Mike'].iloc[-1]:.1f}")

    st.divider()

    # ------------------------
    # BUILD PLOT
    # ------------------------
    plot_df = df[
        ["Time", "STD_Mike", "STD_Mike_MA",
         "STD_Mike_Upper", "STD_Mike_Lower",
         "STD_Tenkan", "STD_Kijun"]
    ].copy()

    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["STD_Mike"])

    st.markdown("### **STD-Mike (Volatility Price) with Bollinger & Ichimoku**")

    fig = go.Figure()

    # MAIN LINE
    fig.add_trace(go.Scatter(
        x=plot_df["Time"],
        y=plot_df["STD_Mike"],
        mode="lines",
        name="STD-Mike",
        line=dict(width=2, color="#1f77b4"),
        hovertemplate="Time: %{x}<br>STD-Mike: %{y:.2f}<extra></extra>",
    ))

    # UPPER BAND
    fig.add_trace(go.Scatter(
        x=plot_df["Time"],
        y=plot_df["STD_Mike_Upper"],
        mode="lines",
        name="Upper Band",
        line=dict(width=1, dash="dot", color="gray"),
        hovertemplate="Time: %{x}<br>Upper: %{y:.2f}<extra></extra>",
    ))

    # LOWER BAND
    fig.add_trace(go.Scatter(
        x=plot_df["Time"],
        y=plot_df["STD_Mike_Lower"],
        mode="lines",
        name="Lower Band",
        line=dict(width=1, dash="dot", color="gray"),
        hovertemplate="Time: %{x}<br>Lower: %{y:.2f}<extra></extra>",
    ))

    # MA (MIDDLE BAND)
    fig.add_trace(go.Scatter(
        x=plot_df["Time"],
        y=plot_df["STD_Mike_MA"],
        mode="lines",
        name="Middle Band",
        line=dict(width=1.4, dash="dash", color="gray"),
        hovertemplate="Time: %{x}<br>MA: %{y:.2f}<extra></extra>",
    ))

    # TENKAN
    fig.add_trace(go.Scatter(
        x=plot_df["Time"],
        y=plot_df["STD_Tenkan"],
        mode="lines",
        name="Tenkan",
        line=dict(width=1, dash="dot", color="red"),
        hovertemplate="Time: %{x}<br>Tenkan: %{y:.2f}<extra></extra>",
    ))

    # KIJUN
    fig.add_trace(go.Scatter(
        x=plot_df["Time"],
        y=plot_df["STD_Kijun"],
        mode="lines",
        name="Kijun",
        line=dict(width=1, dash="dash", color="green"),
        hovertemplate="Time: %{x}<br>Kijun: %{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=20, b=40),
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)
