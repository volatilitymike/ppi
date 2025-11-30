# pages/components/stdExpansion.py

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go


# ======================================================
# 1) CORE STD EXPANSION ON MIKE (same base as before)
#    + STD-Mike Bollinger / BBW suite
# ======================================================
def apply_std_expansion(
    df: pd.DataFrame,
    window: int = 9,
    anchor_lookback: int = 5,
    # STD-Mike Bollinger / BBW knobs
    std_bb_window: int = 20,
    std_bbw_scale_factor: float = 1.0,
    std_bbw_tight_window: int = 5,
    std_bbw_percentile_threshold: float = 10.0,
    std_bbw_anchor_lookback: int = 5,
) -> pd.DataFrame:
    """
    F% STD Expansion on Mike axis (F_numeric):

    - F%_STD      = rolling std of F_numeric over `window`
    - STD_Anchor  = F%_STD shifted by `anchor_lookback` bars
    - STD_Ratio   = F%_STD / STD_Anchor
    - STD_Alert   = '🐦‍🔥' when STD_Ratio >= 2 (double or triple expansion)
    - STD_Mike    = normalized STD: (STD_Ratio - 1) * 100  (Mike-style scale)
    - PLUS: Bollinger / BBW stack on STD-Mike:
        * STD_BB_MA, STD_BB_Upper, STD_BB_Lower
        * STD_BBW (width of STD-Mike bands)
        * STD_BBW_Tight + STD_BBW_Tight_Emoji (🐝)
        * STD_BBW_Anchor, STD_BBW_Ratio, STD_BBW_Alert (🔥)
    """

    if "F_numeric" not in df.columns or df.empty:
        # Ensure all columns exist even when no data
        df["F%_STD"] = np.nan
        df["STD_Anchor"] = np.nan
        df["STD_Ratio"] = np.nan
        df["STD_Alert"] = ""
        df["STD_Level"] = np.nan
        df["STD_Mike"] = np.nan

        df["STD_BB_MA"] = np.nan
        df["STD_BB_Std"] = np.nan
        df["STD_BB_Upper"] = np.nan
        df["STD_BB_Lower"] = np.nan

        df["STD_BBW"] = 0.0
        df["STD_BBW_Tight"] = False
        df["STD_BBW_Tight_Emoji"] = ""
        df["STD_BBW_Anchor"] = np.nan
        df["STD_BBW_Ratio"] = np.nan
        df["STD_BBW_Alert"] = ""
        return df

    # 1) rolling std of F_numeric
    df["F%_STD"] = df["F_numeric"].rolling(window=window, min_periods=1).std()

    # 2) anchor and ratio
    df["STD_Anchor"] = df["F%_STD"].shift(anchor_lookback)
    df["STD_Ratio"] = df["F%_STD"] / df["STD_Anchor"]

    # 3) STD Expansion Levels on F%_STD
    df["STD_Alert"] = ""
    df["STD_Level"] = np.nan  # numeric magnitude (2x, 3x, …)

    valid_ratio = df["STD_Ratio"].replace([np.inf, -np.inf], np.nan).notna()
    exp_mask = valid_ratio & (df["STD_Ratio"] >= 2)

    df.loc[exp_mask, "STD_Alert"] = "🐦‍🔥"
    df.loc[exp_mask, "STD_Level"] = df["STD_Ratio"]

    # 4) Normalized STD-Mike (volatility in Mike units)
    df["STD_Mike"] = np.nan
    df.loc[valid_ratio, "STD_Mike"] = (df.loc[valid_ratio, "STD_Ratio"] - 1.0) * 100.0

    # 5) Run the STD-Mike Bollinger / BBW suite (idea lifted from F% BBW)
    df = calculate_std_mike_bands(df, window=std_bb_window)
    df = calculate_std_mike_bbw(df, scale_factor=std_bbw_scale_factor)
    df = detect_std_mike_bbw_tight(
        df,
        window=std_bbw_tight_window,
        percentile_threshold=std_bbw_percentile_threshold,
    )
    df = add_std_mike_bbw_anchor_and_ratio(df, lookback=std_bbw_anchor_lookback)

    return df


# ======================================================
# 2) STD-MIKE BOLLINGER + BBW + TIGHT + EXPANSION
# ======================================================
def calculate_std_mike_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Bollinger-style bands on STD-Mike:

        STD_BB_MA    = rolling mean of STD_Mike
        STD_BB_Std   = rolling std of STD_Mike
        STD_BB_Upper = STD_BB_MA + 2 * STD_BB_Std
        STD_BB_Lower = STD_BB_MA - 2 * STD_BB_Std
    """
    if "STD_Mike" not in df.columns or df.empty:
        df["STD_BB_MA"] = np.nan
        df["STD_BB_Std"] = np.nan
        df["STD_BB_Upper"] = np.nan
        df["STD_BB_Lower"] = np.nan
        return df

    roll_mean = df["STD_Mike"].rolling(window=window, min_periods=5).mean()
    roll_std = df["STD_Mike"].rolling(window=window, min_periods=5).std()

    df["STD_BB_MA"] = roll_mean
    df["STD_BB_Std"] = roll_std
    df["STD_BB_Upper"] = roll_mean + 2.0 * roll_std
    df["STD_BB_Lower"] = roll_mean - 2.0 * roll_std

    return df


def calculate_std_mike_bbw(df: pd.DataFrame, scale_factor: float = 1.0) -> pd.DataFrame:
    """
    Computes Bollinger Band Width (BBW) for STD-Mike:

        STD_BBW_raw = STD_BB_Upper - STD_BB_Lower      (pure width in STD-Mike units)
        STD_BBW     = STD_BBW_raw / scale_factor

    We *don't* divide by |middle| here because STD_Mike
    naturally oscillates around 0 and the middle can be near 0.
    """
    if not {"STD_BB_Upper", "STD_BB_Lower", "STD_BB_MA"}.issubset(df.columns):
        df["STD_BBW"] = 0.0
        return df

    width = (df["STD_BB_Upper"] - df["STD_BB_Lower"]).abs()
    bbw_scaled = (width / scale_factor).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["STD_BBW"] = bbw_scaled
    return df


def detect_std_mike_bbw_tight(
    df: pd.DataFrame,
    window: int = 5,
    percentile_threshold: float = 10.0,
) -> pd.DataFrame:
    """
    Detects BBW Tight Compression on STD-Mike using a dynamic threshold
    based on the ticker's own STD_BBW distribution.

    - dynamic_threshold = Xth percentile of non-null STD_BBW
    - STD_BBW_Tight = STD_BBW < dynamic_threshold
    - STD_BBW_Tight_Emoji = '🐝' when at least 3 of last `window` bars are tight
    """
    if "STD_BBW" not in df.columns or df["STD_BBW"].dropna().empty:
        df["STD_BBW_Tight"] = False
        df["STD_BBW_Tight_Emoji"] = ""
        return df

    dynamic_threshold = np.percentile(df["STD_BBW"].dropna(), percentile_threshold)

    df["STD_BBW_Tight"] = df["STD_BBW"] < dynamic_threshold
    df["STD_BBW_Tight_Emoji"] = ""

    for i in range(window, len(df)):
        recent = df["STD_BBW_Tight"].iloc[i - window : i]
        if recent.sum() >= 3:
            df.at[df.index[i], "STD_BBW_Tight_Emoji"] = "🐝"

    return df


def add_std_mike_bbw_anchor_and_ratio(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Adds:
        - STD_BBW_Anchor = STD_BBW shifted by `lookback`
        - STD_BBW_Ratio  = STD_BBW / STD_BBW_Anchor
        - STD_BBW_Alert  = '🔥' when STD_BBW_Ratio >= 2 (double+ expansion)
    """
    if "STD_BBW" not in df.columns or df.empty:
        df["STD_BBW_Anchor"] = np.nan
        df["STD_BBW_Ratio"] = np.nan
        df["STD_BBW_Alert"] = ""
        return df

    df["STD_BBW_Anchor"] = df["STD_BBW"].shift(lookback)
    anchor = df["STD_BBW_Anchor"].replace(0, np.nan)
    df["STD_BBW_Ratio"] = df["STD_BBW"] / anchor

    def _std_bbw_alert(row):
        ratio = row["STD_BBW_Ratio"]
        if pd.isna(ratio):
            return ""
        if ratio >= 2:
            return "🔥"  # Double+ expansion of volatility envelope
        return ""

    df["STD_BBW_Alert"] = df.apply(_std_bbw_alert, axis=1)
    return df


# ======================================================
# 3) RENDER COMPONENT (STD snapshot + STD-Mike + bands)
# ======================================================
def render_std_component(df: pd.DataFrame, ticker: str):
    """STD snapshot + (optionally) STD-Mike bands / BBW line plot."""
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

    last_bbw = float(df["STD_BBW"].iloc[-1]) if "STD_BBW" in df.columns else np.nan
    last_bbw_anchor = float(df["STD_BBW_Anchor"].iloc[-1]) if "STD_BBW_Anchor" in df.columns else np.nan
    last_bbw_ratio = float(df["STD_BBW_Ratio"].iloc[-1]) if "STD_BBW_Ratio" in df.columns else np.nan
    last_bbw_alert = df["STD_BBW_Alert"].iloc[-1] if "STD_BBW_Alert" in df.columns else ""

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("STD Now", f"{last_std:.2f}" if not np.isnan(last_std) else "—")
    c2.metric("Anchor", f"{last_anchor:.2f}" if not np.isnan(last_anchor) else "—")
    c3.metric("Ratio", f"{last_ratio:.2f}" if not np.isnan(last_ratio) else "—")
    c4.metric(
        "Level",
        f"{last_level:.2f}x" if not np.isnan(last_level) else "—",
        delta=last_alert,
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("STD-Mike BBW", f"{last_bbw:.2f}" if not np.isnan(last_bbw) else "—")
    c6.metric("BBW Anchor", f"{last_bbw_anchor:.2f}" if not np.isnan(last_bbw_anchor) else "—")
    c7.metric("BBW Ratio", f"{last_bbw_ratio:.2f}" if not np.isnan(last_bbw_ratio) else "—")
    c8.metric("BBW Alert", last_bbw_alert or "—")

    st.divider()

    # --- Normalized STD-Mike + bands (Time on hover via Plotly) ---
    if "STD_Mike" not in df.columns or "Time" not in df.columns:
        st.info("No normalized STD-Mike series available.")
        return

    plot_df = df[["Time", "STD_Mike", "STD_BB_MA", "STD_BB_Upper", "STD_BB_Lower"]].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["STD_Mike"])

    if plot_df.empty:
        st.info("Not enough data to plot normalized STD-Mike.")
        return

    st.markdown("**Normalized STD-Mike (Volatility Path) + BBW context**")

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
    if {"STD_BB_MA", "STD_BB_Upper", "STD_BB_Lower"}.issubset(plot_df.columns):
        bb_df = plot_df.dropna(subset=["STD_BB_MA"])

        if not bb_df.empty:
            # Upper band
            fig.add_trace(
                go.Scatter(
                    x=bb_df["Time"],
                    y=bb_df["STD_BB_Upper"],
                    mode="lines",
                    name="STD BB Upper",
                    line=dict(width=1, dash="dot"),
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
                    line=dict(width=1, dash="dot"),
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
                    line=dict(width=1.5, dash="dash"),
                    hovertemplate="Time: %{x}<br>STD BB MA: %{y:.2f}<extra></extra>",
                )
            )
                # --- STD-Mike BBW (volatility-of-vol) ---
    if "STD_BBW" not in df.columns or "Time" not in df.columns:
        return

    bbw_df = df[["Time", "STD_BBW", "STD_BBW_Tight_Emoji", "STD_BBW_Alert"]].copy()
    bbw_df = bbw_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["STD_BBW"])

    if bbw_df.empty:
        return

    st.markdown("**STD-Mike BBW (Compression / Expansion)**")

    fig_bbw = go.Figure()

    # Main BBW line
    fig_bbw.add_trace(
        go.Scatter(
            x=bbw_df["Time"],
            y=bbw_df["STD_BBW"],
            mode="lines",
            name="STD-Mike BBW",
            hovertemplate=(
                "Time: %{x}<br>"
                "STD-BBW: %{y:.2f}"
                "<extra></extra>"
            ),
        )
    )

    # 🐝 compression markers (on top of line)
    tight_mask = bbw_df["STD_BBW_Tight_Emoji"] == "🐝"
    if tight_mask.any():
        fig_bbw.add_trace(
            go.Scatter(
                x=bbw_df.loc[tight_mask, "Time"],
                y=bbw_df.loc[tight_mask, "STD_BBW"],
                mode="text",
                text=["🐝"] * int(tight_mask.sum()),
                textposition="top center",
                name="STD-BBW Tight",
                hovertemplate=(
                    "Time: %{x}<br>"
                    "STD-BBW Tight 🐝"
                    "<extra></extra>"
                ),
            )
        )

    # 🔥 expansion markers
    hot_mask = bbw_df["STD_BBW_Alert"] == "🔥"
    if hot_mask.any():
        fig_bbw.add_trace(
            go.Scatter(
                x=bbw_df.loc[hot_mask, "Time"],
                y=bbw_df.loc[hot_mask, "STD_BBW"],
                mode="text",
                text=["🔥"] * int(hot_mask.sum()),
                textposition="bottom center",
                name="STD-BBW Expansion",
                hovertemplate=(
                    "Time: %{x}<br>"
                    "STD-BBW Expansion 🔥"
                    "<extra></extra>"
                ),
            )
        )

    fig_bbw.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=20, b=40),
    )

    st.plotly_chart(fig_bbw, use_container_width=True)


    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=20, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)
