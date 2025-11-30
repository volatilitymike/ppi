# import pandas as pd
# import numpy as np
# import streamlit as st
# import plotly.graph_objects as go


# # ============================================================
# #  STD EXPANSION + NORMALIZED "VOLATILITY PRICE" (STD-Mike)
# # ============================================================

# def apply_std_expansion(
#     df: pd.DataFrame,
#     window: int = 9,
#     anchor_lookback: int = 5,
# ) -> pd.DataFrame:
#     """
#     Convert STD movement of Mike into a normalized "volatility price".

#     Outputs:
#     - F%_STD              → rolling std of F_numeric
#     - STD_Anchor          → anchor STD (shifted)
#     - STD_Ratio           → current STD vs anchor STD
#     - STD_Alert / Level   → 🐦‍🔥 expansion flags on STD_Ratio
#     - STD_Mike            → normalized volatility price: (ratio - 1) * 100
#     - STD_Mike_MA         → Bollinger MA on STD-Mike
#     - STD_Mike_Upper      → upper BB on STD-Mike
#     - STD_Mike_Lower      → lower BB on STD-Mike
#     - STD_Mike_BBW        → BBW on STD-Mike (volatility of volatility)
#     - STD_Mike_BBW_Tight  → boolean compression flag
#     - STD_Mike_BBW_Emoji  → '🐝' when compression cluster detected
#     - STD_Tenkan          → intraday baseline of volatility
#     - STD_Kijun           → structure line of volatility
#     """

#     if df.empty or "F_numeric" not in df.columns:
#         df["F%_STD"] = df["STD_Anchor"] = df["STD_Ratio"] = np.nan
#         df["STD_Alert"] = ""
#         df["STD_Level"] = np.nan

#         df["STD_Mike"] = df["STD_Mike_MA"] = np.nan
#         df["STD_Mike_Upper"] = df["STD_Mike_Lower"] = np.nan
#         df["STD_Mike_BBW"] = np.nan

#         df["STD_Mike_BBW_Tight"] = False
#         df["STD_Mike_BBW_Emoji"] = ""

#         df["STD_Tenkan"] = df["STD_Kijun"] = np.nan
#         return df

#     # ------------------------
#     # 1) Rolling std of Mike
#     # ------------------------
#     df["F%_STD"] = df["F_numeric"].rolling(window=window, min_periods=1).std()

#     # ------------------------
#     # 2) Anchor & ratio
#     # ------------------------
#     df["STD_Anchor"] = df["F%_STD"].shift(anchor_lookback)
#     df["STD_Ratio"] = df["F%_STD"] / df["STD_Anchor"]

#     ratio = df["STD_Ratio"].replace([np.inf, -np.inf], np.nan)

#     # ------------------------
#     # 3) Expansion alerts on STD_Ratio
#     # ------------------------
#     df["STD_Alert"] = ""
#     df["STD_Level"] = np.nan

#     exp_mask = ratio >= 2
#     df.loc[exp_mask, "STD_Alert"] = "🐦‍🔥"
#     df.loc[exp_mask, "STD_Level"] = ratio

#     # ------------------------
#     # 4) STD-Mike (Volatility Price)
#     # ------------------------
#     df["STD_Mike"] = (ratio - 1) * 100  # 0 = no expansion, +100 ≈ 2x STD

#     # ------------------------
#     # 5) Bollinger Bands on STD-Mike
#     # ------------------------
#     std_mike = df["STD_Mike"].replace([np.inf, -np.inf], np.nan)

#     df["STD_Mike_MA"] = std_mike.rolling(window=window, min_periods=1).mean()
#     std_dev = std_mike.rolling(window=window, min_periods=1).std()

#     df["STD_Mike_Upper"] = df["STD_Mike_MA"] + 2 * std_dev
#     df["STD_Mike_Lower"] = df["STD_Mike_MA"] - 2 * std_dev

#     df["STD_Mike_BBW"] = (
#         (df["STD_Mike_Upper"] - df["STD_Mike_Lower"]) / df["STD_Mike_MA"]
#     ).replace([np.inf, -np.inf], np.nan)

#     # ------------------------
#     # 5b) BBW compression on STD-Mike (🐝)
#     # ------------------------
#     df = detect_std_mike_bbw_tight(
#         df,
#         window=5,
#         percentile_threshold=10.0,
#     )

#     # ------------------------
#     # 6) Tenkan / Kijun on STD-Mike
#     # ------------------------
#     df["STD_Tenkan"] = (
#         df["STD_Mike"].rolling(9, min_periods=1).max()
#         + df["STD_Mike"].rolling(9, min_periods=1).min()
#     ) / 2

#     df["STD_Kijun"] = (
#         df["STD_Mike"].rolling(26, min_periods=1).max()
#         + df["STD_Mike"].rolling(26, min_periods=1).min()
#     ) / 2

#     return df


# # ============================================================
# #  BBW COMPRESSION ON STD-MIKE (🐝)
# # ============================================================

# def detect_std_mike_bbw_tight(
#     df: pd.DataFrame,
#     window: int = 5,
#     percentile_threshold: float = 10.0,
# ) -> pd.DataFrame:
#     """
#     BBW compression on STD-Mike:

#     - dynamic_threshold = Xth percentile of STD_Mike_BBW
#     - STD_Mike_BBW_Tight = STD_Mike_BBW < dynamic_threshold
#     - STD_Mike_BBW_Emoji = '🐝' if at least 3 of last `window` bars are tight
#     """
#     if "STD_Mike_BBW" not in df.columns or df["STD_Mike_BBW"].dropna().empty:
#         df["STD_Mike_BBW_Tight"] = False
#         df["STD_Mike_BBW_Emoji"] = ""
#         return df

#     vals = df["STD_Mike_BBW"].dropna()
#     threshold = np.percentile(vals, percentile_threshold)

#     df["STD_Mike_BBW_Tight"] = df["STD_Mike_BBW"] < threshold
#     df["STD_Mike_BBW_Emoji"] = ""

#     for i in range(window, len(df)):
#         recent = df["STD_Mike_BBW_Tight"].iloc[i - window : i]
#         if recent.sum() >= 3:
#             df.at[df.index[i], "STD_Mike_BBW_Emoji"] = "🐝"

#     return df


# # ============================================================
# #  VISUAL COMPONENT
# # ============================================================

# def render_std_component(df: pd.DataFrame, ticker: str):
#     """Snapshot + STD-Mike Volatility Chart (BB, Tenkan, Kijun, 🐝, Hover Time)."""

#     if df.empty or "STD_Mike" not in df.columns:
#         st.info("No STD-Mike data available.")
#         return

#     st.subheader(f"📊 {ticker} — STD-Mike Volatility Overview")

#     # ------------------------
#     # Snapshot
#     # ------------------------
#     last_std = df["F%_STD"].iloc[-1]
#     last_ratio = df["STD_Ratio"].iloc[-1]
#     last_lvl = df["STD_Level"].iloc[-1]
#     last_alert = df["STD_Alert"].iloc[-1]
#     last_std_mike = df["STD_Mike"].iloc[-1]

#     last_bbw = df["STD_Mike_BBW"].iloc[-1] if "STD_Mike_BBW" in df.columns else np.nan

#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("STD Now", f"{last_std:.2f}")
#     c2.metric("Ratio", f"{last_ratio:.2f}")
#     c3.metric(
#         "Expansion",
#         f"{last_lvl:.2f}x" if not pd.isna(last_lvl) else "—",
#         delta=last_alert,
#     )
#     c4.metric("Volatility Price (STD-Mike)", f"{last_std_mike:.1f}")

#     c5, c6, _, _ = st.columns(4)
#     c5.metric("STD-Mike BBW", f"{last_bbw:.2f}" if not pd.isna(last_bbw) else "—")
#     if "STD_Mike_BBW_Emoji" in df.columns:
#         c6.metric("Compression", df["STD_Mike_BBW_Emoji"].iloc[-1] or "—")

#     st.divider()

#     # ------------------------
#     # BUILD PLOT
#     # ------------------------
#     needed_cols = [
#         "Time",
#         "STD_Mike",
#         "STD_Mike_MA",
#         "STD_Mike_Upper",
#         "STD_Mike_Lower",
#         "STD_Tenkan",
#         "STD_Kijun",
#     ]
#     plot_df = df[needed_cols].copy()
#     plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["STD_Mike"])

#     st.markdown("### **STD-Mike (Volatility Price) with Bollinger, Ichimoku & Bees**")

#     fig = go.Figure()

#     # MAIN LINE
#     fig.add_trace(
#         go.Scatter(
#             x=plot_df["Time"],
#             y=plot_df["STD_Mike"],
#             mode="lines",
#             name="STD-Mike",
#             line=dict(width=2, color="#1f77b4"),
#             hovertemplate="Time: %{x}<br>STD-Mike: %{y:.2f}<extra></extra>",
#         )
#     )

#     # UPPER BAND
#     fig.add_trace(
#         go.Scatter(
#             x=plot_df["Time"],
#             y=plot_df["STD_Mike_Upper"],
#             mode="lines",
#             name="Upper Band",
#             line=dict(width=1, dash="dot", color="gray"),
#             hovertemplate="Time: %{x}<br>Upper: %{y:.2f}<extra></extra>",
#         )
#     )

#     # LOWER BAND
#     fig.add_trace(
#         go.Scatter(
#             x=plot_df["Time"],
#             y=plot_df["STD_Mike_Lower"],
#             mode="lines",
#             name="Lower Band",
#             line=dict(width=1, dash="dot", color="gray"),
#             hovertemplate="Time: %{x}<br>Lower: %{y:.2f}<extra></extra>",
#         )
#     )

#     # MA (MIDDLE BAND)
#     fig.add_trace(
#         go.Scatter(
#             x=plot_df["Time"],
#             y=plot_df["STD_Mike_MA"],
#             mode="lines",
#             name="Middle Band",
#             line=dict(width=1.4, dash="dash", color="gray"),
#             hovertemplate="Time: %{x}<br>MA: %{y:.2f}<extra></extra>",
#         )
#     )

#     # TENKAN
#     fig.add_trace(
#         go.Scatter(
#             x=plot_df["Time"],
#             y=plot_df["STD_Tenkan"],
#             mode="lines",
#             name="Tenkan",
#             line=dict(width=1, dash="dot", color="red"),
#             hovertemplate="Time: %{x}<br>Tenkan: %{y:.2f}<extra></extra>",
#         )
#     )

#     # KIJUN
#     fig.add_trace(
#         go.Scatter(
#             x=plot_df["Time"],
#             y=plot_df["STD_Kijun"],
#             mode="lines",
#             name="Kijun",
#             line=dict(width=1, dash="dash", color="green"),
#             hovertemplate="Time: %{x}<br>Kijun: %{y:.2f}<extra></extra>",
#         )
#     )

#     # 🐝 BBW compression markers on STD-Mike
#     if "STD_Mike_BBW_Emoji" in df.columns:
#         bee_mask = df["STD_Mike_BBW_Emoji"] == "🐝"
#         if bee_mask.any():
#             bees_df = df.loc[bee_mask].copy()
#             bees_df = bees_df.merge(
#                 df[["Time", "STD_Mike"]],
#                 on="Time",
#                 how="left",
#                 suffixes=("", "_val"),
#             )
#             fig.add_trace(
#                 go.Scatter(
#                     x=bees_df["Time"],
#                     y=bees_df["STD_Mike"],
#                     mode="text",
#                     text=["🐝"] * len(bees_df),
#                     textposition="top center",
#                     name="STD-Mike BBW Tight",
#                     hovertemplate=(
#                         "Time: %{x}<br>"
#                         "STD-Mike: %{y:.2f}<br>"
#                         "BBW Compression 🐝"
#                         "<extra></extra>"
#                     ),
#                 )
#             )

#     fig.update_layout(
#         height=260,
#         margin=dict(l=20, r=20, t=20, b=40),
#         showlegend=True,
#     )

#     st.plotly_chart(fig, use_container_width=True)



# pages/components/stdExpansion.py

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# 1) CORE: STD → STD-Mike (Volatility Price)
# ============================================================
def apply_std_expansion(
    df: pd.DataFrame,
    window: int = 9,
    anchor_lookback: int = 5,
    bb_window: int = 20,     # classic BB window
    bb_mult: float = 2.0,    # standard deviation multiplier
) -> pd.DataFrame:

    # Guard
    if "F_numeric" not in df.columns or df.empty:
        df["F%_STD"] = np.nan
        df["STD_Mike"] = np.nan
        return df

    # ------------------------------------------------------------
    # (A) Base rolling standard deviation (Mike's volatility)
    # ------------------------------------------------------------
    df["F%_STD"] = df["F_numeric"].rolling(window=window, min_periods=1).std()

    # ------------------------------------------------------------
    # (B) Convert STD → STD-Mike normalized volatility price
    #     STD-Mike = ((curr_std / std_lookback) - 1) * 100
    # ------------------------------------------------------------
    df["STD_Anchor"] = df["F%_STD"].shift(anchor_lookback)
    ratio = df["F%_STD"] / df["STD_Anchor"]
    df["STD_Mike"] = (ratio - 1.0) * 100.0
    df["STD_Mike"] = df["STD_Mike"].replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------
    # (C) Bollinger Bands on STD-Mike (Volatility Price)
    # ------------------------------------------------------------
    sm = df["STD_Mike"]

    df["STD_BB_MA"] = sm.rolling(bb_window, min_periods=5).mean()
    df["STD_BB_STD"] = sm.rolling(bb_window, min_periods=5).std()

    df["STD_BB_Upper"] = df["STD_BB_MA"] + bb_mult * df["STD_BB_STD"]
    df["STD_BB_Lower"] = df["STD_BB_MA"] - bb_mult * df["STD_BB_STD"]

    # ------------------------------------------------------------
    # (D) BBW on STD-Mike (Volatility of Volatility)
    # ------------------------------------------------------------
    df["STD_Mike_BBW"] = (
        (df["STD_BB_Upper"] - df["STD_BB_Lower"]) / df["STD_BB_MA"]
    ).replace([np.inf, -np.inf], np.nan)

    # 🐝 extremely compressed volatility-volatility
    df["STD_Mike_Bee"] = np.where(
        (df["STD_Mike_BBW"] < 0.20) & (df["STD_Mike_BBW"].notna()),
        "🐝",
        "",
    )

    # 🔥 expansion when BBW explodes upward
    df["STD_Mike_Flare"] = np.where(
        (df["STD_Mike_BBW"] > df["STD_Mike_BBW"].shift(1) * 1.5),
        "🔥",
        "",
    )

    # ------------------------------------------------------------
    # (E) Tenkan / Kijun on STD-Mike (Volatility Trend)
    # ------------------------------------------------------------
    sm_clean = df["STD_Mike"].replace([np.inf, -np.inf], np.nan)

    # Tenkan 9 midpoint
    high_9 = sm_clean.rolling(9, min_periods=1).max()
    low_9 = sm_clean.rolling(9, min_periods=1).min()
    df["STD_Tenkan"] = (high_9 + low_9) / 2.0

    # Kijun 26 midpoint
    high_26 = sm_clean.rolling(26, min_periods=1).max()
    low_26 = sm_clean.rolling(26, min_periods=1).min()
    df["STD_Kijun"] = (high_26 + low_26) / 2.0

    return df


# ============================================================
# 2) PLOT COMPONENT
# ============================================================
def render_std_component(df: pd.DataFrame, ticker: str):
    if df.empty or "STD_Mike" not in df.columns:
        st.info("No STD-Mike data available.")
        return

    st.subheader(f"📊 {ticker} — STD-Mike Volatility Engine")

    st.divider()

    # ------------------------------------------------------------
    # PLOT STD-Mike with BB, Tenkan, Kijun
    # ------------------------------------------------------------
    plot_df = df[
        ["Time", "STD_Mike", "STD_BB_MA", "STD_BB_Upper", "STD_BB_Lower",
         "STD_Tenkan", "STD_Kijun", "STD_Mike_Bee", "STD_Mike_Flare"]
    ].copy()

    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["STD_Mike"])

    if plot_df.empty:
        st.info("Not enough data to plot STD-Mike.")
        return

    st.markdown("### **STD-Mike (Volatility Price) with BB, BBW & Trend Lines**")

    fig = go.Figure()

    # Main STD-Mike
    fig.add_trace(
        go.Scatter(
            x=plot_df["Time"], y=plot_df["STD_Mike"],
            mode="lines", name="STD-Mike",
            line=dict(width=2, color="#2ecc71"),
            hovertemplate="Time: %{x}<br>STD-Mike: %{y:.2f}<extra></extra>",
        )
    )

    # Upper / Lower / MA (Bollinger)
    fig.add_trace(go.Scatter(
        x=plot_df["Time"], y=plot_df["STD_BB_Upper"],
        mode="lines", name="BB Upper",
        line=dict(width=1, dash="dot", color="gray"),
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["Time"], y=plot_df["STD_BB_Lower"],
        mode="lines", name="BB Lower",
        line=dict(width=1, dash="dot", color="gray"),
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["Time"], y=plot_df["STD_BB_MA"],
        mode="lines", name="BB MA",
        line=dict(width=1.5, dash="dash", color="white"),
    ))

    # Tenkan
    fig.add_trace(go.Scatter(
        x=plot_df["Time"], y=plot_df["STD_Tenkan"],
        mode="lines", name="STD Tenkan",
        line=dict(width=1, dash="dot", color="red"),
    ))

    # Kijun
    fig.add_trace(go.Scatter(
        x=plot_df["Time"], y=plot_df["STD_Kijun"],
        mode="lines", name="STD Kijun",
        line=dict(width=1, dash="dash", color="green"),
    ))

    # 🐝 compression markers
    bee_mask = plot_df["STD_Mike_Bee"] == "🐝"
    if bee_mask.any():
        fig.add_trace(go.Scatter(
            x=plot_df.loc[bee_mask, "Time"],
            y=plot_df.loc[bee_mask, "STD_Mike"],
            mode="text", text="🐝", name="Volatility Compression",
            textposition="top center", textfont=dict(size=18),
        ))

    # 🔥 expansion markers
    flare_mask = plot_df["STD_Mike_Flare"] == "🔥"
    if flare_mask.any():
        fig.add_trace(go.Scatter(
            x=plot_df.loc[flare_mask, "Time"],
            y=plot_df.loc[flare_mask, "STD_Mike"],
            mode="text", text="🔥", name="Volatility Expansion",
            textposition="bottom center", textfont=dict(size=18),
        ))

    fig.update_layout(
        height=270,
        margin=dict(l=20, r=20, t=20, b=40),
        legend=dict(orientation="h", y=-0.25),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------
    # BBW PANEL (raw volatility-of-volatility)
    # ------------------------------------------------------------
    if "STD_Mike_BBW" in df.columns:
        bbw_df = df[["Time", "STD_Mike_BBW"]].dropna()

        st.markdown("### **STD-Mike BBW (Volatility-of-Volatility)**")

        bbw_fig = go.Figure()
        bbw_fig.add_trace(go.Scatter(
            x=bbw_df["Time"],
            y=bbw_df["STD_Mike_BBW"],
            mode="lines",
            line=dict(width=2, color="#f39c12"),
            name="STD-Mike BBW",
            hovertemplate="Time: %{x}<br>BBW: %{y:.3f}<extra></extra>",
        ))

        bbw_fig.update_layout(
            height=170,
            margin=dict(l=20, r=20, t=20, b=40),
        )

        st.plotly_chart(bbw_fig, use_container_width=True)
