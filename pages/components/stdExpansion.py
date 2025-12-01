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




# def render_std_expander(intraday_df, mike_col="F_numeric"):
#     """
#     SIMPLE STD LINE ONLY
#     - rolling STD of F_numeric (Mike)
#     - plot only STD as a clean line
#     - hover shows Time + STD value
#     """

#     with st.expander("Mike STD (Simple Volatility Line)", expanded=False):

#         # Slider for rolling window
#         window = st.slider(
#             "Rolling STD window (bars)",
#             5, 60, 20, step=5,
#             key=f"std_window_{mike_col}",
#         )

#         df = intraday_df.copy()

#         # Ensure Mike column exists
#         if mike_col not in df.columns:
#             st.warning(f"Column '{mike_col}' not found.")
#             return df

#         # Compute rolling STD
#         df["STD_Line"] = df[mike_col].rolling(window).std()

#         # X-axis: use Time if present, else index
#         if "Time" in df.columns:
#             x_vals = df["Time"]
#         else:
#             x_vals = df.index

#         # Build Plotly line with hover
#         fig = go.Figure()
#         fig.add_trace(
#             go.Scatter(
#                 x=x_vals,
#                 y=df["STD_Line"],
#                 mode="lines",
#                 name="Rolling STD",
#                 hovertemplate="Time: %{x}<br>STD: %{y:.2f}<extra></extra>",
#             )
#         )

#         fig.update_layout(
#             height=320,
#             margin=dict(l=10, r=10, t=40, b=40),
#             xaxis_title="Time",
#             yaxis_title="STD",
#         )

#         st.markdown("### 📉 STD (Volatility Intensity)")
#         st.plotly_chart(fig, use_container_width=True)

#         st.caption("Low → calm · Rising → tension · Spike → chaos · Falling → reset")

#         return df
