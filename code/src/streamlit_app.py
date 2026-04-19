import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from predict import get_recession_risk_for_date, get_probability_window

from components import (
    inject_global_styles,
    render_page_header,
    render_risk_card,
    render_interpretation_banner,
    render_reference_row,
    render_signal_legend,
    render_section_title,
)

# =========================
# Page Config (MOVE THIS TO TOP)
# =========================

st.set_page_config(
    page_title="Recession Risk Dashboard",
    page_icon="📉",
    layout="wide"
)

# =========================
# Global Styling + Header
# =========================

inject_global_styles()

render_page_header(
    "📉 Economic Recession Risk Dashboard",
    "Estimate recession risk across 3, 6, and 12 month windows using calibrated LightGBM models."
)

# =========================
# Sidebar Controls
# =========================

st.sidebar.header("Controls")

selected_date = st.sidebar.date_input(
    "Choose a date",
    value=pd.Timestamp("2008-09-30")
)

months_before = st.sidebar.slider(
    "Months before selected date",
    min_value=12,
    max_value=120,
    value=36,
    step=12
)

months_after = st.sidebar.slider(
    "Months after selected date",
    min_value=0,
    max_value=60,
    value=12,
    step=12
)

run_button = st.sidebar.button("Analyze Risk")

# =========================
# Plot Function
# =========================

def plot_probability_window(window_df: pd.DataFrame, selected_date: pd.Timestamp):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(window_df.index, window_df["Risk_3M"], label="3M")
    ax.plot(window_df.index, window_df["Risk_6M"], label="6M")
    ax.plot(window_df.index, window_df["Risk_12M"], label="12M")

    ax.axvline(selected_date, linestyle="--", linewidth=2, label="Selected Date")

    ax.set_title("Historical Recession Risk")
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    st.pyplot(fig)

# =========================
# Main App
# =========================

if run_button:
    try:
        result = get_recession_risk_for_date(str(selected_date))
        reference_date = pd.to_datetime(result["ReferenceDateUsed"])

        render_section_title(
            f"Risk Analysis for {selected_date}",
            f"Reference monthly row used: {result['ReferenceDateUsed']}"
        )

        render_reference_row(result["ReferenceDateUsed"])

        st.markdown("")

        col1, col2, col3 = st.columns(3)

        with col1:
            render_risk_card(
                "3-Month Window",
                result["Risk_3M"],
                result["Label_3M"],
                result["Threshold_3M"]
            )

        with col2:
            render_risk_card(
                "6-Month Window",
                result["Risk_6M"],
                result["Label_6M"],
                result["Threshold_6M"]
            )

        with col3:
            render_risk_card(
                "12-Month Window",
                result["Risk_12M"],
                result["Label_12M"],
                result["Threshold_12M"]
            )

        st.markdown("")

        render_interpretation_banner(result["OverallInterpretation"])

        render_signal_legend()

        # =========================
        # Summary Table
        # =========================

        render_section_title(
            "Model Signal Summary",
            "Probability outputs and alert status across each horizon."
        )

        summary_df = pd.DataFrame({
            "Window": ["3 Months", "6 Months", "12 Months"],
            "Probability": [
                f"{result['Risk_3M']:.1%}",
                f"{result['Risk_6M']:.1%}",
                f"{result['Risk_12M']:.1%}",
            ],
            "Label": [
                result["Label_3M"],
                result["Label_6M"],
                result["Label_12M"],
            ],
            "Above Threshold": [
                result["AboveThreshold_3M"],
                result["AboveThreshold_6M"],
                result["AboveThreshold_12M"],
            ],
        })

        st.dataframe(summary_df, use_container_width=True)

        # =========================
        # Visualization
        # =========================

        render_section_title(
            "Historical Context",
            "Recession risk before and after the selected date."
        )

        window_df = get_probability_window(
            center_date=str(selected_date),
            months_before=months_before,
            months_after=months_after
        )

        plot_probability_window(window_df, reference_date)

    except Exception as e:
        st.error(f"Error: {e}")

else:
    render_section_title("How to Use")

    st.markdown(
        """
        1. Choose a date in the sidebar  
        2. Click **Analyze Risk**  
        3. Review multi-horizon risk outputs  
        4. Inspect historical behavior in the chart  
        """
    )

    render_section_title("Suggested Demo Dates")

    demo_dates = pd.DataFrame({
        "Scenario": [
            "Pre-GFC warning period",
            "GFC crisis peak",
            "COVID crash",
            "Stable expansion",
        ],
        "Date": [
            "2007-11-30",
            "2008-09-30",
            "2020-04-30",
            "2013-06-30",
        ]
    })

    st.dataframe(demo_dates, use_container_width=True)