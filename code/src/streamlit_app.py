import os

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI

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
# Page Config
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

# Extra styling to make the AI explanation button stand out
st.markdown(
    """
    <style>
    div.stButton > button[kind="primary"] {
        font-size: 1.05rem;
        font-weight: 700;
        padding: 0.9rem 1.2rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

render_page_header(
    "📉 Economic Recession Risk Dashboard",
    "Estimate recession risk across 3, 6, and 12 month windows using calibrated LightGBM models."
)

# =========================
# Session State
# =========================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analysis_selected_date" not in st.session_state:
    st.session_state.analysis_selected_date = None

if "ai_explanation" not in st.session_state:
    st.session_state.ai_explanation = None

if "ai_explanation_generated_for" not in st.session_state:
    st.session_state.ai_explanation_generated_for = None

# =========================
# OpenAI Helper
# =========================

@st.cache_resource
def get_openai_client():
    """
    Create a cached OpenAI client so Streamlit does not rebuild it on every rerun.
    Returns None if the API key is missing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    print(api_key)
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def generate_llm_explanation(requested_date: str, result: dict) -> str:
    """
    Uses an LLM to generate a short explanation of the risk outputs.
    """
    client = get_openai_client()
    if client is None:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment before using AI explanations."
        )

    prompt = f"""
You are helping explain a recession-risk dashboard to a user.

Important constraints:
- Write 3 to 5 sentences.
- Be clear, concise, and non-technical.
- Use the supplied model outputs as evidence.
- The bulk of your analysis should center around using historical/macroeconomic context for the requested date to explain outputs.
- Do NOT invent exact internal model features that were not provided.
- Do NOT give financial or investment advice.
- If the horizons disagree, mention that disagreement explicitly.

Here is the dashboard data:

Requested date: {requested_date}
Reference monthly row actually used by the model: {result["ReferenceDateUsed"]}
Overall interpretation: {result["OverallInterpretation"]}

3-month probability: {result["Risk_3M"]:.4f}
3-month label: {result["Label_3M"]}
3-month threshold: {result["Threshold_3M"]:.4f}
3-month above threshold: {result["AboveThreshold_3M"]}

6-month probability: {result["Risk_6M"]:.4f}
6-month label: {result["Label_6M"]}
6-month threshold: {result["Threshold_6M"]:.4f}
6-month above threshold: {result["AboveThreshold_6M"]}

12-month probability: {result["Risk_12M"]:.4f}
12-month label: {result["Label_12M"]}
12-month threshold: {result["Threshold_12M"]:.4f}
12-month above threshold: {result["AboveThreshold_12M"]}

Return only the explanation text.
""".strip()

    response = client.responses.create(
        model="gpt-5.4",
        instructions=(
            "You explain macroeconomic model outputs carefully and conservatively. "
            "Prefer plain English over jargon."
        ),
        input=prompt,
        max_output_tokens=250,
    )

    explanation = response.output_text.strip()
    if not explanation:
        raise RuntimeError("The model returned an empty explanation.")
    return explanation


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

        st.session_state.analysis_result = result
        st.session_state.analysis_selected_date = str(selected_date)

        # Reset explanation whenever a new analysis is run
        st.session_state.ai_explanation = None
        st.session_state.ai_explanation_generated_for = None

    except Exception as e:
        st.error(f"Error: {e}")

if st.session_state.analysis_result is not None:
    try:
        result = st.session_state.analysis_result
        analyzed_date = st.session_state.analysis_selected_date
        reference_date = pd.to_datetime(result["ReferenceDateUsed"])

        render_section_title(
            f"Risk Analysis for {analyzed_date}",
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

        # =========================
        # AI Explanation Trigger
        # =========================

        st.markdown("---")
        st.subheader("AI Explanation")
        st.markdown(
            "Generate a contextual interpretation of the 3M, 6M, and 12M recession probabilities after reviewing the model outputs."
        )

        button_col1, button_col2, button_col3 = st.columns([1, 1.4, 1])

        with button_col2:
            generate_ai_button = st.button(
                "Generate AI Explanation",
                use_container_width=True
            )

        if generate_ai_button:
            with st.spinner("Generating explanation..."):
                try:
                    explanation = generate_llm_explanation(
                        requested_date=analyzed_date,
                        result=result
                    )
                    st.session_state.ai_explanation = explanation
                    st.session_state.ai_explanation_generated_for = analyzed_date
                except Exception as llm_error:
                    st.session_state.ai_explanation = None
                    st.warning(f"Could not generate AI explanation: {llm_error}")

        if (
            st.session_state.ai_explanation
            and st.session_state.ai_explanation_generated_for == analyzed_date
        ):
            st.info(st.session_state.ai_explanation)

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
            center_date=analyzed_date,
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
        4. Click **Generate AI Explanation**  
        5. Inspect historical behavior in the chart  
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