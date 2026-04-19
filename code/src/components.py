import streamlit as st


# =========================
# Theme / Color Helpers
# =========================

def get_risk_palette(label: str) -> dict:
    """
    Returns a color palette for each risk label.
    """
    palettes = {
        "Low": {
            "accent": "#22c55e",
            "bg": "rgba(34, 197, 94, 0.12)",
            "border": "rgba(34, 197, 94, 0.45)",
            "text": "#bbf7d0",
            "pill_bg": "rgba(34, 197, 94, 0.18)",
        },
        "Elevated": {
            "accent": "#f59e0b",
            "bg": "rgba(245, 158, 11, 0.12)",
            "border": "rgba(245, 158, 11, 0.45)",
            "text": "#fde68a",
            "pill_bg": "rgba(245, 158, 11, 0.18)",
        },
        "Moderate": {
            "accent": "#fb923c",
            "bg": "rgba(251, 146, 60, 0.12)",
            "border": "rgba(251, 146, 60, 0.45)",
            "text": "#fed7aa",
            "pill_bg": "rgba(251, 146, 60, 0.18)",
        },
        "High": {
            "accent": "#ef4444",
            "bg": "rgba(239, 68, 68, 0.12)",
            "border": "rgba(239, 68, 68, 0.45)",
            "text": "#fecaca",
            "pill_bg": "rgba(239, 68, 68, 0.18)",
        },
        "Severe": {
            "accent": "#dc2626",
            "bg": "rgba(220, 38, 38, 0.14)",
            "border": "rgba(220, 38, 38, 0.5)",
            "text": "#fecaca",
            "pill_bg": "rgba(220, 38, 38, 0.22)",
        },
    }
    return palettes.get(label, palettes["Elevated"])


def get_interpretation_palette(text: str) -> dict:
    """
    Map overall interpretation text to a broad palette.
    """
    text_lower = text.lower()

    if "low" in text_lower:
        return get_risk_palette("Low")
    if "elevated" in text_lower:
        return get_risk_palette("Elevated")
    if "moderate" in text_lower:
        return get_risk_palette("Moderate")
    if "high" in text_lower:
        return get_risk_palette("High")
    return get_risk_palette("Elevated")


# =========================
# Global Styling
# =========================

def inject_global_styles():
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1250px;
            }

            .rr-section-title {
                font-size: 1.1rem;
                font-weight: 700;
                color: #f3f4f6;
                margin-bottom: 0.7rem;
                letter-spacing: 0.2px;
            }

            .rr-muted {
                color: #9ca3af;
                font-size: 0.95rem;
            }

            .rr-card {
                border-radius: 20px;
                padding: 1.1rem 1.2rem;
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                box-shadow: 0 10px 30px rgba(0,0,0,0.18);
                min-height: 180px;
            }

            .rr-card-title {
                font-size: 0.95rem;
                font-weight: 600;
                color: #d1d5db;
                margin-bottom: 0.8rem;
            }

            .rr-card-prob {
                font-size: 2.1rem;
                font-weight: 800;
                line-height: 1.1;
                margin-bottom: 0.55rem;
                letter-spacing: -0.02em;
            }

            .rr-pill {
                display: inline-block;
                padding: 0.28rem 0.7rem;
                border-radius: 999px;
                font-size: 0.85rem;
                font-weight: 700;
                margin-bottom: 0.9rem;
            }

            .rr-subtext {
                color: #cbd5e1;
                font-size: 0.9rem;
                line-height: 1.45;
            }

            .rr-banner {
                border-radius: 20px;
                padding: 1rem 1.2rem;
                margin-top: 0.2rem;
                margin-bottom: 0.4rem;
                box-shadow: 0 10px 28px rgba(0,0,0,0.16);
            }

            .rr-banner-title {
                font-size: 0.92rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.35rem;
            }

            .rr-banner-body {
                font-size: 1.05rem;
                font-weight: 600;
                line-height: 1.45;
            }

            .rr-legend-wrap {
                display: flex;
                gap: 0.6rem;
                flex-wrap: wrap;
                margin-top: 0.35rem;
            }

            .rr-legend-item {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.35rem 0.7rem;
                border-radius: 999px;
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.08);
                color: #d1d5db;
                font-size: 0.84rem;
            }

            .rr-dot {
                width: 10px;
                height: 10px;
                border-radius: 50%;
                display: inline-block;
            }

            .rr-mini-card {
                border-radius: 16px;
                padding: 0.85rem 1rem;
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.08);
            }

            .rr-mini-label {
                font-size: 0.82rem;
                color: #9ca3af;
                margin-bottom: 0.3rem;
            }

            .rr-mini-value {
                font-size: 1.1rem;
                font-weight: 700;
                color: #f3f4f6;
            }

            div[data-testid="stDataFrame"] {
                border-radius: 16px;
                overflow: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


# =========================
# Reusable UI Blocks
# =========================

def render_page_header(title: str, subtitle: str | None = None):
    st.markdown(
        f"""
        <div style="margin-bottom: 1.25rem;">
            <div style="font-size: 2rem; font-weight: 800; color: #f9fafb; letter-spacing: -0.02em;">
                {title}
            </div>
            {f'<div class="rr-muted" style="margin-top: 0.35rem;">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_risk_card(title: str, prob: float, label: str, threshold: float):
    palette = get_risk_palette(label)

    st.markdown(
        f"""
        <div class="rr-card"
             style="
                background: linear-gradient(180deg, rgba(17,24,39,0.92), rgba(17,24,39,0.78));
                border: 1px solid {palette['border']};
             ">
            <div class="rr-card-title">{title}</div>
            <div class="rr-card-prob" style="color: {palette['accent']};">{prob:.1%}</div>
            <div class="rr-pill"
                 style="
                    background: {palette['pill_bg']};
                    color: {palette['text']};
                    border: 1px solid {palette['border']};
                 ">
                {label} Risk
            </div>
            <div class="rr-subtext">
                Threshold signal: <strong style="color: #f9fafb;">{threshold:.2f}</strong><br>
                This reflects the model’s calibrated alert cutoff for this time window.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_interpretation_banner(text: str):
    palette = get_interpretation_palette(text)

    st.markdown(
        f"""
        <div class="rr-banner"
             style="
                background: linear-gradient(135deg, {palette['bg']}, rgba(17,24,39,0.88));
                border: 1px solid {palette['border']};
             ">
            <div class="rr-banner-title" style="color: {palette['text']};">
                Overall Interpretation
            </div>
            <div class="rr-banner-body" style="color: #f9fafb;">
                {text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_reference_row(reference_date: str):
    st.markdown(
        f"""
        <div class="rr-mini-card">
            <div class="rr-mini-label">Reference Monthly Row Used</div>
            <div class="rr-mini-value">{reference_date}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_signal_legend():
    items = [
        ("Low", "#22c55e"),
        ("Elevated", "#f59e0b"),
        ("Moderate", "#fb923c"),
        ("High", "#ef4444"),
        ("Severe", "#dc2626"),
    ]

    html_items = "".join(
        f'<div class="rr-legend-item"><span class="rr-dot" style="background:{color};"></span><span>{label}</span></div>'
        for label, color in items
    )

    legend_html = f"""
<div class="rr-section-title">Risk Legend</div>
<div class="rr-legend-wrap">{html_items}</div>
"""

    st.markdown(legend_html, unsafe_allow_html=True)


def render_section_title(title: str, subtitle: str | None = None):
    st.markdown(
        f"""
        <div style="margin-top: 0.2rem; margin-bottom: 0.85rem;">
            <div class="rr-section-title">{title}</div>
            {f'<div class="rr-muted">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metric_strip(items: list[tuple[str, str]]):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="rr-mini-card">
                    <div class="rr-mini-label">{label}</div>
                    <div class="rr-mini-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )