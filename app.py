import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root so we can import local modules when running via `streamlit run app.py`
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.predict import predict_mental_health

# -----------------------------------------------------------------------------
# Page configuration + global styles
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PulseMind — Guided Digital Well-being",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-base: #05060f;
        --bg-panel: rgba(11, 13, 24, 0.85);
        --bg-card: rgba(18, 22, 39, 0.95);
        --border-soft: rgba(148, 163, 184, 0.24);
        --text-primary: #f8fafc;
        --text-secondary: #cbd5f5;
        --accent: #8b5cf6;
        --accent-soft: #ec4899;
        --accent-glow: #06b6d4;
        --accent-warm: #f97316;
        --shadow-xl: 0 25px 65px rgba(2, 6, 23, 0.55);
    }
    
    * {
        font-family: 'Inter', 'Space Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(120% 90% at 0% 0%, rgba(139, 92, 246, 0.35), transparent 55%),
            radial-gradient(80% 80% at 100% 0%, rgba(14, 165, 233, 0.4), transparent 50%),
            linear-gradient(135deg, #020408, #05060f 60%, #080a13 100%);
        color: var(--text-primary);
    }
    
    [data-testid="block-container"] {
        padding: 3.5rem 3rem 5rem;
        max-width: 1200px;
    }

    #MainMenu, header, footer, [data-testid="stToolbar"] {
        visibility: hidden;
        height: 0;
    }

    .hero {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(139, 92, 246, 0.08));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 32px;
        padding: 2.6rem 3rem;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-xl);
        position: relative;
        overflow: hidden;
    }
    
    .hero::after {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at 80% -10%, rgba(236, 72, 153, 0.3), transparent 45%);
        pointer-events: none;
    }
    
    .hero .chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        background: rgba(99, 102, 241, 0.18);
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    .hero h1 {
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        font-size: 3.1rem;
        letter-spacing: -0.02em;
        margin: 1rem 0 0.35rem;
    }
    
    .hero p {
        max-width: 720px;
        color: var(--text-secondary);
        font-size: 1.15rem;
    }

    .stepper {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
        margin-bottom: 1.25rem;
    }

    .step {
        background: var(--bg-panel);
        border-radius: 20px;
        padding: 1.2rem 1.4rem;
        border: 1px solid var(--border-soft);
        display: flex;
        align-items: center;
        gap: 0.9rem;
        box-shadow: var(--shadow-xl);
        position: relative;
    }

    .step-index {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        background: rgba(148, 163, 184, 0.15);
        color: var(--text-primary);
    }

    .step.done .step-index {
        background: linear-gradient(135deg, var(--accent), var(--accent-glow));
        color: #05060f;
    }

    .step.current {
        border-color: rgba(99, 102, 241, 0.65);
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(15, 23, 42, 0.8));
    }

    .step p {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.75rem;
        margin: 0;
        color: var(--text-secondary);
    }

    .step h4 {
        margin: 0.15rem 0 0;
        font-size: 1.05rem;
        font-weight: 600;
    }

    form[data-testid="stForm"] {
        background: var(--bg-card);
        padding: 2rem 2.25rem;
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 28px;
        box-shadow: var(--shadow-xl);
        margin-bottom: 1.5rem;
    }

    form[data-testid="stForm"] label {
        font-weight: 600;
        color: var(--text-primary);
    }
    
    form[data-testid="stForm"] .stSlider,
    form[data-testid="stForm"] .stNumberInput,
    form[data-testid="stForm"] .stRadio {
        margin-top: 1.2rem;
    }

    form[data-testid="stForm"] div[data-baseweb="slider"] {
        background: rgba(148, 163, 184, 0.25);
        border-radius: 999px;
        height: 6px;
    }

    form[data-testid="stForm"] div[data-baseweb="slider"] > div {
        background: linear-gradient(90deg, var(--accent), var(--accent-soft));
    }

    form[data-testid="stForm"] div[data-baseweb="thumb"] {
        background: #fff;
        border: 3px solid var(--accent);
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.35);
    }

    .survey-hint {
        color: var(--text-secondary);
        font-size: 0.95rem;
        margin-top: -0.35rem;
    }

    .section-title {
        font-size: 1.2rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--text-secondary);
        margin: 3rem 0 0.75rem;
    }

    .input-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.75rem;
        margin-top: 1rem;
    }

    .input-chip {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 1rem 1.25rem;
    }

    .input-chip span {
        display: block;
        font-size: 0.8rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-secondary);
    }

    .input-chip strong {
        font-size: 1.2rem;
        display: block;
        margin-top: 0.3rem;
    }

    .risk-pill {
        border-radius: 999px;
        padding: 0.85rem 1.6rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        display: inline-flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 0.25rem;
        margin-bottom: 1rem;
    }

    .risk-pill span {
        font-size: 0.75rem;
        opacity: 0.8;
    }

    .risk-pill.safe {
        background: linear-gradient(120deg, #14b8a6, #22d3ee);
        color: #02121f;
    }

    .risk-pill.caution {
        background: linear-gradient(120deg, #facc15, #fb923c);
        color: #1f1302;
    }

    .risk-pill.danger {
        background: linear-gradient(120deg, #f87171, #ef4444);
        color: #28060b;
    }

    .score-card {
        background: linear-gradient(160deg, rgba(23, 23, 55, 0.95), rgba(6, 9, 24, 0.95));
        border-radius: 28px;
        padding: 2rem 2.4rem;
        border: 1px solid rgba(120, 119, 198, 0.35);
        box-shadow: var(--shadow-xl);
    }

    .score-card p.eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.35em;
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-bottom: 0.4rem;
    }

    .score-card h1 {
        font-size: 4rem;
        margin: 0.4rem 0;
        font-weight: 700;
        background: linear-gradient(120deg, #8b5cf6, #ec4899, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .score-card .baseline {
        margin-top: 0.4rem;
        font-size: 0.9rem;
        color: var(--text-secondary);
    }

    .recommendation-list {
        list-style: none;
        padding: 0;
        display: grid;
        gap: 0.85rem;
        margin: 1rem 0 0;
    }

    .recommendation-list li {
        background: rgba(20, 24, 45, 0.85);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 18px;
        padding: 1rem 1.25rem;
        color: var(--text-secondary);
    }

    .recommendation-list strong {
        color: var(--text-primary);
    }

    .stButton>button,
    .pill-button {
        border-radius: 999px;
        border: none;
        padding: 0.85rem 1.5rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        background: linear-gradient(120deg, var(--accent), var(--accent-soft));
        color: #05060f;
        box-shadow: 0 20px 45px rgba(139, 92, 246, 0.35);
    }

    .stButton>button:hover {
        transform: translateY(-1px);
    }
    
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: var(--text-secondary);
        font-size: 0.9rem;
    }

    .dataframe {
        background: rgba(12, 16, 32, 0.95);
        border-radius: 20px;
    }
    
    @media (max-width: 768px) {
        [data-testid="block-container"] {
            padding: 2rem 1.1rem 3rem;
        }

        .hero h1 {
            font-size: 2.2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Survey + prediction helpers
# -----------------------------------------------------------------------------
SURVEY_STEPS = [
    {"title": "Digital Tempo", "subtitle": "Screen exposure per day"},
    {"title": "Social Universe", "subtitle": "How wide are your feeds?"},
    {"title": "Rest & Resilience", "subtitle": "Recharge and manage stress"},
]

FEATURE_LABELS = {
    "screen_time_hours": "Screen Time",
    "social_media_platforms_used": "Social Platforms",
    "hours_on_TikTok": "TikTok Time",
    "sleep_hours": "Sleep",
    "stress_level": "Stress",
}

DEFAULT_INPUTS = {
    "screen_time_hours": 7.0,
    "social_media_platforms_used": 3,
    "hours_on_TikTok": 2.0,
    "sleep_hours": 7.0,
    "stress_level": 5,
}

RISK_CLASS_MAP = {
    "Low": "safe",
    "Medium": "caution",
    "High": "danger",
}

for key, value in DEFAULT_INPUTS.items():
    st.session_state.setdefault(key, value)

st.session_state.setdefault("survey_step", 0)
st.session_state.setdefault("prediction_result", None)
st.session_state.setdefault("last_error", None)


def clamp_step():
    total = len(SURVEY_STEPS)
    st.session_state.survey_step = max(0, min(st.session_state.survey_step, total - 1))


def reset_prediction():
    st.session_state.prediction_result = None
    st.session_state.last_error = None


def go_to_step(step_index: int):
    st.session_state.survey_step = max(0, min(step_index, len(SURVEY_STEPS) - 1))


def restart_survey():
    for key, value in DEFAULT_INPUTS.items():
        st.session_state[key] = value
    st.session_state.survey_step = 0
    reset_prediction()


def run_prediction():
    user_features = {key: float(st.session_state[key]) for key in DEFAULT_INPUTS}
    try:
        with st.spinner("Synthesizing your digital pulse..."):
                result = predict_mental_health(user_features)
                st.session_state.prediction_result = result
        st.session_state.last_error = None
    except Exception as exc:
        st.session_state.prediction_result = None
        st.session_state.last_error = str(exc)


def render_stepper(current_step: int):
    tiles = []
    for idx, step in enumerate(SURVEY_STEPS):
        state = "current" if idx == current_step else "done" if idx < current_step else "todo"
        tiles.append(
            f"""
            <div class="step {state}">
                <div class="step-index">{idx + 1}</div>
                <div>
                    <p>{step["subtitle"]}</p>
                    <h4>{step["title"]}</h4>
                </div>
                </div>
            """
        )
    st.markdown(f'<div class="stepper">{"".join(tiles)}</div>', unsafe_allow_html=True)


def recommendations_from_contrib(contributions):
    recs = []
    for contrib in contributions[:3]:
        feature = contrib["feature"]
        value = contrib["value"]
        if feature == "screen_time_hours" and value > 8:
            recs.append("📱 <strong>Trim screen bursts:</strong> set evening cutoffs or batch notifications.")
        elif feature == "hours_on_TikTok" and value > 3:
            recs.append("🎧 <strong>Micro detox blocks:</strong> swap short-form loops with mindful breaks.")
        elif feature == "sleep_hours" and value < 7:
            recs.append("😴 <strong>Protect sleep:</strong> aim for 7-9 hours with a fixed wind-down routine.")
        elif feature == "stress_level" and value > 6:
            recs.append("🧘 <strong>Diffuse stress:</strong> add breathwork, stretching, or quick journaling sessions.")
        elif feature == "social_media_platforms_used" and value > 5:
            recs.append("🌐 <strong>Curate feeds:</strong> mute noisy platforms and focus on meaningful spaces.")
    if not recs:
        recs.append("✨ Digital habits look balanced—keep honoring the healthy boundaries you have in place.")
    return recs


def render_feature_contributions(result):
    contributions = result["contributions"]
    contrib_df = pd.DataFrame(contributions)
    contrib_df["feature_display"] = contrib_df["feature"].map(FEATURE_LABELS).fillna(contrib_df["feature"])

    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        fig_bar = go.Figure()
        positive = contrib_df[contrib_df["normalized_contribution"] > 0]
        negative = contrib_df[contrib_df["normalized_contribution"] < 0]

        if not positive.empty:
            fig_bar.add_trace(
                go.Bar(
                    y=positive["feature_display"],
                    x=positive["normalized_contribution"],
                    orientation="h",
                    name="Boosts score",
                    marker_color="#22d3ee",
                    text=[f"+{x:.3f}" for x in positive["normalized_contribution"]],
                    textposition="auto",
                )
            )

        if not negative.empty:
            fig_bar.add_trace(
                go.Bar(
                    y=negative["feature_display"],
                    x=negative["normalized_contribution"],
                    orientation="h",
                    name="Pulls score down",
                    marker_color="#f87171",
                    text=[f"{x:.3f}" for x in negative["normalized_contribution"]],
                    textposition="auto",
                )
            )
        
        fig_bar.update_layout(
            title="Feature contributions to mental health score",
            xaxis_title="Normalized contribution",
            yaxis_title="",
            barmode="relative",
            showlegend=True,
            height=420,
            margin=dict(l=0, r=0, t=70, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc"),
            xaxis=dict(gridcolor="rgba(148,163,184,0.3)"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.2)"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with viz_col2:
        fig_pie = px.pie(
            contrib_df,
            values="abs_contribution",
            names="feature_display",
            title="Relative feature importance",
            color_discrete_sequence=["#8b5cf6", "#06b6d4", "#ec4899", "#f97316", "#14b8a6"],
        )
        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Contribution: %{value:.3f}<extra></extra>",
        )
        fig_pie.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=70, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc"),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("#### Detailed breakdown")
    display_df = contrib_df[["feature_display", "value", "normalized_contribution", "direction"]].copy()
    display_df.columns = ["Feature", "Your value", "Contribution", "Impact"]
    display_df["Impact"] = display_df["Impact"].str.replace("_", " ").str.title()
    display_df["Contribution"] = display_df["Contribution"].map(lambda x: f"{x:.4f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_results(result):
    st.markdown('<div class="section-title">Insight Report</div>', unsafe_allow_html=True)

    user_inputs = {
        "Screen Time": f"{st.session_state.screen_time_hours:.1f} hrs/day",
        "Social Platforms": f"{st.session_state.social_media_platforms_used}",
        "TikTok Time": f"{st.session_state.hours_on_TikTok:.1f} hrs/day",
        "Sleep": f"{st.session_state.sleep_hours:.1f} hrs/night",
        "Stress": f"{st.session_state.stress_level}/10",
    }

    summary_col, result_col = st.columns([1.25, 0.75])

    with summary_col:
        st.markdown("#### Input snapshot")
        chips = "".join(
            [f'<div class="input-chip"><span>{label}</span><strong>{value}</strong></div>' for label, value in user_inputs.items()]
        )
        st.markdown(f'<div class="input-grid">{chips}</div>', unsafe_allow_html=True)

    with result_col:
        risk_label = result["risk_category"]
        risk_class = RISK_CLASS_MAP.get(risk_label, "caution")
        st.markdown(
            f'<div class="risk-pill {risk_class}"><span>Risk category</span><strong>{risk_label} risk</strong></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="score-card">
                <p class="eyebrow">Mental health score</p>
                <h1>{result["predicted_score"]:.2f}</h1>
                <p>Higher values indicate stronger mental well-being.</p>
                <div class="baseline">Model baseline: {result["base_value"]:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Risk levels adapt dynamically: Low (calm and resilient), Medium (watch for stress creep), High (prioritize recovery rituals)."
        )

    st.markdown("#### Feature contributions & importance")
    render_feature_contributions(result)

    st.markdown("#### Personalized recommendations")
    recs = recommendations_from_contrib(result["contributions"])
    rec_html = "".join([f"<li>{rec}</li>" for rec in recs])
    st.markdown(f'<ul class="recommendation-list">{rec_html}</ul>', unsafe_allow_html=True)

    st.button("Start another scenario", use_container_width=True, on_click=restart_survey)


# -----------------------------------------------------------------------------
# UI flow
# -----------------------------------------------------------------------------
clamp_step()
current_step = st.session_state.survey_step

st.markdown(
    """
    <div class="hero">
        <span class="chip">PulseMind Survey</span>
        <h1>Modern, calm, and data-forward.</h1>
        <p>Move through curated micro-questions, then let our model illuminate your risk level,
        score, and the exact habits that shape it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

render_stepper(current_step)


def render_digital_step():
    next_clicked = False
    with st.form("step-digital"):
        st.markdown("#### Q1 · Daily digital tempo")
        st.caption("Calibrate how much glow time you average each day.")
        st.slider(
            "Screen time (hours / day)",
            min_value=0.0,
            max_value=16.0,
            step=0.1,
            key="screen_time_hours",
            format="%.1f",
        )
        st.caption("Use the slider to capture your average exposure to screens each day.")

        col_prev, col_next = st.columns(2)
        col_prev.form_submit_button("◀ Back", disabled=True)
        next_clicked = col_next.form_submit_button("Next · Social pulse")

    if next_clicked:
        go_to_step(1)
        reset_prediction()


def render_social_step():
    prev_clicked = False
    next_clicked = False
    with st.form("step-social"):
        st.markdown("#### Q2 · Social pulse")
        st.caption("How many feeds compete for your attention?")
        st.slider(
            "Active social platforms",
            min_value=0,
            max_value=10,
            step=1,
            key="social_media_platforms_used",
        )
        st.slider(
            "Hours on TikTok (per day)",
            min_value=0.0,
            max_value=12.0,
            step=0.1,
            key="hours_on_TikTok",
            format="%.1f",
        )

        col_prev, col_next = st.columns(2)
        prev_clicked = col_prev.form_submit_button("◀ Back")
        next_clicked = col_next.form_submit_button("Next · Rest & resilience")

    if prev_clicked:
        go_to_step(0)
        reset_prediction()
    elif next_clicked:
        go_to_step(2)
        reset_prediction()


def render_rest_step():
    prev_clicked = False
    predict_clicked = False
    with st.form("step-rest"):
        st.markdown("#### Q3 · Rest & resilience")
        st.caption("Balance calm sleep with honest stress check-ins.")
        st.slider(
            "Sleep hours (per night)",
            min_value=0.0,
            max_value=12.0,
            step=0.1,
            key="sleep_hours",
            format="%.1f",
        )
        st.slider(
            "Stress level",
            min_value=0,
            max_value=10,
            step=1,
            key="stress_level",
        )
        st.caption("0 = calm and grounded · 10 = overwhelmed")

        col_prev, col_action = st.columns(2)
        prev_clicked = col_prev.form_submit_button("◀ Back")
        predict_clicked = col_action.form_submit_button("Predict risk 🔮")

    if prev_clicked:
        go_to_step(1)
        reset_prediction()
    elif predict_clicked:
        run_prediction()


if current_step == 0:
    render_digital_step()
elif current_step == 1:
    render_social_step()
else:
    render_rest_step()


if st.session_state.last_error:
    st.error(f"Prediction issue: {st.session_state.last_error}")


if st.session_state.prediction_result is not None:
    render_results(st.session_state.prediction_result)


st.markdown(
    """
    <div class="footer">
        PulseMind blends data transparency with soothing design — use these insights to keep your digital habits human.
    </div>
    """,
    unsafe_allow_html=True,
)
