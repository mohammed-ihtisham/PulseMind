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
    
    [data-testid="block-container"],
    div[data-testid="block-container"],
    .stMainBlockContainer,
    div.stMainBlockContainer,
    div.block-container {
        padding: 1.5rem 3rem 2rem;
        max-width: 1200px;
    }
    
    .results-mode [data-testid="block-container"],
    .results-mode .stMainBlockContainer {
        padding: 3.5rem 3rem 5rem;
    }
    
    /* Remove Streamlit's default top spacing */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    .stApp > header {
        display: none !important;
    }

    #MainMenu, header, footer, [data-testid="stToolbar"] {
        visibility: hidden;
        height: 0;
    }

    .hero {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(139, 92, 246, 0.08));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 1.4rem 2rem;
        margin-bottom: 1.2rem;
        box-shadow: var(--shadow-xl);
        position: relative;
        overflow: hidden;
    }
    
    .results-mode .hero {
        padding: 2.6rem 3rem;
        margin-bottom: 2rem;
        border-radius: 32px;
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
        font-size: 2.2rem;
        letter-spacing: -0.02em;
        margin: 0.5rem 0 0.25rem;
    }
    
    .results-mode .hero h1 {
        font-size: 3.1rem;
        margin: 1rem 0 0.35rem;
    }
    
    .hero p {
        max-width: 720px;
        color: var(--text-secondary);
        font-size: 0.95rem;
        margin-top: 0.25rem;
    }
    
    .results-mode .hero p {
        font-size: 1.15rem;
        margin-top: 0;
    }

    .stepper {
        position: relative;
        margin-bottom: 1.2rem;
        padding: 0.5rem 0.4rem 0;
    }
    
    .results-mode .stepper {
        margin-bottom: 2.4rem;
    }

    .stepper-track {
        position: absolute;
        top: 28px;
        left: 40px;
        right: 40px;
        height: 3px;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.2);
        overflow: hidden;
    }

    .stepper-progress {
        height: 100%;
        background: linear-gradient(90deg, rgba(139, 92, 246, 0.6), rgba(6, 182, 212, 0.6));
        transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 999px;
    }
    
    .stepper-nodes {
        position: relative;
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0;
        z-index: 1;
        align-items: flex-start;
    }
    
    .step {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        gap: 0.5rem;
        padding: 0 0.5rem;
    }
    
    .step-index {
        width: 40px;
        height: 40px;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 1rem;
        background: rgba(30, 41, 59, 0.8);
        border: 2px solid rgba(148, 163, 184, 0.4);
        color: rgba(203, 213, 225, 0.9);
        position: relative;
        z-index: 2;
        transition: all 0.3s ease;
    }
    
    .step.done .step-index {
        background: linear-gradient(135deg, #8b5cf6, #06b6d4);
        color: #ffffff;
        border-color: transparent;
        box-shadow: 0 0 0 0 rgba(139, 92, 246, 0);
    }
    
    .step.current .step-index {
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        border: 2px solid rgba(139, 92, 246, 0.8);
        color: #ffffff;
        box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.3), 0 0 20px rgba(139, 92, 246, 0.4);
        animation: pulse-glow 2s ease-in-out infinite;
    }

    @keyframes pulse-glow {
        0%, 100% {
            box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.3), 0 0 20px rgba(139, 92, 246, 0.4);
        }
        50% {
            box-shadow: 0 0 0 6px rgba(139, 92, 246, 0.4), 0 0 30px rgba(139, 92, 246, 0.6);
        }
    }

    .step.todo .step-index {
        background: rgba(30, 41, 59, 0.6);
        border-color: rgba(148, 163, 184, 0.3);
        color: rgba(203, 213, 225, 0.7);
    }
    
    .step p {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.7rem;
        margin: 0.5rem 0 0;
        color: rgba(203, 213, 225, 0.85);
        font-weight: 500;
        line-height: 1.3;
    }
    
    .step h4 {
        margin: 0.25rem 0 0;
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.01em;
    }

    form[data-testid="stForm"] {
        background: linear-gradient(135deg, rgba(18, 22, 39, 0.98), rgba(15, 23, 42, 0.95));
        padding: 2rem 2.2rem;
        border: 1px solid rgba(139, 92, 246, 0.15);
        border-radius: 28px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.05) inset;
        margin-bottom: 0.8rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    form[data-testid="stForm"]:hover {
        border-color: rgba(139, 92, 246, 0.25);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.08) inset;
    }
    
    .results-mode form[data-testid="stForm"] {
        padding: 2rem 2.25rem;
        border-radius: 28px;
        margin-bottom: 1.5rem;
    }

    /* Modern question title styling */
    form[data-testid="stForm"] h3 {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        margin-bottom: 0.5rem !important;
        background: linear-gradient(135deg, #ffffff, rgba(203, 213, 225, 0.9));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    form[data-testid="stForm"] .stCaption {
        font-size: 0.95rem !important;
        color: rgba(203, 213, 225, 0.85) !important;
        margin-top: -0.2rem !important;
        margin-bottom: 1.5rem !important;
        line-height: 1.5;
        font-weight: 400;
    }

    form[data-testid="stForm"] label {
        font-weight: 600;
        font-size: 0.95rem;
        color: rgba(248, 250, 252, 0.95);
        letter-spacing: 0.01em;
        margin-bottom: 0.75rem;
        display: block;
    }
    
    form[data-testid="stForm"] .stSlider,
    form[data-testid="stForm"] .stNumberInput,
    form[data-testid="stForm"] .stRadio {
        margin-top: 1.2rem;
    }
    
    .results-mode form[data-testid="stForm"] .stSlider,
    .results-mode form[data-testid="stForm"] .stNumberInput,
    .results-mode form[data-testid="stForm"] .stRadio {
        margin-top: 1.2rem;
    }

    /* Enhanced slider design */
    form[data-testid="stForm"] div[data-baseweb="slider"] {
        background: rgba(148, 163, 184, 0.15);
        border-radius: 999px;
        height: 8px;
        position: relative;
        overflow: visible;
    }

    form[data-testid="stForm"] div[data-baseweb="slider"] > div {
        background: linear-gradient(90deg, #8b5cf6, #ec4899, #06b6d4);
        border-radius: 999px;
        height: 100%;
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.4);
    }

    form[data-testid="stForm"] div[data-baseweb="thumb"] {
        background: #ffffff;
        border: 3px solid #8b5cf6;
        box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.2), 0 4px 12px rgba(139, 92, 246, 0.4);
        width: 24px !important;
        height: 24px !important;
        transition: all 0.2s ease;
    }
    
    form[data-testid="stForm"] div[data-baseweb="thumb"]:hover {
        transform: scale(1.15);
        box-shadow: 0 0 0 6px rgba(139, 92, 246, 0.3), 0 6px 16px rgba(139, 92, 246, 0.5);
    }
    
    /* Slider value display */
    form[data-testid="stForm"] div[data-baseweb="slider"] + div {
        color: rgba(248, 250, 252, 0.9);
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Modern button styling - all buttons use same grey style */
    form[data-testid="stForm"] button {
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.2s ease !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        background: rgba(148, 163, 184, 0.1) !important;
        color: rgba(203, 213, 225, 0.9) !important;
    }
    
    form[data-testid="stForm"] button[disabled] {
        background: rgba(148, 163, 184, 0.05) !important;
        color: rgba(203, 213, 225, 0.4) !important;
        border-color: rgba(148, 163, 184, 0.2) !important;
        cursor: not-allowed !important;
    }
    
    form[data-testid="stForm"] button:not([disabled]):hover {
        background: rgba(148, 163, 184, 0.15) !important;
        border-color: rgba(148, 163, 184, 0.4) !important;
        transform: translateY(-1px);
    }
    
    form[data-testid="stForm"] button:not([disabled]):active {
        transform: translateY(0);
    }
    
    /* Right-align Next button - target third column (with spacer) */
    form[data-testid="stForm"] div[data-testid="column"]:nth-child(3),
    form[data-testid="stForm"] div[data-testid="column"]:last-child {
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    
    form[data-testid="stForm"] div[data-testid="column"]:nth-child(3) > div,
    form[data-testid="stForm"] div[data-testid="column"]:last-child > div,
    form[data-testid="stForm"] div[data-testid="column"]:nth-child(3) .stVerticalBlock,
    form[data-testid="stForm"] div[data-testid="column"]:last-child .stVerticalBlock {
        width: 100% !important;
        display: flex !important;
        justify-content: flex-end !important;
        padding-right: 0 !important;
    }
    
    form[data-testid="stForm"] div[data-testid="column"]:nth-child(3) .stButton,
    form[data-testid="stForm"] div[data-testid="column"]:nth-child(3) button,
    form[data-testid="stForm"] div[data-testid="column"]:last-child .stButton,
    form[data-testid="stForm"] div[data-testid="column"]:last-child button {
        margin-left: auto !important;
        margin-right: 0 !important;
        float: right !important;
    }
    
    /* Modern checkbox styling */
    form[data-testid="stForm"] .stCheckbox {
        margin-top: 0.5rem;
    }
    
    form[data-testid="stForm"] .stCheckbox label {
        font-size: 0.9rem;
        color: rgba(248, 250, 252, 0.9);
        font-weight: 500;
    }
    
    form[data-testid="stForm"] .stCheckbox input[type="checkbox"] {
        accent-color: #8b5cf6;
        width: 18px;
        height: 18px;
        cursor: pointer;
    }
    
    /* Better spacing for form sections */
    form[data-testid="stForm"] > div {
        margin-bottom: 1rem;
    }
    
    form[data-testid="stForm"] > div:last-child {
        margin-bottom: 0;
    }

    .survey-hint {
        color: var(--text-secondary);
        font-size: 0.95rem;
        margin-top: -0.35rem;
    }

    .section-title {
        font-size: 1.4rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--text-primary);
        margin: 3.5rem 0 1.5rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 1rem;
        background: linear-gradient(135deg, #ffffff, rgba(203, 213, 225, 0.9));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .section-title::before {
        content: "";
        width: 4px;
        height: 2rem;
        background: linear-gradient(180deg, #8b5cf6, #06b6d4);
        border-radius: 999px;
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.5);
    }

    .input-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.85rem;
        margin-top: 1rem;
    }

    .input-chip {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 16px;
        padding: 1.15rem 1.75rem;
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.05) inset;
        display: flex;
        align-items: center;
        justify-content: space-between;
        width: 100%;
        flex: 1;
    }

    .input-chip::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.6), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .input-chip:hover {
        transform: translateY(-2px);
        border-color: rgba(139, 92, 246, 0.4);
        box-shadow: 0 8px 30px rgba(139, 92, 246, 0.25), 0 0 0 1px rgba(255, 255, 255, 0.1) inset;
    }

    .input-chip:hover::before {
        opacity: 1;
    }

    .input-chip span {
        font-size: 0.75rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: rgba(203, 213, 225, 0.65);
        font-weight: 700;
        font-family: 'Space Grotesk', 'Inter', sans-serif;
    }

    .input-chip strong {
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 0%, rgba(203, 213, 225, 0.95) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Space Grotesk', 'Inter', sans-serif;
    }

    .risk-pill {
        border-radius: 999px;
        padding: 1rem 1.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 0.25rem;
        margin-bottom: 1.5rem;
        width: 100%;
    }

    .risk-pill span {
        font-size: 0.75rem;
        opacity: 0.8;
    }

    .risk-pill.healthy {
        background: linear-gradient(120deg, #22c55e, #14b8a6);
        color: #02121f;
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

    .risk-pill.critical {
        background: linear-gradient(120deg, #dc2626, #991b1b);
        color: #ffffff;
    }

    .score-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.98), rgba(15, 23, 42, 0.98));
        backdrop-filter: blur(30px);
        border-radius: 32px;
        padding: 2.5rem 3rem;
        border: 1px solid rgba(139, 92, 246, 0.3);
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.5),
            0 0 0 1px rgba(255, 255, 255, 0.05) inset,
            0 0 80px rgba(139, 92, 246, 0.15);
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .score-card::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.15), transparent 70%);
        animation: rotate-glow 20s linear infinite;
        pointer-events: none;
    }

    @keyframes rotate-glow {
        0% {
            transform: rotate(0deg);
        }
        100% {
            transform: rotate(360deg);
        }
    }

    .score-card:hover {
        transform: translateY(-4px);
        border-color: rgba(139, 92, 246, 0.5);
        box-shadow: 
            0 25px 80px rgba(0, 0, 0, 0.6),
            0 0 0 1px rgba(255, 255, 255, 0.1) inset,
            0 0 100px rgba(139, 92, 246, 0.25);
    }

    .score-card > * {
        position: relative;
        z-index: 1;
    }

    .score-card p.eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.4em;
        font-size: 0.7rem;
        color: rgba(203, 213, 225, 0.8);
        margin-bottom: 0.6rem;
        font-weight: 600;
    }

    .score-card h1 {
        font-size: 4.5rem;
        margin: 0.5rem 0;
        font-weight: 800;
        line-height: 1;
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        filter: drop-shadow(0 0 20px rgba(139, 92, 246, 0.5));
        animation: score-glow 3s ease-in-out infinite;
    }

    @keyframes score-glow {
        0%, 100% {
            filter: drop-shadow(0 0 20px rgba(139, 92, 246, 0.5));
        }
        50% {
            filter: drop-shadow(0 0 30px rgba(139, 92, 246, 0.8));
        }
    }

    .score-card p {
        margin-top: 0.8rem;
        font-size: 0.95rem;
        color: rgba(203, 213, 225, 0.9);
        line-height: 1.6;
    }

    .score-spectrum {
        margin-top: 1.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(139, 92, 246, 0.2);
    }

    .score-spectrum-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: rgba(203, 213, 225, 0.8);
        margin-bottom: 0.75rem;
        font-weight: 600;
    }

    .score-spectrum-bar {
        width: 100%;
        height: 12px;
        border-radius: 999px;
        background: linear-gradient(90deg, #dc2626 0%, #ef4444 20%, #f87171 40%, #facc15 60%, #fb923c 80%, #14b8a6 90%, #22c55e 100%);
        position: relative;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) inset;
    }

    .score-spectrum-marker {
        position: absolute;
        top: 50%;
        transform: translate(-50%, -50%);
        width: 20px;
        height: 20px;
        background: #ffffff;
        border: 3px solid currentColor;
        border-radius: 50%;
        box-shadow: 0 0 0 4px rgba(0, 0, 0, 0.5), 0 4px 12px rgba(0, 0, 0, 0.4);
        z-index: 10;
    }

    .score-spectrum-labels {
        display: flex;
        justify-content: space-between;
        margin-top: 0.5rem;
        font-size: 0.7rem;
        color: rgba(203, 213, 225, 0.6);
    }

    .score-spectrum-labels span {
        font-weight: 500;
    }

    .score-boundaries {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(139, 92, 246, 0.15);
        font-size: 0.8rem;
        color: rgba(203, 213, 225, 0.7);
        line-height: 1.6;
    }

    .score-boundaries strong {
        color: rgba(203, 213, 225, 0.9);
        font-weight: 600;
    }

    .risk-explanation {
        margin-top: 1rem;
        padding: 0.85rem 1rem;
        background: rgba(139, 92, 246, 0.1);
        border-left: 3px solid rgba(139, 92, 246, 0.5);
        border-radius: 8px;
        font-size: 0.85rem;
        color: rgba(203, 213, 225, 0.9);
        line-height: 1.5;
    }

    .disclaimer {
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        padding: 1.25rem 1.5rem;
        background: rgba(251, 191, 36, 0.1);
        border: 1px solid rgba(251, 191, 36, 0.3);
        border-left: 4px solid rgba(251, 191, 36, 0.6);
        border-radius: 12px;
        font-size: 0.85rem;
        color: rgba(203, 213, 225, 0.95);
        line-height: 1.6;
    }

    .disclaimer strong {
        color: rgba(251, 191, 36, 0.9);
        font-weight: 600;
    }

    /* Equal height columns - ensure they display side by side */
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        gap: 1.5rem !important;
        align-items: stretch !important;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: stretch !important;
        flex: 1 1 auto !important;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child {
        flex: 1.25 1 auto !important;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child {
        flex: 0.75 1 auto !important;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="column"] > div {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }

    /* Input grid is now a 2x2 grid, no need for min-height */

    .personalized-suggestions {
        margin-top: 1rem;
    }

    .suggestion-item {
        background: linear-gradient(135deg, rgba(18, 22, 39, 0.98), rgba(15, 23, 42, 0.95));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-left: 4px solid rgba(139, 92, 246, 0.6);
        border-radius: 16px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.7rem;
        position: relative;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.05) inset;
        display: flex;
        flex-direction: column;
        min-height: 100px;
    }

    .suggestion-item:hover {
        transform: translateY(-2px);
        border-color: rgba(139, 92, 246, 0.4);
        border-left-color: rgba(139, 92, 246, 0.8);
        box-shadow: 0 8px 30px rgba(139, 92, 246, 0.25), 0 0 0 1px rgba(255, 255, 255, 0.1) inset;
    }

    .suggestion-item::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, rgba(139, 92, 246, 0.8), rgba(6, 182, 212, 0.8));
        border-radius: 4px 0 0 4px;
    }

    .suggestion-text {
        font-size: 0.9rem;
        line-height: 1.6;
        color: rgba(203, 213, 225, 0.95);
        margin: 0;
        flex: 1;
    }

    .suggestion-meta {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 0.75rem;
        font-size: 0.75rem;
        color: rgba(203, 213, 225, 0.6);
        min-height: 1.5rem;
    }

    .suggestion-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.6rem;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.7rem;
        letter-spacing: 0.05em;
    }

    .recommendation-list {
        list-style: none;
        padding: 0;
        display: grid;
        gap: 0.6rem;
        margin: 0.5rem 0 0;
    }

    .recommendation-list li {
        background: rgba(20, 24, 45, 0.85);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 18px;
        padding: 0.75rem 1rem;
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

    /* Prediction loading overlay */
    .prediction-loader-backdrop {
        position: fixed;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        background: radial-gradient(circle at 30% 20%, rgba(99, 102, 241, 0.18), transparent 35%), rgba(5, 6, 15, 0.82);
        backdrop-filter: blur(10px);
        z-index: 9999;
    }
    
    .prediction-loader {
        position: relative;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 1rem;
        padding: 1.75rem 2rem;
        background: rgba(12, 16, 32, 0.9);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 22px;
        box-shadow: 0 25px 60px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(255, 255, 255, 0.05) inset;
        min-width: 280px;
    }
    
    .prediction-loader .ring {
        width: 86px;
        height: 86px;
        border-radius: 50%;
        background: conic-gradient(from 0deg, #8b5cf6, #06b6d4, #ec4899, #8b5cf6);
        animation: loader-spin 1.2s linear infinite;
        position: relative;
        filter: drop-shadow(0 0 16px rgba(139, 92, 246, 0.45));
    }
    
    .prediction-loader .ring::after {
        content: "";
        position: absolute;
        inset: 8px;
        border-radius: 50%;
        background: linear-gradient(135deg, rgba(12, 16, 32, 0.95), rgba(5, 6, 15, 0.95));
        box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.04) inset;
    }
    
    .prediction-loader .orb {
        position: absolute;
        top: -6px;
        left: 50%;
        transform: translateX(-50%);
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: linear-gradient(135deg, #ffffff, #a78bfa);
        box-shadow: 0 0 12px rgba(167, 139, 250, 0.8), 0 0 28px rgba(6, 182, 212, 0.65);
        animation: loader-spin 1.2s linear infinite;
        z-index: 2;
    }
    
    .prediction-loader p {
        margin: 0;
        color: rgba(248, 250, 252, 0.92);
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    
    @keyframes loader-spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    /* Modern section headings */
    h4 {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        margin: 0 0 0.75rem 0 !important;
        color: var(--text-primary) !important;
        background: linear-gradient(135deg, #ffffff, rgba(203, 213, 225, 0.9));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Compact spacing for Personalized Insights/Recommendations sections */
    .personalized-suggestions h4 {
        margin: 1rem 0 0.6rem 0 !important;
    }
    
    .personalized-suggestions h4:first-of-type {
        margin-top: 0 !important;
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
    {"title": "Rest & Resilience", "subtitle": "Sleep and recovery patterns"},
]

FEATURE_LABELS = {
    "screen_time_hours": "Screen Time",
    "social_media_platforms_used": "Social Platforms",
    "hours_on_TikTok": "TikTok Time",
    "sleep_hours": "Sleep",
}

DEFAULT_INPUTS = {
    "screen_time_hours": 2.0,
    "social_media_platforms_used": 3,
    "hours_on_TikTok": 0.0,
    "sleep_hours": 8.0,
}

RISK_CLASS_MAP = {
    "Healthy": "healthy",
    "Low": "safe",
    "Medium": "caution",
    "High": "danger",
    "Critical": "critical",
}

# Unified risk palette + scoring boundaries (keep bar + tag colors in sync)
RISK_COLORS = {
    "Healthy": {"primary": "#15803d", "secondary": "#166534", "text": "#e8f9ec"},
    "Low": {"primary": "#86efac", "secondary": "#22c55e", "text": "#053218"},
    "Medium": {"primary": "#facc15", "secondary": "#f59e0b", "text": "#1f1302"},
    "High": {"primary": "#ef4444", "secondary": "#dc2626", "text": "#ffffff"},
    "Critical": {"primary": "#dc2626", "secondary": "#b91c1c", "text": "#ffffff"},
}

SCORE_MIN = -8.0
SCORE_MAX = 9.0
SCORE_BOUNDARIES = {
    "Critical": 0.0,
    "High": 2.0,
    "Medium": 4.0,
    "Low": 6.0,
}

# Initialize all input values at top level (like the old working version)
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
    # Collect all feature values from session_state
    user_features = {}
    for key in DEFAULT_INPUTS.keys():
        value = st.session_state.get(key, DEFAULT_INPUTS[key])
        # Convert to float, handling both int and float values
        user_features[key] = float(value)
    
    loading_placeholder = st.empty()
    loader_html = """
    <div class="prediction-loader-backdrop">
        <div class="prediction-loader">
            <div class="ring"><span class="orb"></span></div>
            <p>Analyzing your digital pulse...</p>
        </div>
    </div>
    """

    try:
        loading_placeholder.markdown(loader_html, unsafe_allow_html=True)
        result = predict_mental_health(user_features)
        st.session_state.prediction_result = result
        st.session_state.last_error = None
    except Exception as exc:
        st.session_state.prediction_result = None
        st.session_state.last_error = str(exc)
    finally:
        loading_placeholder.empty()


def render_stepper(current_step: int, all_complete: bool = False):
    progress_pct = 0
    total_steps = len(SURVEY_STEPS)
    if total_steps > 1:
        if all_complete:
            progress_pct = 100
        else:
            progress_pct = (current_step / (total_steps - 1)) * 100

    tiles = []
    for idx, step in enumerate(SURVEY_STEPS):
        if all_complete:
            state = "done"
            icon = "✓"
        else:
            state = "current" if idx == current_step else "done" if idx < current_step else "todo"
            icon = "✓" if state == "done" else str(idx + 1)
        tiles.append(
            f'<div class="step {state}">'
            f'<div class="step-index">{icon}</div>'
            f'<p>{step["subtitle"]}</p>'
            f'<h4>{step["title"]}</h4>'
            f'</div>'
        )

    html = (
        '<div class="stepper">'
        '<div class="stepper-track">'
        f'<div class="stepper-progress" style="width:{progress_pct}%"></div>'
        '</div>'
        '<div class="stepper-nodes">'
        f'{"".join(tiles)}'
        '</div>'
        '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


def generate_simple_recommendations(contributions):
    """Generate detailed recommendations for all four factors based on value ranges."""
    recommendations = []
    
    # Calculate total absolute contribution for threshold
    total_abs_contrib = sum(abs(c["abs_contribution"]) for c in contributions)
    if total_abs_contrib == 0:
        return []
    
    # Process each contribution - include features that matter
    for idx, contrib in enumerate(contributions):
        feature = contrib["feature"]
        value = contrib["value"]
        direction = contrib["direction"]
        abs_contrib = contrib["abs_contribution"]
        pct_contrib = (abs_contrib / total_abs_contrib * 100) if total_abs_contrib > 0 else 0
        
        feature_label = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
        
        # Include if:
        # 1. Top contributor (always include)
        # 2. Decreases score and has >= 10% contribution (needs improvement)
        # 3. Increases score with >= 15% contribution (good habit to note)
        
        should_include = False
        if idx == 0:  # Always include top contributor
            should_include = True
        elif direction == "decreases_score" and pct_contrib >= 10:
            should_include = True
        elif direction == "increases_score" and pct_contrib >= 15:
            should_include = True
        
        if not should_include:
            continue
        
        rec = None
        
        # ========== SLEEP HOURS ==========
        if feature == "sleep_hours":
            # Base recommendations on ACTUAL VALUE, not direction
            if value < 5:
                rec = "✨ Focus on sleep: Your sleep is critically low. Aim for 7-9 hours nightly by establishing a consistent bedtime routine and creating a restful environment."
            elif value < 6:
                rec = "✨ Focus on sleep: You're not getting enough sleep. Prioritize increasing to 7-9 hours by going to bed earlier and avoiding screens before bed."
            elif value < 7:
                rec = "✨ Focus on sleep: Your sleep duration is below the recommended 7-9 hours. Try adjusting your schedule to allow for more restful sleep."
            elif value < 8:
                rec = "✨ Focus on sleep: Consider aiming for the optimal 7-9 hour range by maintaining a consistent sleep schedule and improving your sleep quality."
            elif value >= 9:
                rec = "✨ Great job on sleep: You're getting excellent sleep (9+ hours), which is supporting your mental well-being. Keep maintaining this healthy pattern!"
            elif value >= 8:
                rec = "✨ Great job on sleep: Your sleep routine (8+ hours) is supporting your mental health. Continue prioritizing this important habit!"
            elif value >= 7:
                rec = "✨ Great job on sleep: You're getting adequate sleep (7+ hours), which is helping your mental well-being. Maintain this healthy routine!"
            else:
                # Edge case fallback
                rec = "✨ Focus on sleep: Review and adjust your sleep habits to improve your mental well-being score."
        
        # ========== SCREEN TIME ==========
        elif feature == "screen_time_hours":
            # Base recommendations on ACTUAL VALUE, not direction
            if value >= 10:
                rec = "✨ Focus on screen time: Your screen time is excessive (10+ hours). Set strict digital boundaries, use screen time tracking apps, and create device-free zones in your home."
            elif value >= 8:
                rec = "✨ Focus on screen time: Your screen time is very high (8+ hours). Consider implementing evening cutoffs, taking regular breaks, and using blue light filters to reduce impact."
            elif value >= 6:
                rec = "✨ Focus on screen time: Your screen time is elevated. Try scheduling tech-free periods and reducing unnecessary usage to improve your mental well-being."
            elif value >= 4:
                rec = "✨ Focus on screen time: Consider reducing your screen time further. Take intentional breaks and set boundaries around device usage."
            elif value <= 2:
                rec = "✨ Great job on screen time: Your minimal screen time (2 hours or less) is excellent for your mental health. Keep maintaining this balanced approach!"
            elif value <= 4:
                rec = "✨ Great job on screen time: Your low screen time (4 hours or less) is supporting your mental well-being. Continue this healthy habit!"
            elif value <= 6:
                rec = "✨ Maintain screen time: Your moderate screen time is helping your mental health. Keep up the good work with maintaining boundaries!"
            else:
                # Edge case fallback
                rec = "✨ Focus on screen time: Review and adjust your screen time habits to optimize your mental well-being score."
        
        # ========== TIKTOK HOURS ==========
        elif feature == "hours_on_TikTok":
            # Base recommendations on ACTUAL VALUE, not direction
            if value == 0:
                rec = "✨ Great job on TikTok usage: You're not using TikTok, which is supporting your mental well-being. Keep up this healthy approach!"
            elif 0 < value <= 0.5:
                rec = "✨ Great job on TikTok usage: Your minimal TikTok usage (under 1 hour) is excellent for your mental health. Continue maintaining this balance!"
            elif 0.5 < value <= 1:
                rec = "✨ Maintain TikTok usage: Your low TikTok usage is supporting your mental well-being. Keep up the good work with mindful consumption!"
            elif value >= 5:
                rec = "✨ Focus on TikTok usage: Your TikTok usage is excessive (5+ hours daily). Set strict app limits, use screen time features, and replace usage with offline activities like reading or exercise."
            elif value >= 3:
                rec = "✨ Focus on TikTok usage: Your TikTok usage is high (3+ hours daily). Consider taking regular breaks, disabling notifications, and setting usage reminders to reduce impact."
            elif value >= 2:
                rec = "✨ Focus on TikTok usage: Your TikTok usage is elevated. Try mindful consumption and intentional breaks from short-form content to improve your mental well-being."
            elif value > 1:
                rec = "✨ Focus on TikTok usage: Consider reducing your TikTok usage further. Practice intentional consumption and take regular breaks from short-form content."
            else:
                # Edge case fallback
                rec = "✨ Focus on TikTok usage: Review your TikTok usage habits and consider taking more breaks from short-form content."
        
        # ========== SOCIAL MEDIA PLATFORMS ==========
        elif feature == "social_media_platforms_used":
            # Base recommendations on ACTUAL VALUE, not direction
            if value >= 7:
                rec = "✨ Focus on social media platforms: You're using too many platforms (7+). Consider consolidating to 2-3 core platforms, muting notifications, and unfollowing accounts that don't add value."
            elif value >= 5:
                rec = "✨ Focus on social media platforms: You're using many platforms (5+). Try reducing to 2-3 core platforms and setting boundaries around checking frequency to improve your mental well-being."
            elif value >= 4:
                rec = "✨ Focus on social media platforms: Consider reducing the number of platforms you use. Focus on quality over quantity and curate your feeds intentionally."
            elif value >= 3:
                rec = "✨ Focus on social media platforms: Review which platforms add value to your life. Consider focusing on fewer platforms and being more intentional about usage."
            elif value >= 2:
                rec = "✨ Focus on social media platforms: Consider being more selective about which platforms you actively use to optimize your mental well-being."
            elif value == 0:
                rec = "✨ Great job on social media platforms: You're not using social media platforms, which is excellent for your mental health. Keep maintaining this approach!"
            elif value == 1:
                rec = "✨ Great job on social media platforms: Using just one platform is supporting your mental well-being. Continue this focused approach!"
            elif value == 2:
                rec = "✨ Great job on social media platforms: Using 2 platforms is helping your mental health. Keep maintaining this balanced approach!"
            elif value <= 3:
                rec = "✨ Maintain social media platforms: Your focused platform usage is supporting your well-being. Keep up the good work!"
            else:
                # Edge case fallback
                rec = "✨ Focus on social media platforms: Review and adjust your platform usage to improve your mental well-being score."
        
        # Add recommendation if we have one
        if rec:
            recommendations.append(rec)
    
    return recommendations


def recommendations_from_contrib(contributions):
    """Generate actionable recommendations based on feature contributions."""
    recs = []
    
    # Calculate total absolute contribution for percentages
    total_abs_contrib = sum(abs(c["abs_contribution"]) for c in contributions)
    if total_abs_contrib == 0:
        recs.append("✨ Digital habits look balanced—keep honoring the healthy boundaries you have in place.")
        return recs
    
    # Process each contribution based on direction and value
    for contrib in contributions:
        feature = contrib["feature"]
        value = contrib["value"]
        direction = contrib["direction"]
        abs_contrib = contrib["abs_contribution"]
        pct_contrib = (abs_contrib / total_abs_contrib * 100) if total_abs_contrib > 0 else 0
        
        # Only add recommendations for features that decrease score (need improvement)
        # Or for top contributing features that increase score (reinforce)
        if direction == "decreases_score" or (direction == "increases_score" and pct_contrib >= 20):
            # Screen Time recommendations
            if feature == "screen_time_hours":
                if direction == "decreases_score":
                    if value >= 10:
                        recs.append("📱 <strong>Reduce screen time:</strong> You're spending {:.1f} hours daily on screens. Set digital boundaries, use screen time tracking, and create device-free zones.".format(value))
                    elif value >= 7:
                        recs.append("📱 <strong>Optimize screen time:</strong> Your {:.1f} hours of daily screen use is impacting well-being. Set evening cutoffs and take regular breaks.".format(value))
                    elif value > 5:
                        recs.append("📱 <strong>Monitor screen habits:</strong> Consider scheduling tech-free periods and reducing unnecessary usage.")
                elif direction == "increases_score" and value <= 6:
                    recs.append("📱 <strong>Maintain healthy boundaries:</strong> Your balanced screen time ({:.1f} hrs/day) supports mental well-being. Keep it up!".format(value))
            
            # Sleep Hours recommendations
            elif feature == "sleep_hours":
                if direction == "decreases_score":
                    if value < 5:
                        recs.append("😴 <strong>Prioritize sleep:</strong> Only {:.1f} hours of sleep is severely impacting well-being. Aim for 7-9 hours with a consistent bedtime routine.".format(value))
                    elif value < 7:
                        recs.append("😴 <strong>Improve sleep duration:</strong> Getting {:.1f} hours nightly isn't enough. Aim for 7-9 hours by maintaining a regular sleep schedule.".format(value))
                    elif value < 8:
                        recs.append("😴 <strong>Optimize sleep:</strong> Consider increasing to 7-9 hours per night for optimal mental health.")
                elif direction == "increases_score" and value >= 7:
                    recs.append("😴 <strong>Excellent sleep routine:</strong> Your {:.1f} hours of sleep supports mental well-being. Maintain this healthy pattern!".format(value))
            
            # TikTok Hours recommendations
            elif feature == "hours_on_TikTok":
                if direction == "decreases_score":
                    if value >= 4:
                        recs.append("🎧 <strong>Reduce TikTok usage:</strong> {:.1f} hours daily on TikTok is excessive. Set app limits and replace with offline activities.".format(value))
                    elif value >= 2:
                        recs.append("🎧 <strong>Moderate TikTok consumption:</strong> Take regular breaks, disable notifications, and set usage reminders for your {:.1f} hours/day.".format(value))
                    elif value > 0:
                        recs.append("🎧 <strong>Mindful consumption:</strong> Practice intentional breaks from short-form content.")
            
            # Social Media Platforms recommendations
            elif feature == "social_media_platforms_used":
                if direction == "decreases_score":
                    if value >= 6:
                        recs.append("🌐 <strong>Consolidate platforms:</strong> Using {:.0f} platforms is overwhelming. Reduce to 2-3 core platforms and mute notifications.".format(value))
                    elif value >= 4:
                        recs.append("🌐 <strong>Streamline social media:</strong> Consider reducing from {:.0f} platforms to focus on meaningful connections.".format(value))
                    elif value > 2:
                        recs.append("🌐 <strong>Curate your feeds:</strong> Focus on quality over quantity and be intentional about platform usage.")
        
        # Limit to top 3 recommendations
        if len(recs) >= 3:
            break
    
    # If no specific recommendations, provide a general one
    if not recs:
        # Check if there are any concerning features
        has_concerning = any(c["direction"] == "decreases_score" for c in contributions[:3])
        if has_concerning:
            top_issue = contributions[0]
            feature_label = FEATURE_LABELS.get(top_issue["feature"], top_issue["feature"].replace("_", " ").title())
            recs.append("✨ <strong>Focus on {}</strong>: Review and adjust this habit to improve your mental well-being score.".format(feature_label.lower()))
        else:
            recs.append("✨ <strong>Maintain healthy habits:</strong> Your digital habits look balanced. Keep honoring the boundaries you have in place.")
    
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
    # Anchor to auto-scroll when results appear
    st.markdown('<div id="results-anchor"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">Insight Report</div>', unsafe_allow_html=True)
    
    st.markdown(
        '''
        <div class="disclaimer" style="margin-top: 0.25rem; margin-bottom: 2rem;">
            <strong>Disclaimer:</strong> This is an educational tool, not medical advice. 
            The risk categories are based on a synthetic dataset and should not be interpreted 
            as clinical diagnoses. If you are experiencing mental health concerns, please 
            contact campus counseling services or crisis hotlines.
        </div>
        ''',
        unsafe_allow_html=True
    )

    # Use .get() with defaults to safely read values
    user_inputs = {
        "Screen Time": f"{st.session_state.get('screen_time_hours', 2.0):.1f} hrs/day",
        "Social Platforms": f"{st.session_state.get('social_media_platforms_used', 3)} platform{'s' if st.session_state.get('social_media_platforms_used', 3) != 1 else ''}",
        "TikTok Time": f"{st.session_state.get('hours_on_TikTok', 0.0):.1f} hrs/day",
        "Sleep": f"{st.session_state.get('sleep_hours', 8.0):.1f} hrs/night",
    }

    summary_col, result_col = st.columns([1.25, 0.75])

    with summary_col:
        st.markdown("#### 📊 Input Snapshot")
        chips = "".join(
            [f'<div class="input-chip"><span>{label}</span><strong>{value}</strong></div>' for label, value in user_inputs.items()]
        )
        st.markdown(f'<div class="input-grid">{chips}</div>', unsafe_allow_html=True)
        
        # Personalized recommendations - simple format
        recommendations = generate_simple_recommendations(result["contributions"])
        
        if recommendations:
            st.markdown('<div class="personalized-suggestions">', unsafe_allow_html=True)
            st.markdown("#### 💡 Personalized Recommendations")
            
            # Display simple recommendations
            for rec in recommendations:
                st.markdown(
                    f'''
                    <div class="suggestion-item">
                        <p class="suggestion-text">{rec}</p>
                        <div class="suggestion-meta"></div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
            
            st.markdown('</div>', unsafe_allow_html=True)

    with result_col:
        risk_label = result["risk_category"]
        risk_class = RISK_CLASS_MAP.get(risk_label, "caution")
        # Format display: "Healthy" should not have "risk" suffix
        display_label = risk_label if risk_label == "Healthy" else f"{risk_label} Risk"
        
        colors = RISK_COLORS.get(risk_label, RISK_COLORS["Medium"])
        pill_style = (
            f"background: linear-gradient(120deg, {colors['primary']}, {colors['secondary']});"
            f"color: {colors['text']};"
        )
        
        # Evenly spaced spectrum stops for clearer visual segments (while marker stays exact)
        stop_critical, stop_high, stop_medium, stop_low = 20, 40, 60, 80

        bar_gradient = (
            "linear-gradient(90deg, "
            f"{RISK_COLORS['Critical']['primary']} 0%, {RISK_COLORS['Critical']['primary']} {stop_critical}%, "
            f"{RISK_COLORS['High']['primary']} {stop_critical}%, {RISK_COLORS['High']['primary']} {stop_high}%, "
            f"{RISK_COLORS['Medium']['primary']} {stop_high}%, {RISK_COLORS['Medium']['primary']} {stop_medium}%, "
            f"{RISK_COLORS['Low']['primary']} {stop_medium}%, {RISK_COLORS['Low']['primary']} {stop_low}%, "
            f"{RISK_COLORS['Healthy']['primary']} {stop_low}%, {RISK_COLORS['Healthy']['primary']} 100%)"
        )
        
        st.markdown(
            f'<div class="risk-pill {risk_class}" style="{pill_style}"><span>Risk category</span><strong>{display_label}</strong></div>',
            unsafe_allow_html=True,
        )
        
        # Place marker at the center of its risk segment for clearer categorical signal
        category_centers = {
            "Critical": 10,
            "High": 30,
            "Medium": 50,
            "Low": 70,
            "Healthy": 90,
        }
        score_normalized = category_centers.get(risk_label, 50)
        marker_color = colors["primary"]
        
        # User-friendly mental health score descriptions by risk category
        score_descriptions = {
            "Healthy": "Your mental well-being is in excellent shape. Your digital habits and lifestyle choices are supporting positive mental health.",
            "Low": "Your mental well-being is in good condition. Your current habits are promoting healthy mental health patterns.",
            "Medium": "Your mental well-being shows signs that could benefit from attention. Consider making small adjustments to improve your digital wellness.",
            "High": "Your mental well-being is under strain. Your current habits are significantly impacting your mental health and need attention.",
            "Critical": "Your mental well-being requires immediate attention. Your current lifestyle factors are severely affecting your mental health and substantial changes are needed."
        }
        score_description = score_descriptions.get(risk_label, "Monitor your mental well-being score and consider adjustments to your digital habits.")
        
        bounds_text = (
            f"Critical (&lt; {SCORE_BOUNDARIES['Critical']:.0f}), "
            f"High ({SCORE_BOUNDARIES['Critical']:.0f} to {SCORE_BOUNDARIES['High']:.0f}), "
            f"Medium ({SCORE_BOUNDARIES['High']:.0f} to {SCORE_BOUNDARIES['Medium']:.0f}), "
            f"Low ({SCORE_BOUNDARIES['Medium']:.0f} to {SCORE_BOUNDARIES['Low']:.0f}), "
            f"Healthy (&ge; {SCORE_BOUNDARIES['Low']:.0f})"
        )
        
        st.markdown(
            f"""
            <div class="score-card">
                <p class="eyebrow">Mental Health Score</p>
                <h1>{result["predicted_score"]:.2f}</h1>
                <p>{score_description}</p>
                <div class="score-spectrum">
                    <div class="score-spectrum-label">Score Range</div>
                    <div class="score-spectrum-bar" style="background: {bar_gradient};">
                        <div class="score-spectrum-marker" style="left: {score_normalized}%; color: {marker_color};"></div>
                    </div>
                    <div class="score-spectrum-labels">
                        <span>Critical</span>
                        <span>High</span>
                        <span>Medium</span>
                        <span>Low</span>
                        <span>Healthy</span>
                    </div>
                </div>
                <div class="score-boundaries">
                    <strong>Score Boundaries:</strong> {bounds_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Feature contributions & importance")
    render_feature_contributions(result)

    # Smoothly scroll the freshly-added results into view
    st.markdown(
        """
        <script>
        (function() {
          const anchor = document.getElementById('results-anchor');
          if (!anchor || window.__pmScrolledToResults) return;
          window.__pmScrolledToResults = true;
          requestAnimationFrame(() => {
            anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
          });
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )

    st.button("Start another scenario", use_container_width=True, on_click=restart_survey)


# -----------------------------------------------------------------------------
# UI flow
# -----------------------------------------------------------------------------
clamp_step()
current_step = st.session_state.survey_step
compact_mode = st.session_state.prediction_result is None

if compact_mode:
    st.markdown(
        """
        <style>
        /* Ultra-compact mode for question phase - fit everything in viewport */
        /* Remove ALL top spacing to start at viewport top */
        [data-testid="block-container"] {
            padding: 0 2.5rem 1.5rem !important;
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        /* Remove any top margin/padding from first element */
        [data-testid="block-container"] > div:first-child {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        /* Ensure hero starts at top */
        .hero {
            padding: 1.2rem 1.8rem !important;
            margin-top: 0 !important;
            margin-bottom: 1.4rem !important;
            border-radius: 20px !important;
        }
        /* Remove any spacing from Streamlit's internal elements */
        [data-testid="stVerticalBlock"] > div:first-child {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        .hero h1 {
            font-size: 1.9rem !important;
            margin: 0.4rem 0 0.2rem !important;
        }
        .hero p {
            font-size: 0.88rem !important;
            margin-top: 0.2rem !important;
            line-height: 1.4;
        }
        .hero .chip {
            font-size: 0.75rem !important;
            padding: 0.3rem 0.7rem !important;
        }
        .stepper {
            margin-bottom: 1.4rem !important;
        }
        form[data-testid="stForm"] {
            padding: 1.6rem 1.8rem !important;
            margin-bottom: 0.8rem !important;
            border-radius: 24px !important;
        }
        form[data-testid="stForm"] h3 {
            font-size: 1.35rem !important;
            margin-bottom: 0.4rem !important;
        }
        form[data-testid="stForm"] .stCaption {
            font-size: 0.9rem !important;
            margin-top: -0.2rem !important;
            margin-bottom: 1.2rem !important;
        }
        form[data-testid="stForm"] .stSlider,
        form[data-testid="stForm"] .stNumberInput,
        form[data-testid="stForm"] .stRadio {
            margin-top: 0.8rem !important;
        }
        form[data-testid="stForm"] label {
            font-size: 0.92rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="hero">
        <span class="chip">Digital Well-being Assessment</span>
        <h1>Understand your digital well-being.</h1>
        <p>Answer a few quick questions about your digital habits and sleep patterns. 
        We'll analyze your responses and provide personalized insights into your mental health risk level 
        and actionable recommendations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

def render_digital_step():
    next_clicked = False
    
    with st.form("step-digital", clear_on_submit=False):
        st.markdown("#### Q1 · Daily digital tempo")
        st.caption("Calibrate how much glow time you average each day.")
        
        # Slider uses its own key; we mirror it into screen_time_hours on submit
        screen_time_kwargs = dict(
            min_value=0.0,
            max_value=12.0,
            step=0.1,
            key="screen_time_slider",
            format="%.1f",
        )
        # Initialize from the feature value if present, otherwise default
        if "screen_time_hours" in st.session_state:
            screen_time_kwargs["value"] = float(st.session_state["screen_time_hours"])
        else:
            screen_time_kwargs["value"] = DEFAULT_INPUTS["screen_time_hours"]
        slider_value = st.slider("Screen time (hours / day)", **screen_time_kwargs)
        st.caption("Use the slider to capture your average exposure to screens each day.")

        col_prev, col_spacer, col_next = st.columns([1, 3, 1])
        col_prev.form_submit_button("◀ Back", disabled=True)
        next_clicked = col_next.form_submit_button("Next →")

    if next_clicked:
        # Safe: screen_time_hours is NOT a widget key anymore
        st.session_state["screen_time_hours"] = float(slider_value)
        go_to_step(1)
        st.rerun()


def render_social_step():
    prev_clicked = False
    next_clicked = False
    
    with st.form("step-social", clear_on_submit=False):
        st.markdown("#### Q2 · Social pulse")
        st.caption("Tap the platforms you actively use in a typical week.")

        platform_defs = [
            ("platform_instagram", "Instagram"),
            ("platform_tiktok", "TikTok"),
            ("platform_youtube", "YouTube"),
            ("platform_twitter", "X / Twitter"),
            ("platform_snapchat", "Snapchat"),
            ("platform_reddit", "Reddit"),
            ("platform_facebook", "Facebook"),
            ("platform_other", "Other / niche"),
        ]

        cols = st.columns(3)
        checkbox_keys = []
        for idx, (state_key, label) in enumerate(platform_defs):
            col = cols[idx % 3]
            col.checkbox(
                label,
                value=st.session_state.get(state_key, False),
                key=state_key
            )
            checkbox_keys.append(state_key)

        # TikTok slider uses its own key; we mirror it into hours_on_TikTok on submit
        tiktok_kwargs = dict(
            min_value=0.0,
            max_value=12.0,
            step=0.1,
            key="hours_on_TikTok_slider",
            format="%.1f",
        )
        if "hours_on_TikTok" in st.session_state:
            tiktok_kwargs["value"] = float(st.session_state["hours_on_TikTok"])
        else:
            tiktok_kwargs["value"] = DEFAULT_INPUTS["hours_on_TikTok"]
        slider_value = st.slider("Hours on TikTok (per day)", **tiktok_kwargs)

        col_prev, col_spacer, col_next = st.columns([1, 3, 1])
        prev_clicked = col_prev.form_submit_button("◀ Back")
        next_clicked = col_next.form_submit_button("Next →")

    if prev_clicked or next_clicked:
        # Calculate platform count from checkboxes (values are already in session_state)
        active_count = sum(1 for state_key in checkbox_keys if st.session_state.get(state_key, False))
        st.session_state["social_media_platforms_used"] = active_count
        # Mirror TikTok slider value into the feature key used by prediction
        st.session_state["hours_on_TikTok"] = float(slider_value)
    
    if prev_clicked:
        go_to_step(0)
        st.rerun()
    elif next_clicked:
        go_to_step(2)
        st.rerun()


def render_rest_step():
    prev_clicked = False
    predict_clicked = False
    
    with st.form("step-rest", clear_on_submit=False):
        st.markdown("#### Q3 · Rest & resilience")
        st.caption("Your sleep patterns help us understand your well-being.")
        
        # Slider with key - value explicitly read from session_state for reliability
        sleep_kwargs = dict(
            min_value=0.0,
            max_value=12.0,
            step=0.1,
            key="sleep_hours",
            format="%.1f",
        )
        if "sleep_hours" not in st.session_state:
            sleep_kwargs["value"] = DEFAULT_INPUTS["sleep_hours"]
        sleep_val = st.slider("Sleep hours (per night)", **sleep_kwargs)

        col_prev, col_action = st.columns(2)
        prev_clicked = col_prev.form_submit_button("◀ Back")
        predict_clicked = col_action.form_submit_button("Predict risk 🔮")

    if prev_clicked:
        go_to_step(1)
        st.rerun()
    elif predict_clicked:
        run_prediction()
        st.rerun()


if compact_mode:
    # Check if all steps are complete (prediction has been run)
    all_steps_complete = st.session_state.prediction_result is not None
    render_stepper(current_step, all_complete=all_steps_complete)

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
