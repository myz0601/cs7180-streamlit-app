import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import shap
from datetime import date
import plotly.express as px

# ============================================================
# PAGE CONFIGURATION
# ============================================================


st.set_page_config(
    page_title="Animal Shelter Adoption Predictor",
    page_icon="🐾",
    layout="wide"
)

st.markdown("""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

/* ===== Global font ===== */
html, body, [class*="css"]  {
    font-family: 'Nunito', sans-serif !important;
}

h1, h2, h3 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    letter-spacing: -0.02em !important;
    color: #6A5E58 !important;
}

h1 {
    font-weight: 800 !important;
}

h2 {
    font-weight: 700 !important;
}

h3 {
    font-weight: 700 !important;
}

/* ===== Paragraphs / labels / normal text ===== */
p, li, label {
    font-family: 'Nunito', sans-serif !important;
}

/* ===== Labels ===== */
.stSelectbox label,
.stDateInput label,
.stNumberInput label {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
}

/* ===== Button font ===== */
.stButton > button {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
}
            
        /* ===== Hero title ===== */
    .hero-wrap {
        margin-bottom: 1.2rem;
    }

    /* ===== Hero title ===== */
    .hero-wrap {
        margin-bottom: 1.1rem;
    }

    .hero-title {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: clamp(2.4rem, 4vw, 3.4rem);
        font-weight: 800;
        line-height: 0.98;
        letter-spacing: -0.045em;
        color: #625852;
        margin-bottom: 0.55rem;
        max-width: 900px;
    }

    .hero-subtitle {
        font-family: 'Nunito', sans-serif !important;
        font-size: 1.08rem;
        line-height: 1.55;
        color: #7D726B;
        max-width: 1050px;
        margin-bottom: 0.35rem;
    }

    .hero-paw {
        color: #8AC0D3;
        margin-right: 0.16rem;
    }

    .hero-paw {
        color: #8AC0D3;
        margin-right: 0.18rem;
    }
            
    
    /* ===== Overall page background ===== */
    .stApp {
        background-color: #FFFBF3;
    }

    /* ===== Main content area ===== */
    .block-container {
        background-color: #FFFBF3;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* ===== Sidebar ===== */
    section[data-testid="stSidebar"] {
        background-color: #FBF6EC;
        border-right: 1px solid #F0E7D9;
    }

    /* ===== Sidebar text ===== */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] p {
        color: #5C4B43;
    }

    /* ===== Main text tone ===== */
    h1, h2, h3 {
        color: #5B5368;
    }

    p, label, div {
        color: #6A5E58;
    }
            
        /* ===== Expander / help box ===== */
    details {
        background-color: #FFFDF8 !important;
        border: 1px solid #E9DED0 !important;
        border-radius: 18px !important;
    }

    summary {
        border-radius: 18px !important;
        font-weight: 600 !important;
        color: #6A5E58 !important;
    }

    /* ===== Section headings ===== */
    h1, h2 {
        color: #6A5E58 !important;
    }

    h3 {
        color: #7A6E78 !important;
    }

    /* ===== Labels ===== */
    .stSelectbox label,
    .stDateInput label,
    .stNumberInput label {
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        color: #6A5E58 !important;
        margin-bottom: 0.18rem !important;
    }

    /* ===== Shared input shell ===== */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] > div {
        background: #FFFCF7 !important;
        border: 1.4px solid #E8DDD2 !important;
        border-radius: 16px !important;
        min-height: 48px !important;
        box-shadow: 0 2px 6px rgba(207, 195, 214, 0.10) !important;
        transition: all 0.18s ease !important;
    }

    /* ===== Hover state ===== */
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover,
    div[data-baseweb="base-input"] > div:hover {
        border: 1.8px solid #8AC0D3 !important;
        box-shadow: 0 0 0 4px rgba(138, 192, 211, 0.18) !important;
    }

    /* ===== Focus state ===== */
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="base-input"] > div:focus-within {
        border: 1.8px solid #8AC0D3 !important;
        box-shadow: 0 0 0 3px rgba(138, 192, 211, 0.18) !important;
        background: #FFFFFF !important;
    }

    /* ===== Input text ===== */
    div[data-baseweb="select"] span,
    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input {
        color: #5F5662 !important;
        font-size: 0.98rem !important;
        font-weight: 500 !important;
    }

    /* ===== Dropdown arrow ===== */
    div[data-baseweb="select"] svg {
        color: #9B8FA4 !important;
    }

    /* ===== Date input calendar icon ===== */
    .stDateInput svg {
        color: #9B8FA4 !important;
    }

    /* ===== Number input buttons ===== */
    .stNumberInput button {
        background-color: #F6EEF7 !important;
        border: none !important;
        border-radius: 10px !important;
        color: #7B6F87 !important;
        min-height: 34px !important;
        min-width: 34px !important;
        transition: all 0.18s ease !important;
    }

    .stNumberInput button:hover {
        background-color: #EBDDEB !important;
        color: #665A70 !important;
    }

    /* ===== Caption under age ===== */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #9A9088 !important;
    }

    /* ===== Divider ===== */
    hr {
        border: none !important;
        border-top: 1px solid #E6DDD0 !important;
    }

    /* ===== Primary button ===== */
    .stButton > button {
        background: #CFC3D6 !important;
        color: #5A4F60 !important;
        border: 1px solid #C7B8CF !important;
        border-radius: 18px !important;
        padding: 0.65rem 1.25rem !important;
        font-weight: 800 !important;
        box-shadow: 0 3px 8px rgba(207, 195, 214, 0.14) !important;
        min-height: 58px !important;
    }

    .stButton > button:hover {
        background: #D9CFDF !important;
        color: #564C5C !important;
        border: 1px solid #CFC0D7 !important;
    }
            
        /* ===== Section header ===== */
    .section-kicker {
        font-size: 0.88rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #A58FA8;
        text-transform: uppercase;
        margin-top: 0.6rem;
        margin-bottom: 0.3rem;
    }

    .section-note {
        font-size: 1rem;
        color: #8A7F79;
        margin-top: -0.2rem;
        margin-bottom: 1.1rem;
    }

    /* ===== CTA box ===== */
    .cta-soft {
        background: linear-gradient(180deg, #FFFDF9 0%, #FCF8F1 100%);
        border: 1px dashed #D9CFDC;
        border-radius: 18px;
        padding: 0.72rem 1rem;
        text-align: center;
        color: #7A6E78;
        font-weight: 700;
        margin-top: 0.8rem;
        margin-bottom: 0.5rem;
    }
            
        /* ===== Expander outer box ===== */
    div[data-testid="stExpander"] {
        border: 1px solid #E6DDD0 !important;
        border-radius: 22px !important;
        background: linear-gradient(180deg, #FFFDF9 0%, #FCF7EE 100%) !important;
        box-shadow: 0 2px 8px rgba(207, 195, 214, 0.08) !important;
        margin-top: 0.5rem !important;
        margin-bottom: 1.4rem !important;
        overflow: hidden !important;
    }

    /* ===== Remove weird inner borders/outlines ===== */
    div[data-testid="stExpander"] * {
        box-shadow: none !important;
    }

    /* ===== Expander header container ===== */
    div[data-testid="stExpander"] details > summary {
        padding: 0.95rem 1rem !important;
        background: #F8F2F7 !important;
        border: none !important;
        border-radius: 22px 22px 0 0 !important;
        color: #7A6E78 !important;
        font-weight: 700 !important;
        min-height: 56px !important;
        display: flex !important;
        align-items: center !important;
    }

    /* ===== Header hover ===== */
    div[data-testid="stExpander"] details > summary:hover {
        background: #F3EAF4 !important;
    }

    /* ===== Expanded body wrapper ===== */
    div[data-testid="stExpander"] details[open] > summary {
        border-bottom: 1px solid #E8DDE8 !important;
    }

    /* ===== Expander content area ===== */
    div[data-testid="stExpander"] details > div {
        background: #FFFDF9 !important;
        padding: 1rem 1rem 1.1rem 1rem !important;
    }

    /* ===== Text inside expander ===== */
    div[data-testid="stExpander"] details > div p,
    div[data-testid="stExpander"] details > div li,
    div[data-testid="stExpander"] details > div div {
        color: #6F646C !important;
        line-height: 1.7 !important;
    }
            
        /* ===== Form card wrapper ===== */
    .st-key-form_card {
    background: linear-gradient(180deg, #FFFDF9 0%, #FCF8F1 100%) !important;
    border: 1px solid #E9E1D6 !important;
    border-radius: 28px !important;
    padding: 0.95rem 1.25rem 0.95rem 1.25rem !important;
    box-shadow: 0 2px 10px rgba(207, 195, 214, 0.05) !important;
    margin-top: 0.55rem !important;
    margin-bottom: 1rem !important;
}

    .st-key-form_card hr {
        border: none !important;
        border-top: 1px solid #E8DED2 !important;
        margin-top: 1.2rem !important;
        margin-bottom: 1.1rem !important;
    }
            
        /* ===== Fix vertical centering in select/date/number fields ===== */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] > div {
        display: flex !important;
        align-items: center !important;
        min-height: 46px !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    div[data-baseweb="select"] > div > div {
        display: flex !important;
        align-items: center !important;
        min-height: 46px !important;
    }

    div[data-baseweb="select"] span {
        display: flex !important;
        align-items: center !important;
        line-height: 1 !important;
        min-height: 46px !important;
    }

    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input,
    .stDateInput input,
    .stNumberInput input {
        height: 46px !important;
        line-height: 46px !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        margin: 0 !important;
    }

    /* ===== Number input outer row ===== */
    .stNumberInput > div {
        display: flex !important;
        align-items: center !important;
    }

    /* ===== Slightly tighten labels so they don't feel detached ===== */
    .stSelectbox label,
    .stDateInput label,
    .stNumberInput label {
        margin-bottom: 0.12rem !important;
        line-height: 1.2 !important;
    }
            
        /* ===== Select focus border (the blue outline you see when clicked) ===== */
    div[data-baseweb="select"] > div:focus-within {
        border: 1.8px solid #8AC0D3 !important;
        box-shadow: 0 0 0 4px rgba(138, 192, 211, 0.18) !important;
        background: #FFFFFF !important;
    }

    /* ===== Dropdown menu panel ===== */
    div[role="listbox"] {
        background: #FFFDF9 !important;
        border: 1px solid #D8E8EE !important;
        border-radius: 18px !important;
        box-shadow: 0 10px 24px rgba(138, 192, 211, 0.14) !important;
        padding: 0.2rem !important;
    }

    /* ===== Dropdown options ===== */
    div[role="option"],
    li[role="option"] {
        background: #FFFDF9 !important;
        color: #6A5E58 !important;
        border-radius: 12px !important;
    }

    /* ===== Hovered option ===== */
    div[role="option"]:hover,
    li[role="option"]:hover {
        background: #EDF7FB !important;
        color: #5E544F !important;
    }

    /* ===== Selected option ===== */
    div[role="option"][aria-selected="true"],
    li[role="option"][aria-selected="true"] {
        background: #E4F1F6 !important;
        color: #5A4F60 !important;
        font-weight: 700 !important;
    }

    /* ===== Custom stepper buttons ===== */
    [class*="st-key-stepper_"] .stButton > button {
        background: #F6EEF7 !important;
        color: #7B6F87 !important;
        border: none !important;
        border-radius: 12px !important;
        min-height: 46px !important;
        padding: 0 !important;
        box-shadow: none !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
    }

    [class*="st-key-stepper_"] .stButton > button:hover {
        background: #EBDDEB !important;
        color: #665A70 !important;
        border: none !important;
    }

    .stepper-display {
        background: #FFFCF7;
        border: 1.4px solid #E8DDD2;
        border-radius: 16px;
        min-height: 46px;
        display: flex;
        align-items: center;
        padding: 0 1rem;
        color: #5F5662;
        font-size: 0.98rem;
        font-weight: 500;
        box-shadow: 0 2px 6px rgba(207, 195, 214, 0.10);
    }
            
    
        /* ===== Data Overview hero ===== */
    .overview-hero {
        background: linear-gradient(180deg, #FFFDF9 0%, #FCF8F1 100%);
        border: 1px solid #E9E1D6;
        border-radius: 28px;
        padding: 1.35rem 1.4rem 1.15rem 1.4rem;
        box-shadow: 0 2px 10px rgba(207, 195, 214, 0.05);
        margin-top: 0.2rem;
        margin-bottom: 1.3rem;
    }

    .overview-kicker {
        font-size: 0.86rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #A58FA8;
        margin-bottom: 0.35rem;
    }

    .overview-title {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: clamp(2.2rem, 4vw, 3.2rem);
        font-weight: 800;
        line-height: 1.02;
        letter-spacing: -0.035em;
        color: #625852;
        margin-bottom: 0.45rem;
    }

    .overview-subtitle {
        font-size: 1.05rem;
        line-height: 1.7;
        color: #7D726B;
        max-width: 1100px;
        margin-bottom: 0.9rem;
    }

    .overview-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }

    .overview-chip {
        display: inline-flex;
        align-items: center;
        padding: 0.42rem 0.82rem;
        border-radius: 999px;
        background: #F7F1F8;
        border: 1px solid #E2D7E6;
        color: #75697B;
        font-size: 0.92rem;
        font-weight: 700;
    }

    .overview-chip.blue {
        background: #EEF7FA;
        border: 1px solid #D5EAF1;
        color: #5F7E89;
    }

    .overview-chip.pink {
        background: #F9F0F4;
        border: 1px solid #EACFDA;
        color: #8A6E7A;
    }
            
    /* ===== Data Overview insight card ===== */
        [class*="st-key-overview_card_"] {
        background: linear-gradient(180deg, #FFFDF9 0%, #FCF8F1 100%) !important;
        border: 1px solid #E9E1D6 !important;
        border-radius: 28px !important;
        padding: 1.05rem 1.2rem 1rem 1.2rem !important;
        box-shadow: 0 2px 10px rgba(207, 195, 214, 0.05) !important;
        margin-top: 0.5rem !important;
        margin-bottom: 1.2rem !important;
    }

    .overview-section-kicker {
        font-size: 0.84rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8AC0D3;
        margin-bottom: 0.28rem;
    }

    .overview-section-title {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: clamp(1.8rem, 3vw, 2.5rem);
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.03em;
        color: #625852;
        margin-bottom: 0.35rem;
    }

    .overview-section-note {
        font-size: 1rem;
        line-height: 1.68;
        color: #7D726B;
        margin-bottom: 0.9rem;
        max-width: 1050px;
    }

    .overview-callout {
        background: #F7F1F8;
        border: 1px solid #E3D7E6;
        border-radius: 18px;
        padding: 0.9rem 1rem;
        color: #6E6371;
        margin-top: 0.7rem;
    }

    .overview-callout strong {
        color: #625852;
    }
    
    .overview-callout.purple {
        background: #F7F1F8;
        border: 1px solid #E3D7E6;
        border-radius: 18px;
        padding: 0.9rem 1rem;
        color: #6E6371;
        margin-top: 0.7rem;
    }
            
    .overview-callout.blue {
        background: #EEF7FA;
        border: 1px solid #D7E9F0;
        border-radius: 18px;
        padding: 0.9rem 1rem;
        color: #637A83;
        margin-top: 0.7rem;
    }
            
    
        /* ===== Markdown-friendly callout containers ===== */
    [class*="st-key-overview_callout_purple"] {
        background: #F7F1F8 !important;
        border: 1px solid #E3D7E6 !important;
        border-radius: 18px !important;
        padding: 0.95rem 1rem 0.8rem 1rem !important;
        margin-top: 0.75rem !important;
        margin-bottom: 0.2rem !important;
    }

    [class*="st-key-overview_callout_blue"] {
        background: #EEF7FA !important;
        border: 1px solid #D7E9F0 !important;
        border-radius: 18px !important;
        padding: 0.95rem 1rem 0.8rem 1rem !important;
        margin-top: 0.75rem !important;
        margin-bottom: 0.2rem !important;
    }

    [class*="st-key-overview_callout_"] p {
        color: #6E6371 !important;
        line-height: 1.7 !important;
        margin-bottom: 0.45rem !important;
    }

    [class*="st-key-overview_callout_"] strong {
        color: #5F565F !important;
    }
        
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD ALL ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load("data/best_model.joblib")
    preprocessor = joblib.load("data/preprocessor.joblib")
    explainer = joblib.load("data/shap_explainer.joblib")

    reg_preprocessor = joblib.load("data/regression_preprocessor.joblib")
    reg_model = joblib.load("data/regression_best_model.joblib")

    with open("data/feature_names.json") as f:
        feature_names = json.load(f)

    feat_imp = pd.read_csv("data/feature_importances.csv", index_col=0)
    reg_feat_imp = pd.read_csv("data/regression_feature_importances.csv", index_col=0)

    adoption_by_type = pd.read_csv("data/adoption_by_type.csv")
    adoption_by_condition = pd.read_csv("data/adoption_by_condition.csv")
    with open("data/model_metadata.json") as f:
        metadata = json.load(f)

    return (
        model, preprocessor, explainer,
        reg_preprocessor, reg_model, feature_names,
        feat_imp, reg_feat_imp, adoption_by_type, adoption_by_condition, metadata
    )

(model, preprocessor, explainer,
 reg_preprocessor, reg_model, feature_names,
 feat_imp, reg_feat_imp, adoption_by_type, adoption_by_condition, metadata) = load_artifacts()

# ============================================================
# HELPER: BUILD RAW INPUT DATAFRAME
# ============================================================

def build_input_df(animal_type, primary_color, sex, intake_condition,
                   intake_type, has_name, age_days, intake_month, intake_dayofweek):
    intake_month_sin = np.sin(2 * np.pi * intake_month / 12)
    intake_month_cos = np.cos(2 * np.pi * intake_month / 12)
    intake_dayofweek_sin = np.sin(2 * np.pi * intake_dayofweek / 7)
    intake_dayofweek_cos = np.cos(2 * np.pi * intake_dayofweek / 7)
    intake_is_weekend = 1 if intake_dayofweek >= 5 else 0
    return pd.DataFrame([{
        'animal_type': animal_type,
        'primary_color': primary_color,
        'sex': sex,
        'intake_condition': intake_condition,
        'intake_type': intake_type,
        'has_name': has_name,
        'age_at_intake_days': age_days,
        'intake_is_weekend': intake_is_weekend,
        'intake_month_sin': intake_month_sin,
        'intake_month_cos': intake_month_cos,
        'intake_dayofweek_sin': intake_dayofweek_sin,
        'intake_dayofweek_cos': intake_dayofweek_cos
    }])

# ============================================================
# HELPER: CLEAN FEATURE NAMES FOR DISPLAY
# ============================================================

def clean_feature_name(name):
    name = (name
            .replace('cat__', '').replace('num__', '')
            .replace('cyc__', '').replace('pass__', '')
            .replace('_', ' ').replace('x0 ', '').replace('x1 ', ''))
    return name.title()

def adjust_stepper_value(key, delta, min_value, max_value):
    current = st.session_state.get(key, min_value)
    st.session_state[key] = max(min_value, min(max_value, current + delta))


def stepper_input(label, key, min_value, max_value, default=0):
    if key not in st.session_state:
        st.session_state[key] = default

    st.markdown(f"**{label}**")

    with st.container(key=f"stepper_{key}"):
        c1, c2, c3 = st.columns([6, 1, 1], gap="small")

        with c1:
            st.markdown(
                f"<div class='stepper-display'>{st.session_state[key]}</div>",
                unsafe_allow_html=True
            )

        with c2:
            st.button(
                "−",
                key=f"{key}_minus",
                use_container_width=True,
                on_click=adjust_stepper_value,
                args=(key, -1, min_value, max_value),
                disabled=st.session_state[key] <= min_value
            )

        with c3:
            st.button(
                "+",
                key=f"{key}_plus",
                use_container_width=True,
                on_click=adjust_stepper_value,
                args=(key, 1, min_value, max_value),
                disabled=st.session_state[key] >= max_value
            )

    return st.session_state[key]

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

for key, default in [
    ('prediction_made', False),
    ('prob', None),
    ('days_pred', None),
    ('shap_values', None),
    ('original_inputs', None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# SHARED OPTION LISTS
# ============================================================

month_map = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}
day_map = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
    4: "Friday", 5: "Saturday", 6: "Sunday"
}
animal_type_options = ["CAT", "DOG", "BIRD", "RABBIT", "REPTILE", "GUINEA PIG", "WILD", "OTHER"]
sex_options = ["MALE", "FEMALE", "NEUTERED", "SPAYED", "UNKNOWN"]
color_options = ["BLACK", "BROWN/TAN", "TABBY", "GRAY/BLUE/SILVER", "WHITE/PINK",
                 "LIGHT/WARM", "MULTI-COLORED", "CALICO", "TORTIE", "POINT", "OTHER"]
condition_options = ["NORMAL", "UNDER AGE/WEIGHT", "ILL/INJURED MILD",
                     "ILL/INJURED MODERATE", "ILL/INJURED SEVERE",
                     "FERAL/FRACTIOUS", "BEHAVIOR", "OTHER"]
intake_type_options = ["STRAY", "OWNER SURRENDER", "WILDLIFE",
                       "SEIZED/CONFISCATE", "RETURN", "HOLDING", "OTHER"]

# ============================================================
# DAY THRESHOLDS (based on regression model percentile analysis)
# Median: 25 days | 75th percentile: 60 days | 90th percentile: 112 days
# ============================================================

SHORT_STAY_DAYS = 25   # faster than median
LONG_STAY_DAYS  = 60   # above 75th percentile

# ============================================================
# NAVIGATION
# ============================================================

st.sidebar.title("🐾 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Adoption Predictor", "📈 Data Overview"]
)

if page != "🏠 Adoption Predictor":
    st.session_state.prediction_made = False

# ============================================================
# PAGE 1: ADOPTION PREDICTOR
# ============================================================

if page == "🏠 Adoption Predictor":
    st.markdown("   ")
    st.markdown("   ")
    st.markdown("   ")

    st.image("images/banner.png", use_container_width=True)

    st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">
        <span class="hero-paw">🐾</span>Animal Adoption Predictor
    </div>
    <div style="margin-bottom: 0.75rem;"></div>
    <div class="hero-subtitle">
        Enter an animal's characteristics to estimate its likelihood of adoption
        and how long it may stay in the shelter before being adopted.
    </div>
</div>
""", unsafe_allow_html=True)

    # --- HOW TO USE (collapsible) ---
    with st.expander("✦ How to use this tool"):
        st.markdown("""
**Step 1** — Fill in the animal's characteristics using the dropdowns and age fields below.

**Step 2** — Click **Predict Adoption** to see two results:
- **Adoption likelihood** — how likely this animal is to be adopted, based on historical shelter data
- **Estimated wait time** — if adopted, roughly how long it may take

**Step 3** — Use **Scenario Analysis** at the bottom to explore how realistic changes
— such as neutering the animal or treating a medical condition — might affect the outlook.
        """)
    with st.container(border=False, key="form_card"):

        st.markdown("""
    <div class="section-kicker">Intake Form</div>
    <h2 style="margin-top:0; margin-bottom:0.25rem; color:#6A5E58;">
        Animal Information
    </h2>
    <div class="section-note">
        Fill in the animal's characteristics below to estimate adoption likelihood and expected wait time.
    </div>
    """, unsafe_allow_html=True)

        # --- ROW 1: Animal Type, Sex, Primary Color, Has a Name ---
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            animal_type = st.selectbox("Animal Type", animal_type_options)
        with c2:
            sex = st.selectbox("Sex / Sterilization Status", sex_options)
        with c3:
            primary_color = st.selectbox("Primary Color", color_options)
        with c4:
            has_name_input = st.selectbox("Has a Name?", ["Yes", "No"])
            has_name = 1 if has_name_input == "Yes" else 0

        # --- ROW 2: Intake Condition, Intake Type, Intake Date ---
        c5, c6, c7 = st.columns([1, 1, 1.4])
        with c5:
            intake_condition = st.selectbox("Intake Condition", condition_options)
        with c6:
            intake_type = st.selectbox("Intake Type", intake_type_options)
        with c7:
            intake_date = st.date_input(
                "Intake Date",
                value=date.today()
            )
        intake_month = intake_date.month
        intake_dayofweek = intake_date.weekday()

        # --- ROW 3: Age — Years, Months, Extra Days ---
        st.markdown("**Age at Intake**")
        a1, a2, a3 = st.columns(3)
        with a1:
            age_years = st.number_input("Years", min_value=0, max_value=30,
                                        value=0, step=1)
        with a2:
            age_months = st.number_input("Months", min_value=0, max_value=11,
                                        value=0, step=1)
        with a3:
            age_extra_days = st.number_input("Days", min_value=0,
                                            max_value=30, value=0, step=1)

        age_days_input = age_years * 365 + age_months * 30 + age_extra_days
        if age_days_input > 0:
            st.caption(
                f"Age entered: {age_years} year(s), {age_months} month(s), "
                f"{age_extra_days} day(s) — total {age_days_input} days"
            )

        st.image("images/line1.png", use_container_width=True)

        st.markdown("""
        <div class="cta-soft">
            Ready to generate a prediction for this animal? ✨
        </div>
        """, unsafe_allow_html=True)

        btn_l, btn_c, btn_r = st.columns([1.2, 1.6, 1.2])
        with btn_c:
            predict_clicked = st.button(
                "🐾 Predict Adoption",
                type="primary",
                use_container_width=True
            )

    # --- PREDICT BUTTON ---
    if predict_clicked:

        if animal_type == "WILD":
            st.warning(
                "⚠️ Wild animals are almost never adopted — they are typically "
                "returned to their natural habitat. The prediction below reflects "
                "this historical pattern."
            )

        input_df = build_input_df(
            animal_type, primary_color, sex, intake_condition,
            intake_type, has_name, age_days_input, intake_month, intake_dayofweek
        )
        X_input = preprocessor.transform(input_df)
        prob = model.predict_proba(X_input)[0][1]

        X_reg_input = reg_preprocessor.transform(input_df)
        days_pred = max(0, round(reg_model.predict(X_reg_input)[0]))

        shap_vals = explainer.shap_values(X_input)
        sv = shap_vals[0][0] if isinstance(shap_vals, list) else shap_vals[0]

        st.session_state.prediction_made = True
        st.session_state.prob = prob
        st.session_state.days_pred = days_pred
        st.session_state.shap_values = sv
        st.session_state.original_inputs = {
            'animal_type': animal_type,
            'primary_color': primary_color,
            'sex': sex,
            'intake_condition': intake_condition,
            'intake_type': intake_type,
            'has_name': has_name,
            'has_name_label': has_name_input,
            'age_years': age_years,
            'age_months': age_months,
            'age_extra_days': age_extra_days,
            'age_days': age_days_input,
            'intake_date': intake_date,
            'intake_month': intake_month,
            'intake_dayofweek': intake_dayofweek
        }

    # ---- RESULTS ----
    if st.session_state.prediction_made:

        prob = st.session_state.prob
        days_pred = st.session_state.days_pred
        sv = st.session_state.shap_values
        orig = st.session_state.original_inputs

        st.header("Prediction Results")

        # --- Result 1: Adoption likelihood ---
        st.subheader("1️⃣ Adoption Likelihood")
        st.markdown(f"**Estimated adoption probability: {prob:.1%}**")
        st.progress(float(prob))

        if prob >= 0.6:
            st.success(
                f"✅ **This animal is likely to be adopted** ({prob:.1%} probability). "
                "Standard care and visibility should be sufficient."
            )
        elif prob >= 0.35:
            st.warning(
                f"⚠️ **Adoption is uncertain** ({prob:.1%} probability). "
                "Consider additional promotion or monitoring to improve this animal's chances."
            )
        else:
            st.error(
                f"❌ **This animal is at risk of not being adopted** ({prob:.1%} probability). "
                "Review the suggested actions below."
            )
            st.markdown("""
**Suggested actions:**
- Arrange a professional photoshoot to improve the animal's shelter profile
- Feature the animal on the shelter's social media channels
- Contact rescue organizations that may be able to place the animal
- Consider enrolling the animal in a foster program for more exposure
- If a medical or behavioral issue is present, address it and re-evaluate
            """)

        st.markdown("---")

        # --- Result 2: Estimated wait time ---
        st.subheader("2️⃣ Estimated Wait Time (If Adopted)")
        st.markdown(
            "This estimate comes from a separate model trained only on animals that "
            "were eventually adopted. It answers: **if this animal is adopted, roughly "
            "how long might that take?** A high adoption likelihood and a short wait "
            "is the best outcome."
        )

        weeks = round(days_pred / 7)
        st.markdown(f"### 📅 Estimated wait: **{days_pred} days** (~{weeks} weeks)")

        if prob < 0.35:
            if days_pred > LONG_STAY_DAYS:
                st.error(
                    f"🔴 **If adopted, a long wait is expected ({LONG_STAY_DAYS}+ days).** "
                    "Focus on the suggested actions above to improve adoption likelihood first."
                )
            elif days_pred > SHORT_STAY_DAYS:
                st.warning(
                    f"🟡 **If adopted, a moderate wait is expected.** "
                    "See the suggested actions above."
                )
            else:
                st.success(
                    f"🟢 **If adopted, a relatively short wait is expected "
                    f"(under {SHORT_STAY_DAYS} days).**"
                )
        else:
            if days_pred > LONG_STAY_DAYS:
                st.error(f"🔴 **Long wait expected — {LONG_STAY_DAYS}+ days** (longer than 75% of adopted animals)")
                st.markdown("""
**Suggested actions to speed up adoption:**
- Feature this animal on social media and the shelter website immediately
- Arrange updated or professional photos this week
- Consider a foster-to-adopt program to increase visibility
- Add the animal to any featured or spotlight promotions
                """)
            elif days_pred > SHORT_STAY_DAYS:
                st.warning(f"🟡 **Moderate wait expected**")
                st.markdown("""
**Suggested actions:**
- Monitor adoption interest on a weekly basis
- Consider refreshing the animal's photos or profile description
- Track any changes in health or behavior that may affect adoptability
                """)
            else:
                st.success(
                    f"🟢 **Short wait expected — under {SHORT_STAY_DAYS} days** "
                    f"(faster than most adopted animals). "
                    "No additional action needed at this time."
                )

        st.markdown("---")

        # ---- SCENARIO ANALYSIS ----
        st.header("🔄 Scenario Analysis")
        st.markdown(
            "Use the controls below to explore how realistic changes might "
            "improve this animal's chances."
        )

        analysis_mode = st.radio(
            "What would you like to explore?",
            [
                "Test changes for this animal",
                "Compare with a different animal"
            ]
        )

        if analysis_mode == "Test changes for this animal":
            st.markdown(
                "Adjust the characteristics you can realistically change for this animal. "
                "All other characteristics remain fixed."
            )
            st.caption(
                f"Fixed: Animal Type = {orig['animal_type']} | "
                f"Color = {orig['primary_color']} | "
                f"Intake Type = {orig['intake_type']} | "
                f"Intake Date = {orig['intake_date']} | "
                f"Age = {orig['age_years']}y {orig['age_months']}m {orig['age_extra_days']}d"
            )

            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                sc_sex = st.selectbox(
                    "Sex / Sterilization Status",
                    sex_options,
                    index=sex_options.index(orig['sex']),
                    key="same_sc_sex",
                    help="Explore the potential impact of neutering or spaying this animal"
                )
            with sc2:
                sc_has_name = st.selectbox(
                    "Has a Name?",
                    ["Yes", "No"],
                    index=0 if orig['has_name'] == 1 else 1,
                    key="same_sc_has_name",
                    help="Assigning a name has been associated with higher adoption rates"
                )
                sc_has_name_val = 1 if sc_has_name == "Yes" else 0
            with sc3:
                sc_condition = st.selectbox(
                    "Intake Condition",
                    condition_options,
                    index=condition_options.index(orig['intake_condition']),
                    key="same_sc_condition",
                    help="Explore how treatment of a medical condition might change the outlook"
                )

            sc_animal_type = orig['animal_type']
            sc_primary_color = orig['primary_color']
            sc_intake_type = orig['intake_type']
            sc_age = orig['age_days']
            sc_month = orig['intake_month']
            sc_day = orig['intake_dayofweek']

        else:
            st.markdown(
                "Enter a completely different animal's characteristics to compare "
                "predicted outcomes side by side with the current animal."
            )

            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                sc_animal_type = st.selectbox(
                    "Animal Type", animal_type_options,
                    index=animal_type_options.index(orig['animal_type']),
                    key="diff_sc_animal_type"
                )
            with sc2:
                sc_sex = st.selectbox(
                    "Sex / Sterilization Status", sex_options,
                    index=sex_options.index(orig['sex']),
                    key="diff_sc_sex"
                )
            with sc3:
                sc_primary_color = st.selectbox(
                    "Primary Color", color_options,
                    index=color_options.index(orig['primary_color']),
                    key="diff_sc_primary_color"
                )
            with sc4:
                sc_has_name = st.selectbox(
                    "Has a Name?", ["Yes", "No"],
                    index=0 if orig['has_name'] == 1 else 1,
                    key="diff_sc_has_name"
                )
                sc_has_name_val = 1 if sc_has_name == "Yes" else 0

            sc5, sc6, sc7 = st.columns([1, 1, 1.4])
            with sc5:
                sc_condition = st.selectbox(
                    "Intake Condition", condition_options,
                    index=condition_options.index(orig['intake_condition']),
                    key="diff_sc_condition"
                )
            with sc6:
                sc_intake_type = st.selectbox(
                    "Intake Type", intake_type_options,
                    index=intake_type_options.index(orig['intake_type']),
                    key="diff_sc_intake_type"
                )
            with sc7:
                sc_intake_date = st.date_input(
                    "Intake Date",
                    value=orig.get('intake_date', date.today()),
                    key="diff_sc_intake_date"
                )
            sc_month = sc_intake_date.month
            sc_day = sc_intake_date.weekday()

            st.markdown("**Age at Intake**")
            sa1, sa2, sa3 = st.columns(3)
            with sa1:
                sc_age_years = st.number_input(
                    "Years", min_value=0, max_value=30,
                    value=orig['age_years'], step=1,
                    key="diff_sc_age_years"
                )
            with sa2:
                sc_age_months = st.number_input(
                    "Months", min_value=0, max_value=11,
                    value=orig['age_months'], step=1,
                    key="diff_sc_age_months"
                )
            with sa3:
                sc_age_extra_days = st.number_input(
                    "Extra Days", min_value=0, max_value=30,
                    value=orig['age_extra_days'], step=1,
                    key="diff_sc_age_extra_days"
                )
            sc_age = sc_age_years * 365 + sc_age_months * 30 + sc_age_extra_days
            if age_days_input > 0:
                st.caption(
                    f"Age entered: {sc_age_years} year(s), {sc_age_months} month(s), "
                    f"{sc_age_extra_days} day(s) — total {sc_age} days"
                )

        # Compute scenario predictions
        sc_input_df = build_input_df(
            sc_animal_type, sc_primary_color, sc_sex,
            sc_condition, sc_intake_type, sc_has_name_val,
            sc_age, sc_month, sc_day
        )
        sc_X = preprocessor.transform(sc_input_df)
        sc_prob = model.predict_proba(sc_X)[0][1]

        sc_X_reg = reg_preprocessor.transform(sc_input_df)
        sc_days_pred = max(0, round(reg_model.predict(sc_X_reg)[0]))

        sc_diff = sc_prob - prob
        sc_days_diff = sc_days_pred - days_pred

        st.markdown("---")

        # Adoption likelihood comparison
        st.subheader("Adoption Likelihood")
        res1, res2 = st.columns(2)
        with res1:
            st.metric("Current Animal", f"{prob:.1%}")
        with res2:
            label = "Modified Profile" if analysis_mode == "Test changes for this animal" else "Comparison Animal"
            st.metric(label, f"{sc_prob:.1%}", delta=f"{sc_diff:+.1%}",
                      delta_color="normal")

        if abs(sc_diff) <= 0.05:
            st.info("ℹ️ This change would have minimal impact on adoption likelihood.")
        elif sc_diff > 0:
            st.success(
                f"✅ This change is associated with a {sc_diff:.1%} increase in adoption likelihood."
            )
        else:
            st.error(
                f"❌ This change is associated with a {abs(sc_diff):.1%} decrease in adoption likelihood."
            )

        # Wait time comparison
        st.subheader("Estimated Wait Time (If Adopted)")
        d1, d2 = st.columns(2)
        with d1:
            st.metric("Current Animal", f"{days_pred} days")
        with d2:
            st.metric(
                label,
                f"{sc_days_pred} days",
                delta=f"{sc_days_diff:+d} days",
                delta_color="inverse"  # fewer days is better
            )
        st.caption(
            "A high adoption likelihood and a short wait is the best outcome."
        )

# ============================================================
# PAGE 2: DATA OVERVIEW
# ============================================================

elif page == "📈 Data Overview":
    st.markdown("   ")
    st.markdown("   ")
    st.markdown("   ")
    st.markdown("""
    <div class="overview-hero">
        <div class="overview-kicker">Data Dashboard</div>
        <div class="overview-kicker">   </div>
        <div class="overview-title">📈 Data Overview</div>
        <div style="margin-bottom: 0.75rem;"></div>
        <div class="overview-subtitle">
            Explore historical shelter patterns to better understand adoption likelihood,
            wait-time outcomes, and the broader context behind the Adoption Predictor.
        </div>
        <div class="overview-chip-row">
            <div class="overview-chip blue">Interactive charts</div>
            <div class="overview-chip">Historical context</div>
            <div class="overview-chip pink">Actionable insight</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- LOAD DATA ----
    @st.cache_data
    def load_overview_data():
        df = pd.read_csv("data/X_train_raw.csv")
        y = pd.read_csv("data/y_train.csv").values.ravel()
        df['adopted'] = y
        return df

    @st.cache_data
    def load_regression_data():
        return pd.read_csv("data/cleaned_data_regression.csv")

    df_overview = load_overview_data()
    df_reg = load_regression_data()

    # Pre-compute age groups
    age_bins = [0, 180, 730, 2555, float('inf')]
    age_labels = ['Under 6 months', '6 months – 2 years', '2 – 7 years', 'Over 7 years']
    df_overview['age_group'] = pd.cut(
        df_overview['age_at_intake_days'],
        bins=age_bins,
        labels=age_labels,
        right=False
    )

    # ============================================================
    # CHART 1: INTERACTIVE ADOPTION RATE EXPLORER
    # ============================================================
    st.markdown("   ")
    with st.container(border=False, key="overview_card_1"):

        st.markdown("""
        <div class="overview-section-kicker">Insight 01</div>
        <div class="overview-section-title">Adoption Rate Explorer</div>
        <div style="margin-bottom: 0.6rem;"></div>
        <div class="overview-section-note">
            Select a characteristic to see how adoption rates differ across its categories.
            Hover over any bar to see the exact adoption rate and total number of animals
            in that group. All figures are based on historical records from this shelter.
        </div>
        """, unsafe_allow_html=True)

    variable_options = {
        "Animal Type": "animal_type",
        "Intake Condition": "intake_condition",
        "Sex / Sterilization Status": "sex",
        "Intake Type": "intake_type"
    }

    selected_label = st.selectbox(
        "Explore adoption rate by:",
        options=list(variable_options.keys()),
        key="explorer_variable"
    )
    selected_col = variable_options[selected_label]

    explorer_df = (
        df_overview.groupby(selected_col)['adopted']
        .agg(['mean', 'count'])
        .reset_index()
    )
    explorer_df.columns = [selected_col, 'adoption_rate', 'count']
    explorer_df = explorer_df.sort_values('adoption_rate', ascending=False)
    explorer_df['adoption_pct'] = (explorer_df['adoption_rate'] * 100).round(1)
    explorer_df['color'] = explorer_df['adoption_rate'].apply(
        lambda r: '#8AC0D3' if r >= 0.25 else '#CFC3D6' if r >= 0.1 else '#ECCBD5'
    )
    explorer_df['category'] = explorer_df[selected_col].astype(str)

    fig1 = px.bar(
        explorer_df,
        x='category',
        y='adoption_pct',
        color='color',
        color_discrete_map='identity',
        custom_data=['count'],
        labels={'category': selected_label, 'adoption_pct': 'Adoption Rate (%)'},
        title=f'Historical Adoption Rate by {selected_label}'
    )
    fig1.update_traces(
        hovertemplate=(
            '<b>%{x}</b><br>'
            'Adoption Rate: %{y:.1f}%<br>'
            'Total animals in group: %{customdata[0]:,}<extra></extra>'
        )
    )
    fig1.update_layout(
        showlegend=False,
        xaxis_title=None,
        xaxis_tickangle=-20,
        paper_bgcolor="#FFFDF9",
        plot_bgcolor="#FFFDF9",
        font=dict(color="#6A5E58"),
        yaxis=dict(gridcolor="#EAE2D8"),
        xaxis=dict(showgrid=False)
    )
    st.plotly_chart(
        fig1,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.caption(
        "🔵 Blue = 25%+ adoption rate  |  🟣 Lavender = 10–25%  |  "
        "🌸 Blush = below 10%  |  Hover over bars for exact figures."
    )

    if selected_col == 'animal_type':
        top3 = explorer_df.head(3)['category'].str.title().tolist()
        insight_text = (
            f"Based on shelter records, the three most adoptable animal types are "
            f"**{top3[0]}**, **{top3[1]}**, and **{top3[2]}**. "
            "Reptiles and birds are rarely adopted through the shelter — most are placed "
            "through rescue organizations. Wild animals are almost never adopted and are "
            "typically returned to their natural habitat."
        )

    elif selected_col == 'intake_condition':
        normal_rate = explorer_df.loc[
            explorer_df[selected_col] == 'NORMAL', 'adoption_rate'
        ].values
        severe_rate = explorer_df.loc[
            explorer_df[selected_col] == 'ILL/INJURED SEVERE', 'adoption_rate'
        ].values

        if len(normal_rate) > 0 and len(severe_rate) > 0:
            gap = normal_rate[0] - severe_rate[0]
            insight_text = (
                f"Animals arriving in normal condition are adopted at a rate of "
                f"**{normal_rate[0]:.1%}**, compared to only **{severe_rate[0]:.1%}** "
                f"for animals with severe illness or injury — a gap of **{gap:.1%}**. "
                "Treating a medical condition before putting an animal up for adoption "
                "can meaningfully improve its chances."
            )
        else:
            insight_text = (
                "Animals in normal condition have the highest adoption rates. "
                "The more severe the condition at intake, the lower the adoption rate. "
                "Treating a medical condition before putting an animal up for adoption "
                "can meaningfully improve its chances."
            )

    elif selected_col == 'sex':
        male_rate = explorer_df.loc[
            explorer_df[selected_col] == 'MALE', 'adoption_rate'
        ].values
        neutered_rate = explorer_df.loc[
            explorer_df[selected_col] == 'NEUTERED', 'adoption_rate'
        ].values
        female_rate = explorer_df.loc[
            explorer_df[selected_col] == 'FEMALE', 'adoption_rate'
        ].values
        spayed_rate = explorer_df.loc[
            explorer_df[selected_col] == 'SPAYED', 'adoption_rate'
        ].values

        if all(len(r) > 0 for r in [male_rate, neutered_rate, female_rate, spayed_rate]):
            male_lift = neutered_rate[0] - male_rate[0]
            female_lift = spayed_rate[0] - female_rate[0]
            insight_text = (
                f"Sterilization is associated with noticeably higher adoption rates. "
                f"Neutered males are adopted at **{neutered_rate[0]:.1%}**, compared to "
                f"**{male_rate[0]:.1%}** for intact males — a difference of **{male_lift:.1%}**. "
                f"Spayed females are adopted at **{spayed_rate[0]:.1%}**, compared to "
                f"**{female_rate[0]:.1%}** for intact females — a difference of **{female_lift:.1%}**."
            )
        else:
            insight_text = (
                "Sterilized animals have noticeably higher adoption rates than intact animals."
            )

    else:
        best_intake = explorer_df.iloc[0]
        worst_intake = explorer_df.iloc[-1]

        def fmt(val):
            return str(val).replace('/', ' / ').title()

        insight_text = (
            f"**{fmt(best_intake[selected_col])}** animals have the highest adoption rate "
            f"at **{best_intake['adoption_rate']:.1%}**. "
            f"**{fmt(worst_intake[selected_col])}** animals have the lowest rate at "
            f"**{worst_intake['adoption_rate']:.1%}**. "
            "Intake type helps set expectations for how easy or difficult adoption may be."
        )

    with st.container(border=False, key="overview_callout_purple_1"):
        st.markdown("**What this tells us:**")
        st.markdown(insight_text)

    st.markdown("---")

    # ============================================================
    # CHART 2: ADOPTION RATE BY AGE GROUP (FILTERED BY ANIMAL TYPE)
    # ============================================================

    with st.container(border=False, key="overview_card_2"):

        st.markdown("""
        <div class="overview-section-kicker" style="color:#A58FA8;">Insight 02</div>
        <div class="overview-section-title">Adoption Rate by Age Group</div>
        <div style="margin-bottom: 0.6rem;"></div>
        <div class="overview-section-note">
            Age is one of the strongest predictors in our models, but it affects
            different species very differently. Select an animal type below to see
            how adoption rates change with age for that species.
        </div>
        """, unsafe_allow_html=True)

    animal_options = sorted(df_overview['animal_type'].dropna().unique().tolist())
    default_animal = 'DOG' if 'DOG' in animal_options else animal_options[0]

    selected_animal = st.selectbox(
        "Select animal type:",
        options=animal_options,
        index=animal_options.index(default_animal),
        key="age_animal_filter"
    )

    df_age_filtered = df_overview[
        df_overview['animal_type'] == selected_animal
    ].dropna(subset=['age_group'])

    age_summary = (
        df_age_filtered.groupby('age_group', observed=True)['adopted']
        .agg(['mean', 'count'])
        .reset_index()
    )
    age_summary.columns = ['age_group', 'adoption_rate', 'count']
    age_summary['adoption_pct'] = (age_summary['adoption_rate'] * 100).round(1)
    age_summary['color'] = age_summary['adoption_rate'].apply(
        lambda r: '#8AC0D3' if r >= 0.25 else '#CFC3D6' if r >= 0.1 else '#ECCBD5'
    )
    age_summary['age_group'] = age_summary['age_group'].astype(str)

    fig2 = px.bar(
        age_summary,
        x='age_group',
        y='adoption_pct',
        color='color',
        color_discrete_map='identity',
        custom_data=['count'],
        labels={'age_group': 'Age Group', 'adoption_pct': 'Adoption Rate (%)'},
        title=f'Adoption Rate by Age Group — {selected_animal.title()}'
    )
    fig2.update_traces(
        hovertemplate=(
            '<b>%{x}</b><br>'
            'Adoption Rate: %{y:.1f}%<br>'
            'Total animals in group: %{customdata[0]:,}<extra></extra>'
        )
    )
    fig2.update_layout(showlegend=False,
                        xaxis_title=None,
                        paper_bgcolor="#FFFDF9",
                        plot_bgcolor="#FFFDF9",
                        font=dict(color="#6A5E58"),
                        yaxis=dict(gridcolor="#EAE2D8"),
                        xaxis=dict(showgrid=False)
                        )
    st.plotly_chart(
        fig2,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.caption(
        "🔵 Blue = 25%+ adoption rate  |  🟣 Lavender = 10–25%  |  "
        "🌸 Blush = below 10%  |  Hover over bars for exact figures."
    )

    if len(age_summary) > 0:
        best_age = age_summary.loc[age_summary['adoption_rate'].idxmax(), 'age_group']
        worst_age = age_summary.loc[age_summary['adoption_rate'].idxmin(), 'age_group']
        best_rate = age_summary['adoption_rate'].max()
        worst_rate = age_summary['adoption_rate'].min()
        gap = best_rate - worst_rate

        insight_text = (
            f"For **{selected_animal.title()}s**, **{best_age}** animals have the highest "
            f"adoption rate at **{best_rate:.1%}**, while **{worst_age}** animals have the "
            f"lowest at **{worst_rate:.1%}** — a gap of **{gap:.1%}**. "
            "Use the Scenario Analysis on the Adoption Predictor page to see how "
            "age combines with other factors for a specific animal."
        )

        with st.container(border=False, key="overview_callout_purple_2"):
            st.markdown("**What this tells us:**")
            st.markdown(insight_text)

        st.divider()

    # ============================================================
    # CHART 3: LENGTH-OF-STAY DISTRIBUTION WITH INTERACTIVE MARKER
    # ============================================================

    with st.container(border=False, key="overview_card_3"):

        st.markdown("""
        <div class="overview-section-kicker" style="color:#8AC0D3;">Insight 03</div>
        <div class="overview-section-title">How Long Do Adopted Animals Stay?</div>
        <div style="margin-bottom: 0.6rem;"></div>
        <div class="overview-section-note">
            This chart shows the distribution of actual shelter stay lengths for animals
            that were eventually adopted. Enter a number of days below to see where it
            falls relative to real adoption outcomes — for example, enter an estimated
            wait time from the Adoption Predictor to put it in context.
        </div>
        """, unsafe_allow_html=True)

    days = df_reg['days_to_adoption'].dropna()
    days_capped = days[days <= 200]

    median_days = int(days.median())
    pct75_days = int(days.quantile(0.75))
    pct90_days = int(days.quantile(0.90))

    marker_days = st.number_input(
        "Enter a number of days to mark on the chart:",
        min_value=0,
        max_value=int(days.max()),
        value=median_days,
        step=1,
        key="day_marker"
    )

    fig3 = px.histogram(
        days_capped,
        nbins=40,
        labels={
            'value': 'Days in Shelter Before Adoption',
            'count': 'Number of Animals'
        },
        title='Distribution of Shelter Stay Length (Adopted Animals, 0–200 Days)',
        color_discrete_sequence=['#8AC0D3']
    )

    fig3.update_traces(
        hovertemplate='%{x} days<br>Number of animals: %{y:,}<extra></extra>',
        marker_line_color='#FFFFFF',
        marker_line_width=1
    )

    fig3.add_vline(
        x=median_days,
        line_dash='dash',
        line_color='#8AC0D3',
        annotation=dict(
            text=f'Median<br>{median_days}d',
            yref='paper', y=0.99,
            font=dict(color='#5F7E89', size=11),
            bgcolor='rgba(255,255,255,0.85)',
            borderpad=3
        )
    )

    fig3.add_vline(
        x=pct75_days,
        line_dash='dash',
        line_color='#CFC3D6',
        annotation=dict(
            text=f'75th pct<br>{pct75_days}d',
            yref='paper', y=0.85,
            font=dict(color='#74697B', size=11),
            bgcolor='rgba(255,255,255,0.85)',
            borderpad=3
        )
    )

    fig3.add_vline(
        x=pct90_days,
        line_dash='dash',
        line_color='#ECCBD5',
        annotation=dict(
            text=f'90th pct<br>{pct90_days}d',
            yref='paper', y=0.71,
            font=dict(color='#8B6D78', size=11),
            bgcolor='rgba(255,255,255,0.85)',
            borderpad=3
        )
    )

    percentile = int((days <= marker_days).mean() * 100)

    fig3.add_vline(
        x=marker_days,
        line_dash='solid',
        line_color='#8A6FA3',
        line_width=2.5,
        annotation=dict(
            text=f'Your input<br>{marker_days}d ({percentile}th pct)',
            yref='paper', y=0.57,
            font=dict(color='#6D5A80', size=11),
            bgcolor='rgba(255,255,255,0.9)',
            borderpad=3
        )
    )

    fig3.update_layout(
        xaxis_title='Days in Shelter Before Adoption',
        yaxis_title='Number of Animals',
        showlegend=False,
        paper_bgcolor="#FFFDF9",
        plot_bgcolor="#FFFDF9",
        font=dict(color="#6A5E58"),
        yaxis=dict(gridcolor="#EAE2D8"),
        xaxis=dict(showgrid=False)
    )

    st.plotly_chart(
        fig3,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.caption(
        f"Chart is capped at 200 days for readability. "
        f"A small number of animals took longer — up to {int(days.max())} days. "
        f"Based on {len(days):,} adopted animals. Hover over bars for exact counts."
    )

    insight_text = (
        f"Half of all adopted animals found a home within **{median_days} days**, "
        f"and three quarters within **{pct75_days} days**. "
        f"The value you entered — **{marker_days} days** — is shorter than the shelter "
        f"stay of **{percentile}%** of adopted animals on record. "
        "Most animals are adopted relatively quickly, but a small number stay much "
        "longer, which pulls the average higher than you might expect."
    )

    with st.container(border=False, key="overview_callout_blue_3"):
        st.markdown("**What this tells us:**")
        st.markdown(insight_text)

    st.markdown("   ")
    st.image("images/line2.png", use_container_width=True)
    st.markdown("   ")

    # ---- DATASET SUMMARY ----
    with st.container(border=False, key="overview_card_4"):

        st.markdown("""
        <div class="overview-section-kicker" style="color:#A58FA8;">Dataset Notes</div>
        <div class="overview-section-title">About This Data</div>
        <div style="margin-bottom: 0.6rem;"></div>
        <div class="overview-section-note">
            These charts are based on historical shelter records used to build the predictor
            and to provide context for model outputs.
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=False, key="overview_callout_blue_4"):
            st.markdown("""
    **53,441 animal intake records** from the Long Beach Animal Shelter are included in this dataset.

    Of those, **11,976 animals (22.4%)** were adopted.

    The adoption-rate charts are based on the shelter records used to build the prediction models.
    The length-of-stay chart is based on the **11,976 adopted animals only**, since a stay length
    can only be calculated for animals with a recorded adoption date.

    These figures reflect historical patterns and may not capture more recent changes in shelter
    operations, outreach strategy, or adoption trends.
            """)
