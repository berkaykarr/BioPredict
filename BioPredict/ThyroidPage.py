import base64
from datetime import datetime
from pathlib import Path
import pickle
import streamlit as st
import pandas as pd
import joblib
import time
from sklearn.preprocessing import MinMaxScaler
from textwrap import dedent

ARTIFACT_DIR = Path(__file__).resolve().parent  
MODEL_PATH = ARTIFACT_DIR / "xgb_model.pkl"
SCALER_PATH = ARTIFACT_DIR / "thyroid_scaler.pkl" 

@st.cache_resource
def load_model(model_path, scaler_path, model_mtime, scaler_mtime):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model(
    str(MODEL_PATH), str(SCALER_PATH),
    MODEL_PATH.stat().st_mtime, SCALER_PATH.stat().st_mtime
)

if not isinstance(scaler, MinMaxScaler):
    st.error(f"Scaler type is {type(scaler).__name__}. It must be MinMaxScaler.")
    st.stop()


if getattr(scaler, "n_features_in_", None) != 1:
    st.error(f"Scaler n_features_in_ = {getattr(scaler,'n_features_in_', None)}. 1 olmalı.")
    st.stop()

fn = getattr(scaler, "feature_names_in_", None)
if list(fn) != ["Age"]:
    st.error(f"Scaler feature_names_in_ = {fn}. ['Age'] olmalı.")
    st.stop()
with open('category_mappings.pkl', 'rb') as f:
    category_mappings = pickle.load(f)

user_name = st.session_state.get("user_name", "").strip()
user_gender = st.session_state.get("user_gender", "Select")

def validate_inputs(inputs):
    for key, val in inputs.items():
        if val in (None, "Select"):
            st.warning(f"Please provide a valid value for '{key}'!", icon="⚠️")
            return False
        
        if isinstance(val, (int, float)) and val < 0:
            st.warning(f"'{key}' must be a positive value!", icon="⚠️")
            return False

    return True


st.markdown(
    """
    <h1 style="
        font-size: 42px;
        font-weight: 900;
        background: linear-gradient(90deg, #00FFF0, #FF2DA3, #FF6EC7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: transparent;
        text-align: center;
        margin-bottom: 0px;
        font-family: 'Arial Black', sans-serif;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.35);
    ">
    Thyroid Cancer</h1>

    <h2 style="
        font-size: 26px;
        font-weight: 800;
        background: linear-gradient(90deg, #00FFF0, #FF2DA3, #FF6EC7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: transparent;
        text-align: center;
        margin-top: 0px;
        margin-bottom: 10px;
        font-family: 'Arial Black', sans-serif;
        text-shadow: 1.5px 1.5px 3px rgba(0, 0, 0, 0.25);
    ">
    Relapse Prediction After Treatment</h2>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div style="
        text-align: center;
        font-size: 18px;
        font-weight: 400;
        color: #3A3B7A;
        margin-bottom: 30px;
        animation: fadeIn 2.2s ease-in;
        font-style: italic;
        line-height: 1.6;
    ">
        <div>Thyroid disorders can affect metabolism, energy levels, and overall health.</div>
        <div>Accurate diagnosis and treatment are essential for hormonal balance and well-being.</div>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("This tool is for educational purposes only and does not provide a medical diagnosis. Results should be interpreted in conjunction with clinical evaluation.")


with st.expander("", expanded=True):
    has_name = bool(user_name and str(user_name).strip())
    has_gender = user_gender not in ["Select", "", None]

    name_text = user_name.strip() if has_name else "Name Not Specified"
    initials = (user_name.strip()[0] if has_name else "?").upper()

    g = (user_gender or "").strip().lower()
    if g in ["male", "erkek", "m", "man"]:
        gender_label = f"♂️ {user_gender}"
        gender_bg = "#DBEAFE"; gender_fg = "#1E3A8A"
    elif g in ["female", "kadın", "kadin", "f", "woman"]:
        gender_label = f"♀️ {user_gender}"
        gender_bg = "#FCE7F3"; gender_fg = "#9D174D"
    elif has_gender:
        gender_label = f"⚧️ {user_gender}"
        gender_bg = "#E5E7EB"; gender_fg = "#111827"
    else:
        gender_label = "⚧️ Not Specified"
        gender_bg = "#E5E7EB"; gender_fg = "#111827"

    html = dedent(f"""
    <style>
    .profile-card {{
      position: relative; border-radius: 16px; padding: 16px 18px; margin: 4px 0 8px 0;
      background: rgba(255,255,255,0.75); border: 1px solid rgba(99,102,241,0.22);
      box-shadow: 0 8px 28px rgba(17,17,26,0.08); backdrop-filter: blur(6px); overflow: hidden;
    }}
    @media (prefers-color-scheme: dark) {{
      .profile-card {{ background: rgba(20,20,28,0.55); border-color: rgba(99,102,241,0.28);
                       box-shadow: 0 10px 32px rgba(0,0,0,0.35); }}
    }}
    .profile-card::before {{
      content:""; position:absolute; inset:0; padding:1px; border-radius:inherit;
      background: linear-gradient(90deg,#00FFF0,#7A5CFA,#FF2DA3,#FF6EC7,#00FFF0);
      -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
      -webkit-mask-composite: xor; mask-composite: exclude; pointer-events:none;
    }}
    .profile-inner {{
      display: flex;
      justify-content: center;   
      align-items: center;
      width: 100%;
    }}
    .info-col {{
      display: flex;
      flex-wrap: wrap;
      justify-content: center;    
      align-items: center;
      gap: 12px;
      width: 100%;
    }}
    .info-col {{ display:flex; flex-wrap:wrap; align-items:center; gap:10px; }}
    .chip {{
      display:inline-flex; align-items:center; gap:8px; padding:8px 12px; border-radius:999px;
      font-weight:700; font-size:13px; border:1px solid rgba(99,102,241,0.22);
      background: rgba(224,236,255,0.35); color:#171746;
    }}
    @media (prefers-color-scheme: dark) {{
      .chip {{ color:#E5E7EB; background:rgba(99,102,241,0.12); border-color:rgba(99,102,241,0.28); }}
    }}
    .label {{ font-size:12px; font-weight:800; letter-spacing:.02em; text-transform:uppercase; opacity:.75; color:#5b5f97; margin-right:6px; }}
    @media (prefers-color-scheme: dark) {{ .label {{ color:#A5B4FC; }} }}
    </style>

    <div class="profile-card">
      <div class="profile-inner">
        <div class="info-col">
          <div class="chip">
            <span class="label">Name</span>
            <span style="font-weight:800;">{name_text}</span>
          </div>
          <div class="chip" style="background:{gender_bg}55;border-color:{gender_fg}33;color:{gender_fg};">
            <span class="label">Gender</span>
            <span style="font-weight:800;">{gender_label}</span>
          </div>
        </div>
      </div>
    </div>
    """)

    st.markdown(html, unsafe_allow_html=True)

 
age = st.slider("Age", 1, 120)

st.markdown("### Clinical Findings")

col1 = st.columns(1)[0]  

with col1:
    tumor = st.selectbox("Tumor Status", [
        "Select",
        "tumor that is 1 cm or smaller",
        "tumor larger than 1 cm but not larger than 2 cm",
        "tumor larger than 2 cm but not larger than 4 cm",
        "tumor larger than 4 cm",
        "tumor that has grown outside the thyroid",
        "tumor that has invaded nearby structures"
    ])

col4, col5, col6 = st.columns(3)
with col4:
    physical_exam = st.selectbox("Physical Examination", ["Select", "Normal", "Abnormal"])
    adenopathy = st.selectbox("Adenopathy", [
        "Select",
        "No Lympth Adenopathy",
        "Left Side Body Adenopathy",
        "Right Side Body Adenopathy",
        "Extensive and Widespread"
    ])

with col5:
    pathology = st.selectbox("Type of Thyroid Cancer", ["Select", "Papillary", "Follicular", "Other"])
    focality = st.selectbox("Focality (Number of Tumor Foci)", ["Select", "Unifocal", "Multifocal"])

with col6:
    risk = st.selectbox("Risk Status", ["Select", "Low", "Intermediate", "High"])
    thyroid_function = st.selectbox("Thyroid Function", ["Select", "Normal", "Hypo", "Hyper"])

st.markdown("---")
st.markdown("###  Staging and Treatment Response")

col7, col8 = st.columns(2)
with col7:
    nodes = st.selectbox("Lymph Node Status", [
        "Select",
        "No evidence of regional lymph node metastasis",
        "Regional lymph node metastasis in the central neck",
        "Regional lymph node metastasis in the lateral neck"
    ])

with col8:
    metastasis = st.selectbox("Distant Metastasis", [
        "Select",
        "No evidence of distant metastasis",
        "Presence of distant metastasis"
    ])

col9, col10 = st.columns(2)

with col9:
    stage = st.selectbox("Clinical Stage", ["Select", "First-Stage", "Second-Stage", "Third-Stage"])

with col10:
    response = st.selectbox("Treatment Response", [
        "Select", "Excellent", "Indeterminate", "Biochemical Incomplete", "Structural Incomplete"
    ])

st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        height: 40px;
        width: 150px;
        border-radius: 10px;
        border: none;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        display: block;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True
)
if st.button("Predict"):
    
    input_dict = {
        "Age": age,
        "Gender": user_gender if user_gender != "Select" else None,
        "Thyroid Function": thyroid_function,
        "Physical Examination": physical_exam,
        "Adenopathy": adenopathy,
        "Types of Thyroid Cancer (Pathology)": pathology,
        "Focality": focality,
        "Risk": risk,
        "Tumor": tumor,
        "Lymph Nodes": nodes,
        "Stage": stage,
        "Treatment Response": response
    }
    
    if validate_inputs(input_dict):
        try:
            with st.spinner("Making prediction..."):
                time.sleep(2)

            input_df = pd.DataFrame([input_dict])

            for col, cats in category_mappings.items():
                if col in input_df.columns:
                     input_df[col] = input_df[col].astype(pd.api.types.CategoricalDtype(categories=cats))
            print(input_df.dtypes)

            prediction = model.predict(input_df)            
            prob = model.predict_proba(input_df)[0][1]
            percent = round(prob * 100, 2)
         
            if prediction[0] == 0:
                st.balloons()
                st.markdown(
                    f"<h3 style='text-align: center; color: green; font-size: 24px;'> {user_name if user_name else 'This Person'} does <b>NOT</b> have a thyroid condition.</h3>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<h3 style='text-align: center; color: red; font-size: 24px;'> {user_name if user_name else 'This Person'} HAS <b>a thyroid condition.</b>.</h3>",
                    unsafe_allow_html=True
                )

            st.markdown(
                f"<p style='text-align: center; font-size: 18px;'>Thyroid Risk Probability: <b>{prob * 100:.2f}%</b></p>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<p style='text-align: center; font-size: 18px;'>Thyroid Status and Health Report</p>",
                unsafe_allow_html=True
            )
   
            advice_list = []

            if physical_exam == "Abnormal":
                physical_exam_comment = "Abnormal findings detected during physical examination."
                physical_exam_class = "error"
                advice_list.append("❗ Physical exam findings may require further investigation.")
            else:
                physical_exam_comment = "✅ Physical examination findings are normal."
                physical_exam_class = "success"

            if adenopathy != "No Lympth Adenopathy":
                adenopathy_comment = f"Adenopathy status: {adenopathy}"
                adenopathy_class = "warning"
                if adenopathy == "Extensive and Widespread":
                    advice_list.append("⚠️ Extensive adenopathy should be evaluated for potential metastasis.")
            else:
                adenopathy_comment = "✅ No lymph node enlargement (adenopathy) detected."
                adenopathy_class = "success"


            st.markdown(f"""
            <style>
            .status-wrap {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 16px;
            margin: 8px 0 2px 0;
            }}

            .status-card {{
            position: relative;
            border-radius: 14px;
            padding: 16px 16px 14px 16px;
            background: rgba(255,255,255,0.7);
            border: 1px solid rgba(99, 102, 241, 0.18);
            box-shadow: 0 6px 24px rgba(17, 17, 26, 0.08);
            backdrop-filter: blur(6px);
            transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
            overflow: hidden;
            }}

            @media (prefers-color-scheme: dark) {{
            .status-card {{
                background: rgba(24,24,32,0.55);
                border-color: rgba(99, 102, 241, 0.25);
                box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35);
            }}
            }}

            .status-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 28px rgba(17, 17, 26, 0.12);
            border-color: rgba(99, 102, 241, 0.38);
            }}

            .status-accent {{
            position: absolute;
            inset: 0;
            pointer-events: none;
            background: linear-gradient(90deg, #00FFF0, #7A5CFA, #FF2DA3, #FF6EC7, #00FFF0);
            background-size: 300% 100%;
            height: 3px;
            top: 0;
            animation: slide 6s linear infinite;
            opacity: 0.85;
            }}

            @keyframes slide {{
            0%   {{ background-position: 0% 50%; }}
            100% {{ background-position: 300% 50%; }}
            }}

            .status-inner {{
            display: flex;
            align-items: center;
            gap: 12px;
            }}

            .status-icon {{
            font-size: 28px;
            line-height: 1;
            filter: drop-shadow(0 1px 1px rgba(0,0,0,0.18));
            }}

            .status-texts {{
            display: flex;
            flex-direction: column;
            gap: 2px;
            }}

            .status-title {{
            font-weight: 700;
            font-size: 13px;
            letter-spacing: .02em;
            text-transform: uppercase;
            color: #5b5f97;
            opacity: .95;
            }}

            .status-value {{
            font-weight: 800;
            font-size: 18px;
            color: #171746;
            }}

            @media (prefers-color-scheme: dark) {{
            .status-title {{ color: #A5B4FC; }}
            .status-value {{ color: #E5E7EB; }}
            }}

            .badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            font-weight: 700;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(99,102,241,0.12);
            border: 1px solid rgba(99,102,241,0.22);
            color: #4338CA;
            }}
            @media (prefers-color-scheme: dark) {{
            .badge {{
                background: rgba(99,102,241,0.18);
                border-color: rgba(99,102,241,0.28);
                color: #C7D2FE;
            }}
            }}
            </style>

            <div class="status-wrap">

            <!-- Cancer Type Card -->
            <div class="status-card">
                <div class="status-accent"></div>
                <div class="status-inner">
                <div class="status-icon">🧬</div>
                <div class="status-texts">
                    <div class="status-title">Detected Cancer Type</div>
                    <div class="status-value">{pathology}</div>
                </div>
                </div>
            </div>

            <!-- Clinical Stage Card -->
            <div class="status-card">
                <div class="status-accent"></div>
                <div class="status-inner">
                <div class="status-icon">📊</div>
                <div class="status-texts">
                    <div class="status-title">Clinical Stage</div>
                    <div class="status-value">{stage}</div>
                </div>
                </div>
            </div>

            </div>
            """, unsafe_allow_html=True)
            
            if focality == "Multifocal":
                focality_comment = "🔬 Multifocality (more than one focus) detected."
                focality_class = "warning"
                advice_list.append("🔬 If multifocality is present, follow-up and aggressive treatment planning is recommended.")
            else:
                focality_comment = "✅ Single focus (unifocal) tumor present."
                focality_class = "success"

            if risk == "High":
                risk_comment = "❗ Urgent intervention is required for high-risk patients."
                risk_class = "error"
                advice_list.append("❗ Rapid and comprehensive intervention is important for high-risk patients.")
            elif risk == "Intermediate":
                risk_comment = "⚠️ Regular follow-up is recommended for intermediate-risk patients."
                risk_class = "warning"
                advice_list.append("⚠️ Regular monitoring and follow-up are important for intermediate risk.")
            else:
                risk_comment = "✅ Current status can be maintained for low-risk patients."
                risk_class = "success"

            if tumor == "tumor that is 1 cm or smaller":
                tumor_comment = "📏 Tumor is small in size (≤1 cm)."
                tumor_class = "success"
                advice_list.append("📏 Small tumors generally have a good prognosis.")
            elif tumor == "tumor larger than 1 cm but not larger than 2 cm":
                tumor_comment = "📏 Tumor is medium-sized (1-2 cm)." 
                tumor_class = "warning"
                advice_list.append("📏 Surgical and follow-up planning is recommended for medium-sized tumors.")

            elif tumor == "tumor larger than 2 cm but not larger than 4 cm":
                tumor_comment = "📏 Tumor is medium-sized (2-4 cm)."
                tumor_class = "warning"
                advice_list.append("📏 Surgical and follow-up planning is recommended for medium-sized tumors.")
            elif tumor == "tumor larger than 4 cm":
                tumor_comment = "📏 Tumor is large (>4 cm)."
                tumor_class = "error"
                advice_list.append("📏 Imaging is recommended for large tumors to evaluate spread.")
            elif tumor in [
                "tumor that has grown outside the thyroid",
                "tumor that has invaded nearby structures",
                "tumor that has grown outside the thyroid and invaded nearby structures"
            ]:
                tumor_comment = "📏 Tumor may have invaded surrounding structures."
                tumor_class = "error"
                advice_list.append("📏 Tumor has spread to surrounding tissues. Urgent specialist evaluation is required.")
            else:
                tumor_comment = "📏 Tumor status unknown."
                tumor_class = "error"
                advice_list.append("📏 Tumor status unknown. Further tests are required.")

            if "no evidence" in nodes:
                nodes_comment = "✅ No lymph node metastasis detected."
                nodes_class = "success"
                advice_list.append("✅ No lymph node involvement indicates low risk of metastasis.")
            elif "Left Side Body Adenopathy" in nodes or "Right Side Body Adenopathy" in nodes:
                nodes_comment = f"⚠️ Lymph node involvement present: {nodes}"
                nodes_class = "warning"
                advice_list.append("⚠️ Lymph node involvement increases metastasis risk. Close follow-up is recommended.")
            elif "Extensive and Widespread" in nodes:
                nodes_comment = "⚠️ Extensive lymph node involvement present."
                nodes_class = "error"
                advice_list.append("❗ Extensive lymph node involvement should be evaluated for metastasis.")
            else:
                nodes_comment = "⚠️ Lymph node status unknown."
                nodes_class = "error"

            if thyroid_function == "Normal":
                thyroid_function_comment = "✅ Thyroid function is normal."
                thyroid_function_class = "success"
            elif thyroid_function == "Hypo":
                thyroid_function_comment = "⚠️ Hypothyroidism detected."
                thyroid_function_class = "warning"
                advice_list.append("⚠️ Endocrinology consultation is advised for hypothyroidism treatment.")
            elif thyroid_function == "Hyper":
                thyroid_function_comment = "⚠️ Hyperthyroidism detected."
                thyroid_function_class = "warning"
                advice_list.append("⚠️ Endocrinology consultation is advised for hyperthyroidism treatment.")
            else:
                thyroid_function_comment = "⚠️ Thyroid function unknown."
                thyroid_function_class = "error"
                advice_list.append("⚠️ Thyroid function unknown. Further tests required.")

            if stage == "First-Stage":
                stage_comment = "✅ Stage One: Tumor is small and localized."
                stage_class = "success"
                advice_list.append("✅ Tumors in Stage One usually have a good prognosis.")
            elif stage == "Second-Stage":
                stage_comment = "⚠️ Stage Two: Tumor is medium-sized and localized."
                stage_class = "warning"
                advice_list.append("⚠️ Surgical and follow-up planning recommended for Stage Two tumors.")
            elif stage == "Third-Stage":
                stage_comment = "❗ Stage Three: Tumor is large or has spread to surrounding tissues."
                stage_class = "error"
                advice_list.append("❗ Urgent referral to oncology and surgical teams is required for Stage Three tumors.")
            else:
                stage_comment = "❗ Stage unknown."
                stage_class = "error"
                advice_list.append("❗ Stage unknown. Further tests required.")

            if "no evidence" in metastasis:
                metastasis_comment = "✅ No distant metastasis detected."
                metastasis_class = "success"
                advice_list.append("✅ Prognosis is better if no distant metastasis is present.")
            elif "the presence of distant metastasis" in metastasis:
                metastasis_comment = "❗ Distant metastasis detected."
                metastasis_class = "error"
                advice_list.append("❗ Presence of distant metastasis indicates advanced cancer. Oncology consultation recommended.")
            else:
                metastasis_comment = "❗ Distant metastasis status unknown."
                metastasis_class = "error"
                advice_list.append("❗ Distant metastasis status unknown. Further tests required.")

            if response == "Excellent":
                response_comment = "🎯 Complete response to treatment."
                response_class = "success"
                advice_list.append("🎯 Complete response to treatment. Continue to maintain your current status.")
            elif response == "Biochemical Incomplete":
                response_comment = "🔄 Partial response to treatment."
                response_class = "warning"
                advice_list.append("🔄 Partial treatment response; regular monitoring is important.")    
            elif response == "Structural Incomplete":
                response_comment = "🚫 Insufficient response to treatment."
                response_class = "error"
                advice_list.append("🚫 Insufficient treatment response. Alternative methods should be considered.")
            elif response == "Indeterminate":
                response_comment = "Treatment response is unclear."
                response_class = "warning"
                advice_list.append("Unclear treatment response; regular monitoring is important.")
            else:
                st.error(f"🚫 Treatment response: {response}")
                advice_list.append("🚫 Insufficient treatment response. Alternative methods should be considered.")


            advice_html = "".join(f"<li>{item}</li>" for item in advice_list)

            report_html = f"""
            <html>
            <head>
            <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                padding: 20px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                text-align: center;
            }}
            h2 {{
                color: #2E86C1;
            }}
            h3 {{
                color: {'green' if prediction[0] == 0 else 'red'};
            }}
            .section {{
                    margin-bottom: 20px;
                    width: 100%;
                    max-width: 600px;
                }}
            .section ul {{
                    list-style-position: inside;  
                    padding: 0;
                    margin: 0 auto;
                    display: inline-block;    
                    text-align: left;         
            }}
            .success {{ color: green; }}
            .warning {{ color: orange; }}
            .error {{ color: red; }}
            ul {{
                list-style-type: none;
                padding: 0;
                text-align: left;
            }}
            li {{
                margin-bottom: 10px;
            }}
            </style>
            </head>
            <body>
            <h2>Thyroid Cancer Prediction Report</h2>
                <h3>{'✅ No Thyroid Cancer ✅' if prediction[0] == 0 else '⚠️ Thyroid Cancer Detected ⚠️'}</h3>

            <div class="section">
                    {f'<p style="margin:0; padding:0;"><b>Name:</b> {user_name}</p>' if user_name else ""}
                    <b>Age:</b> {age} <br>
                    <b>Gender:</b> {user_gender}
            </div>
            <h3>Health Status</h3>
            <ul> 
                <li><span class="{physical_exam_class}">{physical_exam_comment}</span></li>
                <li><span class="{adenopathy_class}">{adenopathy_comment}</span></li>
                <li><span class="{focality_class}">{focality_comment}</span></li>
                <li><span class="{risk_class}">{risk_comment}</span></li>
                <li><span class="{tumor_class}">{tumor_comment}</span></li>
                <li><span class="{nodes_class}">{nodes_comment}</span></li>
                <li><span class="{thyroid_function_class}">{thyroid_function_comment}</span></li>
                <li><span class="{stage_class}">{stage_comment}</span></li>
                <li><span class="{metastasis_class}">{metastasis_comment}</span></li>
                <li><span class="{response_class}">{response_comment}</span></li>
            </ul>                   
                
            <div class="section">
                    <h3 style="margin:0; padding:0;">Personalized Recommendations for {user_name}</h3>
                    <ul>
             {advice_html}
                    </ul>
            </div>

            <h3>General Health Advice</h3>
            <ul>
                <li>Pay attention to a healthy and balanced diet.</li>
                <li>Don't forget to drink enough water.</li>
                <li>Try to avoid stress.</li>
                <li>Make sure you get enough sleep.</li>
                <li>Don't neglect regular exercise.</li>
                <li>Limit smoking and alcohol consumption.</li>
                <li>Don't skip your annual health check-ups.</li>
                <li>Follow your doctor's recommendations.</li>
                <li>Have regular check-ups to protect your thyroid health.</li>
                <li>Practice stress management techniques.</li>
                <li>Ensure adequate iodine intake.</li>
                <li>Do not use medications that may affect your thyroid health without consulting your doctor.</li>
            </ul>
            </body>
            </html>
            """
            b64_report = base64.b64encode(report_html.encode()).decode()
            href = f'data:text/html;base64,{b64_report}'

            disease_type = "Thyroid Cancer"

            if "all_reports" not in st.session_state:
                st.session_state["all_reports"] = {}
            if disease_type not in st.session_state["all_reports"]:
                st.session_state["all_reports"][disease_type] = []

            st.session_state["all_reports"][disease_type].append({
                "user_name": user_name or "Unknown User",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "html": report_html
            })

            safe_user_name = user_name.lower().replace(" ", "_") if user_name else "user"
            file_name = f"{safe_user_name}_thyroid_report.html"
            b64_report = base64.b64encode(report_html.encode()).decode()
            href = f'data:text/html;base64,{b64_report}'

            st.markdown(f"""
                <div style="text-align: center; margin-top: 40px; animation: fadeIn 3s;">
                    <a href="{href}" download="{file_name}" style="
                        background: linear-gradient(90deg, #8E24AA, #CE93D8);
                        color: white;
                        padding: 14px 30px;
                        font-size: 18px;
                        font-weight: bold;
                        border-radius: 12px;
                        text-decoration: none;
                        box-shadow: 3px 3px 10px rgba(0,0,0,0.2);
                        transition: 0.3s;
                        display: inline-block;
                    ">
                    Thyroid Risk Report 📥
                    </a>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <style>
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            </style>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
