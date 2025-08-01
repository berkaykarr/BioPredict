import base64
from datetime import datetime
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import altair as alt

@st.cache_resource
def load_model_and_scalers():
    model = joblib.load('alzheimers_model.pkl') 
    scalers = None
    if os.path.exists('scalers.joblib'):
        try:
            scalers = joblib.load('scalers.joblib')  
        except Exception:
            scalers = None
    return model, scalers

model, scalers = load_model_and_scalers()

user_name = st.session_state.get("user_name", "").strip()

user_gender = st.session_state.get("user_gender", "Select")


GENDER_LABELS_UI = ["Select", "Female", "Male"]
GENDER_TO_CODE = {"Female": 0, "Male": 1}
CODE_TO_GENDER = {0: "Female", 1: "Male"}

user_gender_label = st.session_state.get("user_gender", "Select")
gender_code_for_model = GENDER_TO_CODE.get(user_gender_label)      

YESNO_LABELS = ["No", "Yes"]
YESNO_TO_CODE = {"No": 0, "Yes": 1}
CODE_TO_LABEL = {0: "No", 1: "Yes"}

ETHNICITY_LABELS_UI = ["Select", "Caucasian", "African American", "Asian", "Other"]
ETHNICITY_TO_CODE = {"Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3}
CODE_TO_ETHNICITY = {0: "Caucasian", 1: "African American", 2: "Asian", 3: "Other"}

EDU_LABELS_UI = ["Select", "None", "High School", "Bachelor's", "Higher"]
EDU_TO_CODE = {"None": 0, "High School": 1, "Bachelor's": 2, "Higher": 3}
CODE_TO_EDU = {0: "None", 1: "High School", 2: "Bachelor's", 3: "Higher"}

ethnicity_label = st.session_state.get("user_ethnicity", "Select")
edu_label = st.session_state.get("user_education", "Select")

def validate_inputs(gender_code, ethnicity, education):
    if gender_code is None:
        st.warning("Please select your gender on the main page.", icon="⚠️")
        return False
    if ethnicity is None or education is None:
        st.warning("Please fill in all the selection boxes!", icon="⚠️")
        return False
    return True

can_submit = gender_code_for_model is not None


RANGE_HINTS = {
    "Age":        {"min_value": 60, "max_value": 90, "value": 65, "step": 1},
    "BMI":        {"min_value": 14.0, "max_value": 40.0, "value": 25.0, "step": 0.1},
    "AlcoholConsumption": {"min_value": 0.0, "max_value": 25.0, "value": 8.0, "step": 0.1},
    "PhysicalActivity":   {"min_value": 0.0, "max_value": 12.0, "value": 7.0, "step": 0.1},
    "DietQuality":        {"min_value": 0.0, "max_value": 10.0, "value": 5.0, "step": 0.1},
    "SystolicBP":  {"min_value": 90,  "max_value": 200, "value": 120, "step": 1},
    "DiastolicBP": {"min_value": 50,  "max_value": 120, "value": 80,  "step": 1},
    "CholesterolTotal":        {"min_value": 100, "max_value": 400, "value": 220, "step": 1},
    "CholesterolLDL":          {"min_value": 40,  "max_value": 250, "value": 130, "step": 1},
    "CholesterolHDL":          {"min_value": 20,  "max_value": 120, "value": 50,  "step": 1},
    "CholesterolTriglycerides":{"min_value": 50,  "max_value": 500, "value": 150, "step": 1},
    "MMSE":                {"min_value": 0.0,  "max_value": 30.0, "value": 26.0, "step": 0.5},
    "FunctionalAssessment":{"min_value": 0.0,  "max_value": 10.0, "value": 6.0,  "step": 0.1},
    "ADL":                 {"min_value": 0.0,  "max_value": 10.0, "value": 7.0,  "step": 0.1},
}


NUMERIC_COLUMNS_SCALED = [
    'Age','BMI','AlcoholConsumption','PhysicalActivity','DietQuality',
    'SystolicBP','DiastolicBP','CholesterolTotal','CholesterolLDL',
    'CholesterolHDL','CholesterolTriglycerides','MMSE',
    'FunctionalAssessment','ADL'
]

INPUT_COLUMNS = [
    'Age','Gender','Ethnicity','EducationLevel','BMI','Smoking',
    'AlcoholConsumption','PhysicalActivity','DietQuality','CardiovascularDisease',
    'Depression','SystolicBP','DiastolicBP','CholesterolTotal','CholesterolLDL',
    'CholesterolHDL','CholesterolTriglycerides','MMSE','FunctionalAssessment',
    'MemoryComplaints','BehavioralProblems','ADL'
]

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
        margin-bottom: 0.1px;
        font-family: 'Arial Black', sans-serif;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.35);
    ">
    Alzheimer's Risk Prediction
    </h1>
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
        <div>Alzheimer's disease is a progressive brain disorder that affects memory and thinking skills.</div>
        <div>Early diagnosis and supportive care can improve quality of life for patients and families.</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption("This tool is for educational purposes only and does not provide a medical diagnosis. Results should be interpreted in conjunction with clinical evaluation.")



with st.expander("Personal Informations", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Age", **RANGE_HINTS["Age"])
    with col2:
        if can_submit:
            _ = st.selectbox(
                "Gender",
                options=GENDER_LABELS_UI,
                index=GENDER_LABELS_UI.index(user_gender_label),
                disabled=True,
                help="Gender selected on Home Page."
            )
        else:
            _ = st.selectbox(
                "Gender",
                options=GENDER_LABELS_UI,
                index=0,
                disabled=True,
                help="Please select your Gender on the home page."  )
    with col3:
        _ = st.selectbox(
            "Ethnicity",
            options=ETHNICITY_LABELS_UI,
            index=ETHNICITY_LABELS_UI.index(ethnicity_label) if ethnicity_label in ETHNICITY_LABELS_UI else 0,
            key="user_ethnicity",
            help="Please select your Ethnicity."
        )
        ethnicity_label = st.session_state["user_ethnicity"]
        can_submit_ethnicity = ethnicity_label != "Select"
        ethnicity = ETHNICITY_TO_CODE.get(ethnicity_label) if can_submit_ethnicity else None

    
    with col4:
        _ = st.selectbox(
            "Education Level",
            options=EDU_LABELS_UI,
            index=EDU_LABELS_UI.index(edu_label) if edu_label in EDU_LABELS_UI else 0,
            key="user_education",
            help="Please select your education level."
        )
        edu_label = st.session_state["user_education"]
        can_submit_edu = edu_label != "Select"
        edu = EDU_TO_CODE.get(edu_label) if can_submit_edu else None


with st.expander("Life Style", expanded=True):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        bmi = st.number_input("BMI", **RANGE_HINTS["BMI"])
    with col2:
        smoking_label = st.radio("Smoking", options=YESNO_LABELS, horizontal=True, index=0)
        smoking = YESNO_TO_CODE[smoking_label]  
    with col3:
        alcohol = st.number_input("AlcoholConsumption", **RANGE_HINTS["AlcoholConsumption"])
    with col4:
        activity = st.number_input("PhysicalActivity", **RANGE_HINTS["PhysicalActivity"])
    with col5:
        diet = st.number_input("DietQuality", **RANGE_HINTS["DietQuality"])

with st.expander("Clinical History", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cvd_label = st.radio("CardiovascularDisease", options=YESNO_LABELS, horizontal=True, index=0)
        cvd = YESNO_TO_CODE[cvd_label]
    with col2:
        dep_label = st.radio("Depression", options=YESNO_LABELS, horizontal=True, index=0)
        dep = YESNO_TO_CODE[dep_label]
    with col3:
        memc_label = st.radio("MemoryComplaints", options=YESNO_LABELS, horizontal=True, index=0)
        memc = YESNO_TO_CODE[memc_label]
    with col4:
        beh_label = st.radio("BehavioralProblems", options=YESNO_LABELS, horizontal=True, index=0)
        beh = YESNO_TO_CODE[beh_label]


with st.expander("Vital & Lipid", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        sbp = st.number_input("SystolicBP", **RANGE_HINTS["SystolicBP"])
    with col2:
        dbp = st.number_input("DiastolicBP", **RANGE_HINTS["DiastolicBP"])
    with col3:
        chol_total = st.number_input("CholesterolTotal", **RANGE_HINTS["CholesterolTotal"])

    col4, col5, col6 = st.columns(3)
    with col4:
        chol_ldl = st.number_input("CholesterolLDL", **RANGE_HINTS["CholesterolLDL"])
    with col5:
        chol_hdl = st.number_input("CholesterolHDL", **RANGE_HINTS["CholesterolHDL"])
    with col6:
        chol_tg = st.number_input("CholesterolTriglycerides", **RANGE_HINTS["CholesterolTriglycerides"])

with st.expander("Cognitive &  Functional", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        mmse = st.number_input("MMSE", **RANGE_HINTS["MMSE"])
    with col2:
        func = st.number_input("FunctionalAssessment", **RANGE_HINTS["FunctionalAssessment"])
    with col3:
        adl = st.number_input("ADL", **RANGE_HINTS["ADL"])

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
can_submit_all = can_submit and can_submit_ethnicity and can_submit_edu

if st.button("Predict"):
    if validate_inputs(gender_code_for_model, ethnicity, edu):

        try:
            with st.spinner("Making prediction..."):
                time.sleep(1.2)

            gender_for_model = gender_code_for_model

            X_user = pd.DataFrame([{
                'Age': age,
                'Gender': gender_for_model,  
                'Ethnicity': ethnicity,
                'EducationLevel': edu,
                'BMI': bmi,
                'Smoking': smoking,
                'AlcoholConsumption': alcohol,
                'PhysicalActivity': activity,
                'DietQuality': diet,
                'CardiovascularDisease': cvd,
                'Depression': dep,
                'SystolicBP': sbp,
                'DiastolicBP': dbp,
                'CholesterolTotal': chol_total,
                'CholesterolLDL': chol_ldl,
                'CholesterolHDL': chol_hdl,
                'CholesterolTriglycerides': chol_tg,
                'MMSE': mmse,
                'FunctionalAssessment': func,
                'MemoryComplaints': memc,
                'BehavioralProblems': beh,
                'ADL': adl
            }], columns=INPUT_COLUMNS)

            if scalers is not None:
                cols = scalers.get('columns', NUMERIC_COLUMNS_SCALED)
                if 'minmax' in scalers:
                    X_user[cols] = scalers['minmax'].transform(X_user[cols])
                elif 'standard' in scalers:
                    X_user[cols] = scalers['standard'].transform(X_user[cols])


            prediction = model.predict(X_user)
            probabilities = model.predict_proba(X_user)
            risk_prob = float(probabilities[0][1])
            risk_pct = int(round(risk_prob * 100))

            if prediction[0] == 0:
                st.balloons()
                st.markdown(
                    f"<h3 style='text-align: center; color: green; font-size: 24px;'>  {user_name if user_name else 'This person'} is <b>NOT</b> at <b>risk</b> of Alzheimer's Disease.</h3>",
                    unsafe_allow_html=True
                )
                risk_text = "Alzheimer risk is LOW"
                color = "green"
            else:
                st.markdown(
                    f"<h3 style='text-align: center; color: red; font-size: 24px;'>  {user_name if user_name else 'This person'} IS at <b>RISK</b> of Alzheimer's Disease!</h3>",
                    unsafe_allow_html=True
                )
                risk_text = "⚠️ Alzheimer risk is HIGH"
                color = "red"   

            st.markdown(
                f"<p style='text-align: center; font-size: 18px;'>Alzheimer probability: <b>{risk_prob * 100:.2f}%</b></p>",
                unsafe_allow_html=True
            )

            st.markdown("""
            <div style='text-align: center; margin-bottom: 30px;'>
                <div style="display: inline-block; margin-right: 30px;">
                    <div style="width: 20px; height: 20px; background-color: #4CAF50; display: inline-block; border-radius: 4px; margin-right: 8px;"></div>
                    <span style="font-size: 17px; font-weight: bold;">Low Risk</span>
                </div>
                <div style="display: inline-block; margin-right: 30px;">
                    <div style="width: 20px; height: 20px; background-color: #FFC107; display: inline-block; border-radius: 4px; margin-right: 8px;"></div>
                    <span style="font-size: 17px; font-weight: bold;">Moderate Risk</span>
                </div>
                <div style="display: inline-block;">
                    <div style="width: 20px; height: 20px; background-color: #F44336; display: inline-block; border-radius: 4px; margin-right: 8px;"></div>
                    <span style="font-size: 17px; font-weight: bold;">High Risk</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            normal_ranges = {
                'Systolic BP': '90–120 mmHg',
                'Diastolic BP': '60–80 mmHg',
                'Total Cholesterol': '<200 mg/dL',
                'HDL Cholesterol': '>40 mg/dL',
                'LDL Cholesterol': '<130 mg/dL',
                'Triglycerides': '<150 mg/dL',
                'MMSE Score': '24–30 (Normal)',
                'Functional Score': '0–3 (Much Better)',
                'ADL Score': '≥5 (Independent)',
                            }

            raw_data_dict = {
                'Systolic BP': sbp,
                'Diastolic BP': dbp,
                'Total Cholesterol': chol_total,
                'HDL Cholesterol': chol_hdl,
                'LDL Cholesterol': chol_ldl,
                'Triglycerides': chol_tg,
                'MMSE Score': mmse,
                'Functional Score': func,
                'ADL Score': adl
            }
            raw_df = pd.DataFrame(raw_data_dict.items(), columns=['Feature', 'Value'])

            def get_alzheimer_color(feature, value):
                if feature == 'Systolic BP':
                    if 90 <= value <= 120:
                        return 'green'
                    elif 121 <= value <= 139 or 80 <= value < 90:
                        return 'yellow'
                    else:
                        return 'red'
                elif feature == 'Diastolic BP':
                    if 60 <= value <= 80:
                        return 'green'
                    elif 81 <= value <= 89 or 50 <= value < 60:
                        return 'yellow'
                    else:
                        return 'red'
                elif feature == 'Total Cholesterol':
                    if value < 200:
                        return 'green'
                    elif 200 <= value <= 239:
                        return 'yellow'
                    else:
                        return 'red'
                elif feature == 'HDL Cholesterol':
                    if value > 60:
                        return 'green'
                    elif 40 <= value <= 60:
                        return 'yellow'
                    else:
                        return 'red'
                elif feature == 'LDL Cholesterol':
                    if value < 100:
                        return 'green'
                    elif 100 <= value <= 129:
                        return 'yellow'
                    else:
                        return 'red'
                elif feature == 'Triglycerides':
                    if value < 150:
                        return 'green'
                    elif 150 <= value <= 199:
                        return 'yellow'
                    else:
                        return 'red'
                elif feature == 'MMSE Score':
                    if value >= 28:
                        return 'green'
                    elif 24 <= value < 28:
                        return 'yellow'
                    else:
                        return 'red'
                elif feature == 'Functional Score':
                    if value <= 3:
                        return 'green'
                    elif 4 <= value <= 5:
                        return 'yellow'
                    else:
                        return 'red'
                elif feature == 'ADL Score':
                    if value >= 6:
                        return 'green'
                    elif 4 <= value < 6:
                        return 'yellow'
                    else:
                        return 'red'
                else:
                    return 'gray'


            raw_df['Color'] = raw_df.apply(lambda row: get_alzheimer_color(row['Feature'], row['Value']), axis=1)
            raw_df['Normal Range'] = raw_df['Feature'].map(normal_ranges)

            max_y = raw_df['Value'].max() * 1.2

            base_chart = alt.Chart(raw_df).mark_bar().encode(
                x=alt.X('Feature:N', sort='-y'),
                y=alt.Y('Value:Q', scale=alt.Scale(domain=[0, max_y])),
                color=alt.Color('Color:N', scale=None),
                tooltip=['Feature', 'Value', 'Normal Range']
            )

            text = base_chart.mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                color='black'
            ).encode(
                text=alt.Text('Value', format='.1f')
            )

            normal_text = alt.Chart(raw_df).mark_text(
                align='center',
                baseline='top',
                dy=10,
                fontSize=11,
                color='#666'
            ).encode(
                x='Feature:N',
                y=alt.value(0),
                text='Normal Range:N'
            )

            final_chart = base_chart + text + normal_text
            st.altair_chart(final_chart, use_container_width=True)

            advice_list = []
         
            risk_text   = "Low risk"
            risk_class  = "success"

            age_comment = "Age is within expected range."
            age_class   = "success"

            mmse_comment = "MMSE is within the normal range."
            mmse_class   = "success"

            dep_comment = "No depression reported."
            dep_class   = "success"

            chol_total_comment = "Total cholesterol is within target."
            chol_total_class   = "success"

            chol_hdl_comment = "HDL level is adequate."
            chol_hdl_class   = "success"

            chol_ldl_comment = "LDL level is within target."
            chol_ldl_class   = "success"

            chol_tg_comment = "Triglycerides are within target."
            chol_tg_class   = "success"

            bp_comment = "Blood pressure is within target."
            bp_class   = "success"

            activity_comment = "Physical activity is adequate."
            activity_class   = "success"

            diet_comment = "Diet quality is adequate."
            diet_class   = "success"

            memc_comment = "No memory complaints."
            memc_class   = "success"

            beh_comment = "No behavioral problems."
            beh_class   = "success"

            func_comment = "Functional assessment is acceptable."
            func_class   = "success"

            adl_comment = "ADL score is acceptable."
            adl_class   = "success"

            if risk_prob >= 0.7:
                risk_text  = "High risk"
                risk_class = "error"
                advice_list.append("High risk. Please consult a neurologist soon.")
            elif risk_prob >= 0.4:
                risk_text  = "Moderate risk"
                risk_class = "warning"
                advice_list.append("Moderate risk. Keep regular check-ups.")
            else:
                risk_text  = "Low risk"
                risk_class = "success"
                advice_list.append("Low risk. Maintain your healthy lifestyle.")

            if age > 65:
                age_comment = "Age is a significant risk factor."
                age_class   = "warning"
                advice_list.append("Age increases risk. Consider periodic memory assessments.")

            if mmse < 24:
                mmse_comment = "MMSE is low."
                mmse_class   = "error"
                advice_list.append("Low MMSE. Neuropsychological assessment is recommended.")

            if dep == 1:
                dep_comment = "History of depression may increase risk."
                dep_class   = "warning"
                advice_list.append("Consider mental health support for depression.")

            if chol_total > 240:
                chol_total_comment = "Total cholesterol is high."
                chol_total_class   = "error"
                advice_list.append("High total cholesterol. Improve cardio‑metabolic health.")
            if chol_hdl < 40:
                chol_hdl_comment = "HDL is low."
                chol_hdl_class   = "error"
                advice_list.append("Low HDL. Consider diet and exercise.")
            if chol_ldl > 130:
                chol_ldl_comment = "LDL is high."
                chol_ldl_class   = "error"
                advice_list.append("High LDL. Dietary changes may help.")
            if chol_tg > 150:
                chol_tg_comment = "Triglycerides are high."
                chol_tg_class   = "error"
                advice_list.append("High triglycerides. Consider diet and exercise.")

            if sbp > 140 or dbp > 90:
                bp_comment = "Blood pressure is high."
                bp_class   = "error"
                advice_list.append("High BP can affect cognitive health. Monitor and adjust lifestyle.")

            if activity < 3:
                activity_comment = "Physical activity is low."
                activity_class   = "warning"
                advice_list.append("Aim for ≥150 min/week of moderate exercise.")
            if diet < 5:
                diet_comment = "Diet quality is low."
                diet_class   = "warning"
                advice_list.append("Consider a Mediterranean‑style diet.")

            if memc == 1:
                memc_comment = "Memory complaints reported."
                memc_class   = "warning"
                advice_list.append("Memory complaints present. Neurologic evaluation recommended.")
            if beh == 1:
                beh_comment = "Behavioral problems reported."
                beh_class   = "warning"
                advice_list.append("Behavioral issues present. Consider psychological support.")

            if func < 4:
                func_comment = "Functional assessment is low."
                func_class   = "warning"
                advice_list.append("Consider support for daily activities.")
            if adl < 5:
                adl_comment = "ADL score is low."
                adl_class   = "warning"
                advice_list.append("You may need assistance with daily living activities.")

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
            <h2>Alzheimer Risk Report</h2>
            <h3>{'✅ No Alzheimer ✅' if prediction[0]==0 else '⚠️ Alzheimer Detected ⚠️'}</h3>

            <div class="section">
                    {f'<p style="margin:0; padding:0;"><b>Ad:</b> {user_name}</p>' if user_name else ""}
                    <b>Age:</b> {age} <br>
                    <b>Gender:</b> {user_gender}
            </div>
                    <h3>Health Status</h3>,
            <ul> 
                <li><span class="{risk_class}">{risk_text}</span></li>
                <li><span class="{age_class}">{age_comment}</span></li>
                <li><span class="{mmse_class}">{mmse_comment}</span></li>
                <li><span class="{dep_class}">{dep_comment}</span></li>
                <li><span class="{chol_total_class}">{chol_total_comment}</span></li>
                <li><span class="{chol_hdl_class}">{chol_hdl_comment}</span></li>
                <li><span class="{chol_ldl_class}">{chol_ldl_comment}</span></li>
                <li><span class="{chol_tg_class}">{chol_tg_comment}</span></li>
                <li><span class="{bp_class}">{bp_comment}</span></li>
                <li><span class="{activity_class}">{activity_comment}</span></li>
                <li><span class="{diet_class}">{diet_comment}</span></li>
                <li><span class="{memc_class}">{memc_comment}</span></li>
                <li><span class="{beh_class}">{beh_comment}</span></li>
                <li><span class="{func_class}">{func_comment}</span></li>
                <li><span class="{adl_class}">{adl_comment}</span></li>
            </ul>
            
            <div class="section">
                    <h3 style="margin:0; padding:0;">Personalized Recommendations for {user_name}</h3>
                    <ul>
             {advice_html}
                    </ul>
            </div>

            <h3>General Health Recommendations</h3>
            <ul>
                <li>Do at least 150 minutes of moderate exercise per week.</li>
                <li>Adopt healthy eating habits, such as the Mediterranean diet.</li>
                <li>Have regular memory and cognitive function tests.</li>
                <li>Keep your blood pressure and cholesterol under control.</li>
                <li>Seek professional support for depression and anxiety.</li>
                <li>Engage in social activities and mental exercises.</li>
                <li>Get assistance if you need help with daily living activities.</li>
                <li>Have regular check-ups to detect early signs of Alzheimer’s.</li>
            </ul>
            </body>
            </html>

            """

            b64_report = base64.b64encode(report_html.encode()).decode()
            href = f'data:text/html;base64,{b64_report}'

            disease_type = "Alzheimer"

            if "all_reports" not in st.session_state:
                st.session_state["all_reports"] = {}
            if disease_type not in st.session_state["all_reports"]:
                st.session_state["all_reports"][disease_type] = []

            st.session_state["all_reports"][disease_type].append({
                "user_name": user_name or "Bilinmeyen",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "html": report_html
            })

            safe_user_name = user_name.lower().replace(" ", "_") if user_name else "user"
            file_name = f"{safe_user_name}_alzheimer_report.html"
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
                    Alzheimer Risk Report 📥
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
            st.error(f"❌ An error occurred: {e}")
