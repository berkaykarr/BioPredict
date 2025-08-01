import base64
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import altair as alt
from textwrap import dedent

@st.cache_resource
def load_diabetes_model():
    model = joblib.load("diabetes_classifier.pkl")
    scaler = joblib.load("diabetes_scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_diabetes_model()

user_name = st.session_state.get("user_name", "").strip()
user_gender = st.session_state.get("user_gender", "Select")
sex = user_gender if user_gender not in ["Select", "", None] else None  

def validate_inputs(inputs, user_name, sex):
    for key, val in inputs.items():
        if val is None:
            st.warning(f"Please enter a valid value for '{key}'!", icon="⚠️")
            return False
        if isinstance(val, (int, float)) and val < 0:
            st.warning(f"'{key}' must be a positive value!", icon="⚠️")
            return False
    if not user_name:
        st.warning("Please enter your name.", icon="⚠️")
        return False
    if sex is None:
        st.warning("Please select your gender on the main page.", icon="⚠️")
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
        margin-bottom: 0.1px;
        font-family: 'Arial Black', sans-serif;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.35);
    ">
    Diabetes Risk Prediction
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
        <div>Diabetes is a chronic condition that affects how your body processes blood sugar.</div>
        <div>Managing blood glucose levels is essential to prevent long-term complications.</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption("This tool is for educational purposes only and does not provide a medical diagnosis. Results should be interpreted in conjunction with clinical evaluation.")


st.markdown("<h2 style='font-weight:600; font-size:23px;'>Personal Informations</h2>", unsafe_allow_html=True)

with st.expander("", expanded=True):
    has_name = bool(user_name)
    has_gender = user_gender not in ["Select", "", None]

    name_text = user_name if has_name else "Name Not Specified"

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
      position: relative; border-radius: 16px; padding: 14px 16px; margin: 4px 0 8px 0;
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
      display:flex; justify-content:center; align-items:center; width:100%;
    }}
    .info-col {{
      display:flex; flex-wrap:wrap; justify-content:center; align-items:center;
      gap:12px; width:100%;
    }}

    .chip {{
      display:inline-flex; align-items:center; gap:8px; padding:8px 12px; border-radius:999px;
      font-weight:700; font-size:13px; border:1px solid rgba(99,102,241,0.22);
      background: rgba(224,236,255,0.35); color:#171746; white-space:nowrap;
    }}
    @media (prefers-color-scheme: dark) {{
      .chip {{ color:#E5E7EB; background:rgba(99,102,241,0.12); border-color:rgba(99,102,241,0.28); }}
    }}
    .label {{
      font-size:12px; font-weight:800; letter-spacing:.02em; text-transform:uppercase;
      opacity:.75; color:#5b5f97; margin-right:6px;
    }}
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


    Age = st.slider("Age", 1, 120)
    age = Age  

    if g in ["female", "kadın", "kadin", "f", "woman"]:
        pregnancies = st.slider("Pregnancies", 0, 20, 0)
    else:
        st.markdown(
            "<p style='color: darkred; font-weight: bold; text-align: center;'>"
            "Number of pregnancies is only applicable for Females."
            "</p>",
            unsafe_allow_html=True
        )
        pregnancies = 0


st.markdown("<h2 style='font-weight:600; font-size:23px;'>Medical Findings and Tests</h2>", unsafe_allow_html=True)

with st.expander("", expanded=True):

    col1, col2 = st.columns(2)
    with col1:

        glucose = st.slider("Glucose (Blood Glucose mg/dL)", 40, 300, 0)
        blood_pressure = st.slider("Blood Pressure (mmHg)", 40, 200, 0)
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 0)
    with col2:
        insulin = st.slider("Insulin (µU/mL)", 0, 900, 0)
        bmi = st.slider("BMI (Body Mass Index)", 10.0, 70.0, 0.0)
        dpf = st.slider("Diabetes Pedigree Function (Family History)", 0.0, 2.5, 0.0, step=0.01)

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

def get_color(feature, value):
    if feature == "Pregnancies":
        return "#4CAF50" if value == 0 else "#FFC107" if value <= 5 else "#F44336"
    elif feature == "Glucose":
        return "#4CAF50" if value < 140 else "#FFC107" if value < 200 else "#F44336"
    elif feature == "BloodPressure":
        return "#F44336" if value < 80 else "#4CAF50" if value <= 120 else "#FFC107" if value <= 139 else "#F44336"
    elif feature == "SkinThickness":
        return "#FFC107" if value < 20 else "#4CAF50" if value <= 30 else "#FFC107"
    elif feature == "Insulin":
        return "#F44336" if value < 16 else "#4CAF50" if value <= 166 else "#FFC107"
    elif feature == "BMI":
        return "#F44336" if value < 18.5 else "#4CAF50" if value < 25 else "#FFC107" if value < 30 else "#F44336"
    elif feature == "DiabetesPedigreeFunction":
        return "#4CAF50" if value < 0.5 else "#FFC107" if value < 1.0 else "#F44336"
    else:
        return "#9E9E9E"
    

if st.button("Predict"):

    diabetes_inputs = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': Age
    }
    if validate_inputs(diabetes_inputs,user_name,sex):
        try:
            with st.spinner("Making prediction..."):
                time.sleep(2)  
            
            input_data = pd.DataFrame([[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                dpf,
                Age
            ]], columns=[
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
            ])

            features_to_scale = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            input_data[features_to_scale] = scaler.transform(input_data[features_to_scale])

            prediction = model.predict(input_data)
            probabilities = model.predict_proba(input_data)
            risk_prob = probabilities[0][1]

            if prediction[0] == 0:
                st.balloons()
                st.markdown(
                    f"<h3 style='text-align: center; color: green; font-size: 24px;'>{user_name if user_name else 'This Person'} is <b>NOT</b> Diabetic.</h3>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<h3 style='text-align: center; color: red; font-size: 24px;'>{user_name if user_name else 'This Person'} IS <b>Diabetic</b>.</h3>",
                    unsafe_allow_html=True
                )

            st.markdown(
                f"<p style='text-align: center; font-size: 18px;'>Diabetes risk probability: <b>{risk_prob * 100:.2f}%</b></p>",
                unsafe_allow_html=True
            )

            raw_data_dict = {
                'Glucose (mg/dL)': glucose,
                'Blood Pressure (mmHg)': blood_pressure,
                'BMI': bmi,
                'Insulin (µU/mL)': insulin
            }
            raw_df = pd.DataFrame(raw_data_dict.items(), columns=['Feature', 'Value'])

            def get_color_diabetes(feature, value):
                if feature == "Glucose (mg/dL)":
                    return "#4CAF50" if value < 140 else "#FFC107" if value < 200 else "#F44336"
                elif feature == "Blood Pressure (mmHg)":
                    return "#4CAF50" if value < 120 else "#FFC107" if value < 140 else "#F44336"
                elif feature == "BMI":
                    return "#4CAF50" if 18.5 <= value < 25 else "#FFC107" if 25 <= value < 30 else "#F44336"
                elif feature == "Insulin (µU/mL)":
                    return "#4CAF50" if 16 <= value <= 166 else "#FFC107" if value < 16 or (166 < value <= 300) else "#F44336"
                return "#9E9E9E"

            raw_df['Color'] = raw_df.apply(lambda row: get_color_diabetes(row['Feature'], row['Value']), axis=1)

            st.markdown("""
                <div style='text-align: center; margin-bottom: 30px;'>
                    <div style="display: inline-block; margin-right: 30px;">
                        <div style="width: 20px; height: 20px; background-color: #4CAF50; display: inline-block; border-radius: 4px; margin-right: 8px; vertical-align: top;"></div>
                        <span style="font-size: 17px; font-weight: bold; vertical-align: top; line-height: 20px;">Safe (Low Risk)</span>
                    </div>
                    <div style="display: inline-block; margin-right: 30px;">
                        <div style="width: 20px; height: 20px; background-color: #FFC107; display: inline-block; border-radius: 4px; margin-right: 8px; vertical-align: top;"></div>
                        <span style="font-size: 17px; font-weight: bold; vertical-align: top; line-height: 20px;">Moderate Risk</span>
                    </div>
                    <div style="display: inline-block;">
                        <div style="width: 20px; height: 20px; background-color: #F44336; display: inline-block; border-radius: 4px; margin-right: 8px; vertical-align: top;"></div>
                        <span style="font-size: 17px; font-weight: bold; vertical-align: top; line-height: 20px;">High Risk</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)


            normal_ranges = {
                "Glucose (mg/dL)": "Normal: <140",
                "Blood Pressure (mmHg)": "Normal: <120",
                "BMI": "Normal: 18.5–24.9",
                "Insulin (µU/mL)": "Normal: 16–166"
            }
            raw_df["Normal Range"] = raw_df["Feature"].map(normal_ranges)

            facet_base = alt.Chart(raw_df).mark_bar().encode(
                y=alt.Y('Value:Q', title='Değer'),
                x=alt.X('Feature:N', axis=None),
                color=alt.Color('Color:N', scale=None),
                tooltip=['Feature', 'Value', 'Normal Range']
            ).properties(
                width=120,
                height=300
            )

            text_labels = alt.Chart(raw_df).mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                fontSize=13,
                color='black'
            ).encode(
                x='Feature:N',
                y='Value:Q',
                text=alt.Text('Value:Q', format='.1f')
            ).properties(width=120)

            range_labels = alt.Chart(raw_df).mark_text(
                align='center',
                baseline='bottom',
                dy=-25,
                fontSize=11,
                color='#555'
            ).encode(
                x='Feature:N',
                y='Value:Q',
                text='Normal Range:N'
            ).properties(width=120)

            final_chart = (facet_base + text_labels + range_labels).facet(
                column=alt.Column('Feature:N', title=None, header=alt.Header(labelFontWeight='bold', labelFontSize=14))
            )

        
            st.altair_chart(final_chart, use_container_width=False)
            st.markdown("</div>", unsafe_allow_html=True)

            advice_list = []

            if glucose < 140:
                glucose_comment = "✅ Glucose level is within the normal range."
                glucose_class = "success"
            elif glucose < 200:
                glucose_comment = "⚠️ Glucose level is borderline high."
                glucose_class = "warning"
                advice_list.append("⚠️ It is recommended to avoid sugary and processed foods.")
            else:
                glucose_comment = "❗ Glucose is very high, indicating a risk of diabetes."
                glucose_class = "error"
                advice_list.append("❗ Your glucose level is very high. Consult a specialist and review your diet.")

            if blood_pressure < 120:
                bp_comment = "✅ Blood pressure is at a normal level."
                bp_class = "success"
            elif blood_pressure < 140:
                bp_comment = "⚠️ Blood pressure is slightly high, monitoring is advised."
                bp_class = "warning"
                advice_list.append("⚠️ Reduce salt intake and regularly monitor your blood pressure.")
            else:
                bp_comment = "❗ Blood pressure is high, there may be a risk of hypertension."
                bp_class = "error"
                advice_list.append("❗ High blood pressure. Specialist consultation is recommended.")

            if 18.5 <= bmi < 25:
                bmi_comment = "✅ Your body mass index is ideal."
                bmi_class = "success"
            elif 25 <= bmi < 30:
                bmi_comment = "⚠️ Overweight, attention needed."
                bmi_class = "warning"
                advice_list.append("⚠️ Weight control can be achieved through exercise and diet.")
            else:
                bmi_comment = "❗ Obesity increases the risk of diabetes."
                bmi_class = "error"
                advice_list.append("❗ Obesity is present. It is important to get support from a dietitian and doctor.")

            if 16 <= insulin <= 166:
                insulin_comment = "✅ Insulin level is normal."
                insulin_class = "success"
            elif insulin < 16:
                insulin_comment = "⚠️ Insulin level may be low."
                insulin_class = "warning"
                advice_list.append("⚠️ Low insulin level may pose a risk of hypoglycemia.")
            else:
                insulin_comment = "❗ Insulin is very high, insulin resistance may be present."
                insulin_class = "error"
                advice_list.append("❗ Medical evaluation is recommended for high insulin levels.")

            if risk_prob >= 0.8:
                advice_list.append("❗ According to the model prediction, diabetes risk is very high. Consult a doctor as soon as possible.")
            elif risk_prob >= 0.6:
                advice_list.append("⚠️ Moderate diabetes risk present. Review your lifestyle.")
            elif risk_prob <= 0.2:
                advice_list.append("✅ Your diabetes risk appears low. Maintain your current lifestyle.")

            if not advice_list:
                advice_list.append("All your health indicators are normal. Keep maintaining this level.")

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
                <h2>📝 Diabetes Report 📝</h2>
            <h3>{'✅ No Diabetes ✅' if prediction[0] == 0 else '⚠️ Diabetes Present ⚠️'}</h3>

            <div class="section">
                {f'<p style="margin:0; padding:0;"><b>Name:</b> {user_name}</p>' if user_name else ""}
                <b>Age:</b> {Age} <br>
                <b>Gender:</b> {sex}
            </div>
            <h3>Health Status</h3>


            <ul>                    
                <li><span class="{glucose_class}">{glucose_comment}</span></li>
                <li><span class="{bp_class}">{bp_comment}</span></li>
                <li><span class="{bmi_class}">{bmi_comment}</span></li>
                <li><span class="{insulin_class}">{insulin_comment}</span></li>
            </ul>
            
            <div class="section">
            <h3 style="margin:0; padding:0;">Personalized Recommendations for {user_name}</h3>
                    <ul>
             {advice_html}
                    </ul>
            </div>

            <h3>General Health Recommendations</h3>
            <ul>
                <li>Healthy eating and regular exercise are very important.</li>
                <li>Limit smoking and alcohol consumption.</li>
                <li>Monitor your blood sugar levels regularly.</li>
                <li>Do not skip your annual health check-ups.</li>
                <li>Maintain a healthy weight to reduce diabetes risk.</li>
                <li>Manage stress through relaxation techniques or mindfulness.</li>
            </ul>
            </body>
            </html>

            """
          

            b64_report = base64.b64encode(report_html.encode()).decode()
            href = f'data:text/html;base64,{b64_report}'
            disease_type = "Diabetes"  
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
            file_name = f"{safe_user_name}_diabetes_report.html"

            st.markdown(f"""
                <div style="text-align: center; margin-top: 40px; animation: fadeIn 3s;">
                    <a href="{href}" download="{file_name}" style="
                        background: linear-gradient(90deg, #4C6EF5, #15AABF);
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
                    Diabetes Report 📥
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
