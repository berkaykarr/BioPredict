from datetime import datetime
from textwrap import dedent
import streamlit as st
import pandas as pd
import joblib
from vega_datasets import data
import altair as alt
import time
import base64



model_data = joblib.load("heart_model.pkl")

clf = model_data['model']
scaler = model_data['scaler']
mms = model_data['mms']
le_dict = model_data['le_dict']
feature_names = model_data['feature_names']
scaler_features = model_data['scaler_features']
mms_features = model_data['mms_features']

def validate_inputs(sex, chest_pain, fasting_bs, resting_ecg, exercise_angina, st_slope, age, oldpeak):
    required_fields = [sex, chest_pain, fasting_bs, resting_ecg, exercise_angina, st_slope]
    
    if "Select" in required_fields:
        st.warning("Please fill in all selection boxes!", icon="⚠️")
        return False
    
    if age <= 0:
        st.warning("Please enter a valid age!", icon="⚠️")
        return False

    if oldpeak < 0.0:
        st.warning("ST depression must be zero or above.", icon="⚠️")
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
    Heart Disease Risk Prediction
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
        <div>Heart attacks occur when blood flow to the heart is blocked.</div>
        <div>Early recognition and emergency treatment are critical for survival.</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption("This tool is for educational purposes only and does not provide a medical diagnosis. Results should be interpreted in conjunction with clinical evaluation.")


user_name = st.session_state.get("user_name", "").strip()

user_gender = st.session_state.get("user_gender", "Select")

sex = user_gender 

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

st.markdown("<h2 style='font-weight:600; font-size:23px;'>Medical Findings and Tests</h2>", unsafe_allow_html=True)
with st.expander("", expanded=True):
    
    col1, col2, col3 = st.columns(3)
    with col1:
        chest_pain = st.selectbox("Chest Pain Type", ["Select", 'ATA', 'NAP', 'ASY', 'TA'])
    with col2:
        fasting_bs = st.selectbox("Is fasting blood sugar > 120 mg/dl?", ["Select", "0", "1"])
    with col3:
        st_slope = st.selectbox("ST Slope", ["Select", 'Up', 'Flat', 'Down'])

    col4, col5 = st.columns(2)
    with col4:
        resting_bp = st.slider("Resting Blood Pressure (mmHg)", min_value=80, max_value=160)
        resting_ecg = st.selectbox("Resting ECG Result", ["Select", 'Normal', 'ST', 'LVH'])
        oldpeak = st.slider("ST Depression (mm)", 0.0, 10.0, 0.0, 0.1)
    with col5:
        cholesterol = st.slider("Cholesterol (mg/dL)", 120, 350, step=5)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Select", 'N', 'Y'])
        max_hr = st.slider("Maximum Heart Rate (bpm)", 60, 220)

        
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
expected_max_hr = 220 - age


def get_color(feature, value):
    if feature == "Cholesterol (mg/dl)":
        return "#4CAF50" if value < 200 else "#FFC107" if value < 239 else "#F44336"
    elif feature == "Resting Blood Pressure":
        return "#4CAF50" if value < 120 else "#FFC107" if value < 139 else "#F44336"
    elif feature == "Maximum Heart Rate":
        return "#4CAF50" if value > expected_max_hr * 0.9 else "#FFC107" if value >= expected_max_hr * 0.7 else "#F44336"
    elif feature == "ST Depression":
        return "#4CAF50" if value < 1 else "#FFC107" if value < 2 else "#F44336"
    return "#9E9E9E"


if st.button("Predict"):
    if validate_inputs(sex, chest_pain, fasting_bs, resting_ecg, exercise_angina, st_slope, age, oldpeak):
        try:
            with st.spinner("Making prediction..."):
                time.sleep(2)

            categorical_input = {
                'Sex': sex,
                'ChestPainType': chest_pain,
                'RestingECG': resting_ecg,
                'ExerciseAngina': exercise_angina,
                'ST_Slope': st_slope
            }

            encoded = {}
            for col, val in categorical_input.items():
                le = le_dict.get(col)
                if le is None:
                    st.error(f"Label encoder for {col} not found!")
                    st.stop()
                if val not in le.classes_:
                    st.error(f"Unknown value for {col}: {val}")
                    st.stop()
                encoded[col] = le.transform([val])[0]

            try:
                fasting_bs_val = int(fasting_bs)
            except ValueError:
                st.error("Invalid value for fasting blood sugar.")
                st.stop()

            input_dict = {
                'Age': age,
                'Sex': encoded['Sex'],
                'ChestPainType': encoded['ChestPainType'],
                'RestingBP': resting_bp,
                'Cholesterol': cholesterol,
                'FastingBS': fasting_bs_val,
                'RestingECG': encoded['RestingECG'],
                'MaxHR': max_hr,
                'ExerciseAngina': encoded['ExerciseAngina'],
                'Oldpeak': oldpeak,
                'ST_Slope': encoded['ST_Slope']
            }

            input_data = pd.DataFrame([input_dict], columns=feature_names)

            input_data[scaler_features] = scaler.transform(input_data[scaler_features])
                
            input_data[mms_features] = mms.transform(input_data[mms_features])
                
            prediction = clf.predict(input_data)
            probabilities = clf.predict_proba(input_data)
            risk_prob = probabilities[0][1]  

            if prediction[0] == 0:
                st.balloons()
                st.markdown(
            f"<h3 style='text-align: center; color: green; font-size: 24px;'> {user_name if user_name else 'This person'} is <b>NOT</b> at <b>risk</b> of Heart Disease.</h3>",
            unsafe_allow_html=True
                )
            else:
                st.markdown(
            f"<h3 style='text-align: center; color: red; font-size: 24px;'> {user_name if user_name else 'This person'} IS at <b>RISK</b> of Heart Disease!</h3>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"<p style='text-align: center; font-size: 18px;'>Probability of Heart Disease: <b>{risk_prob * 100:.2f}%</b></p>",
                unsafe_allow_html=True
            )

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
                'Resting Blood Pressure': 'Normal: <120 mmHg',
                'Cholesterol (x0.1)': 'Normal: <200 (mg/dL)',
                'Maximum Heart Rate': f'Normal: >{int(expected_max_hr * 0.9)} bpm',
                'ST Depression (x10)': 'Normal: <10'
            }

            raw_data_dict = {
                'Resting Blood Pressure': resting_bp,
                'Cholesterol (x0.1)': cholesterol / 10,
                'Maximum Heart Rate': max_hr,
                'ST Depression (x10)': oldpeak * 10
            }
            raw_df = pd.DataFrame(raw_data_dict.items(), columns=['Feature', 'Value'])

            def color_lambda(row):
                if row['Feature'] == 'Cholesterol (x0.1)':
                    return get_color('Cholesterol (mg/dl)', cholesterol)
                elif row['Feature'] == 'ST Depression (x10)':
                    return get_color('ST Depression', oldpeak)
                else:
                    return get_color(row['Feature'], row['Value'])

            raw_df['Color'] = raw_df.apply(color_lambda, axis=1)
            raw_df['Normal Range'] = raw_df['Feature'].map(normal_ranges)

            y_domains = {
                'Resting Blood Pressure': [80, 160],
                'Cholesterol (x0.1)': [10, 60],  
                'Maximum Heart Rate': [60, 220],
                'ST Depression (x10)': [0, 100]  
            }
            max_y = max(limit[1] for limit in y_domains.values())

            base_chart = alt.Chart(raw_df).mark_bar().encode(
                x=alt.X('Feature', sort='-y'),
                y=alt.Y('Value', scale=alt.Scale(domain=[0, max_y])),
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

            normal_range_text = alt.Chart(raw_df).mark_text(
                align='center',
                baseline='top',
                dy=10,
                fontSize=11,
                color='#555'
            ).encode(
                x='Feature:N',
                y=alt.value(0),
                text='Normal Range:N'
            )

            final_chart = base_chart + text + normal_range_text

            st.altair_chart(final_chart, use_container_width=True)
        
            advice_list = []

            expected_max_hr = 220 - age

            if resting_bp <= 120:
                bp_comment = "🩺 Resting blood pressure is at a normal level."
                bp_class = "success"
            elif resting_bp <= 139:
                advice_list.append("⚠️ Your blood pressure is slightly high. You can manage it with stress control and regular exercise.")
                bp_comment = "⚠️ Resting blood pressure is slightly high; monitor your blood pressure."
                bp_class = "warning"
            else:
                advice_list.append("❗ Your blood pressure appears high. Reduce sodium intake and exercise regularly.")
                bp_comment = "❗ Resting blood pressure is high; please consult a healthcare professional."
                bp_class = "error"

            if cholesterol < 200:
                cholesterol_comment = "🧬 Cholesterol level is ideal."
                cholesterol_class = "success"
            elif cholesterol <= 239:
                advice_list.append("⚠️ Cholesterol is borderline high. It is recommended to reduce red meat and processed foods.")
                cholesterol_comment = "⚠️ Cholesterol is borderline high; diet and exercise are recommended."
                cholesterol_class = "warning"
            else:
                advice_list.append("❗ Your cholesterol level is high. Reduce saturated fat intake and consume plenty of vegetables and fiber.")
                cholesterol_comment = "❗ High cholesterol; medical attention is important."
                cholesterol_class = "error"

            if max_hr >= expected_max_hr * 0.9:
                hr_comment = f"❤️ Maximum heart rate is appropriate for age ({max_hr} bpm)."
                hr_class = "success"
            elif max_hr >= expected_max_hr * 0.7:
                advice_list.append("⚠️ Your heart rate is lower than expected. You can start cardio exercises to improve your exercise capacity.")
                hr_comment = f"⚠️ Maximum heart rate is slightly low ({max_hr} bpm). You can increase your exercise capacity."
                hr_class = "warning"
            else:
                advice_list.append("❗ Your heart rate is quite low. There may be underlying cardiovascular issues; please seek specialist evaluation.")
                hr_comment = f"❗ Maximum heart rate is quite low for your age ({max_hr} bpm). Medical evaluation is recommended."
                hr_class = "error"

            if oldpeak <= 1:
                st_comment = "📉 ST depression is at a normal level."
                st_class = "success"
            elif oldpeak <= 2:
                advice_list.append("⚠️ ST depression is moderate. Do not neglect your routine check-ups.")
                st_comment = "⚠️ ST depression is moderate; your heart stress may have increased."
                st_class = "warning"
            else:
                advice_list.append("❗ Your ST depression value is high. A cardiac stress test and doctor consultation may be necessary.")
                st_comment = "❗ ST depression is high; your risk of heart disease may increase."
                st_class = "error"

            if risk_prob >= 0.8:
                advice_list.append("❗ According to the model prediction, your heart disease risk is very high. Consult a cardiologist as soon as possible.")
            elif risk_prob >= 0.6:
                advice_list.append("⚠️ The model predicts a moderate to high risk. It is recommended to review your lifestyle.")
            elif risk_prob <= 0.2:
                advice_list.append("✅ Your heart disease risk is low. You can maintain your health by keeping your current lifestyle.")

            if not advice_list:
                advice_list.append("👏 All your health indicators are normal! Keep maintaining this level.")

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
            <h2>📝 Heart Health Report 📝</h2>
            <h3>{'✅ No Heart Disease ✅' if prediction[0] == 0 else '⚠️ Heart Disease Detected ⚠️'}</h3>

            <div class="section">
                    {f'<p style="margin:0; padding:0;"><b>Ad:</b> {user_name}</p>' if user_name else ""}
                    <b>Age:</b> {age} <br>
                    <b>Gender:</b> {sex}
            </div>

            <h3>Health Status</h3>,
            <ul>
                <li><span class="{bp_class}">{bp_comment}</span></li>
                <li><span class="{cholesterol_class}">{cholesterol_comment}</span></li>
                <li><span class="{hr_class}">{hr_comment}</span></li>
                <li><span class="{st_class}">{st_comment}</span></li>
            </ul>

           <div class="section">
                <h3 style="margin:0; padding:0;">Personalized Recommendations for {user_name}</h3>
                    <ul>
             {advice_html}
                    </ul>
            </div>
                <h3>General Health Recommendations</h3>
             <ul>
                <li>Maintaining a healthy diet and engaging in regular physical activity are crucial for heart health.</li>
                <li>Avoid smoking and alcohol consumption.</li>
                <li>Have regular medical check-ups.</li>
                <li>Monitor your blood pressure and cholesterol levels consistently.</li>
                <li>If you experience any symptoms or discomfort, seek medical attention without delay.</li>
                <li>Ensure you get adequate sleep and manage stress effectively to reduce heart strain.</li>
                <li>Maintain a healthy weight, as obesity is a significant risk factor for heart disease.</li>
            </ul>

            </body>
            </html>
            """
            b64_report = base64.b64encode(report_html.encode()).decode()
            href = f'data:text/html;base64,{b64_report}'
            
            disease_type = "Heart Disease"  
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
            file_name = f"{safe_user_name}_heart_health_report.html"


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
                    Heart Disease Report 📥
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

