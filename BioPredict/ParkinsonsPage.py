import base64
from datetime import datetime
from textwrap import dedent
import streamlit as st
import pandas as pd
import joblib
import time
import altair as alt

@st.cache_resource
def load_parkinson_model():
    model = joblib.load("parkinson_model.pkl")
    scaler = joblib.load("parkinson_scaler.pkl")
    feature_names = joblib.load("parkinson_feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_parkinson_model()

user_name = st.session_state.get("user_name", "").strip()

user_gender = st.session_state.get("user_gender", "Select")

sex = user_gender if user_gender not in ["Select", "", None] else None

def validate_inputs(inputs_dict, user_name, sex):
    for k, v in inputs_dict.items():
        if v is None:
            st.warning(f"❗ Please enter a valid value for '{k}'.")
            return False

        if k not in ['spread1', 'spread2'] and isinstance(v, (int, float)) and v < 0:
            st.warning(f"❗ Please enter a non-negative value for '{k}'.")
            return False

    if not user_name:
        st.warning("⚠️ Please enter your name.")
        return False

    if sex is None:
        st.warning("⚠️ Please select your gender on the home page.")
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
    Parkinson's Disease Risk Prediction
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
        <div>Parkinson's disease is a progressive neurological disorder that affects movement.</div>
        <div>Early detection and intervention can significantly improve quality of life.</div>
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
        gender_label = "⚧️ Belirtilmedi"
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
    

feature_names = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)',  'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 
    'MDVP:Shimmer',  'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'NHR', 'HNR', 'RPDE', 'DFA',
    'spread1', 'spread2', 'D2', 'PPE'
] 

feature_ranges = {
    'MDVP:Fo(Hz)': (50.0, 250.0),
    'MDVP:Fhi(Hz)': (50.0, 300.0),
    'MDVP:Jitter(%)': (0.0, 0.02),
    'MDVP:Jitter(Abs)': (0.0, 0.01),
    'MDVP:RAP': (0.0, 0.01),
    'MDVP:PPQ': (0.0, 0.01),
    'MDVP:Shimmer': (0.0, 0.1),
    'Shimmer:APQ3': (0.0, 0.1),
    'Shimmer:APQ5': (0.0, 0.1),
    'MDVP:APQ': (0.0, 0.1),
    'NHR': (0.0, 0.2),
    'HNR': (0.0, 40.0),
    'RPDE': (0.0, 1.0),
    'DFA': (0.0, 2.0),
    'spread1': (-10.0, 10.0),
    'spread2': (-5.0, 5.0),
    'D2': (0.0, 5.0),
    'PPE': (0.0, 2.0)
}
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

inputs = []

st.markdown("<h2 style='font-weight:600; font-size:23px;'>Medical Findings and Tests</h2>", unsafe_allow_html=True)

with st.expander("Please fill in all values", expanded=True):
    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_names):
        min_val, max_val = feature_ranges.get(feature, (0.0, 1.0))
        step = (max_val - min_val) / 1000 
        if i % 2 == 0:
            inputs.append(col1.slider(feature, min_val, max_val, (min_val + max_val) / 2, step=step))
        else:
            inputs.append(col2.slider(feature, min_val, max_val, (min_val + max_val) / 2, step=step))

if st.button("Predict"):
    parkinsons_inputs = dict(zip(feature_names, inputs))

    if validate_inputs(parkinsons_inputs,user_name,sex):
        try:
            with st.spinner("Making prediction..."):
                time.sleep(2)


            input_data = pd.DataFrame([inputs], columns=feature_names)
            input_data[feature_names] = scaler.transform(input_data[feature_names])

            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)
            risk_prob = probabilities[0][1]

            if prediction == 0:
                st.balloons()
                st.markdown(
                    f"<h3 style='text-align: center; color: green; font-size: 24px;'>{user_name if user_name else 'This Person'} does <b>NOT</b> have Parkinson's Disease.</h3>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<h3 style='text-align: center; color: red; font-size: 24px;'>{user_name if user_name else 'This Person'} <b>HAS</b> Parkinson's Disease.</h3>",
                    unsafe_allow_html=True
                )

            st.markdown(
                f"<p style='text-align: center; font-size: 18px;'>Parkinson's disease risk probability: <b>{risk_prob * 100:.2f}%</b></p>",
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

            groups = {
                'Frequency (Hz)': ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)' ],
                'Jitter Measures': ['MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ'],
                'Shimmer Measures': ['MDVP:Shimmer', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ'],
                'Voice Quality': ['NHR', 'HNR', 'RPDE', 'DFA'],
                'Nonlinear Measures': ['spread1', 'spread2', 'D2', 'PPE']
            }

            normal_ranges = {
                'MDVP:Fo(Hz)': '50 - 250',
                'MDVP:Fhi(Hz)': '50 - 300',
                'MDVP:Jitter(%)': '< 0.02',
                'MDVP:Jitter(Abs)': '< 0.01',
                'MDVP:RAP': '< 0.01',
                'MDVP:PPQ': '< 0.01',
                'MDVP:Shimmer': '< 0.1',
                'Shimmer:APQ3': '< 0.1',
                'Shimmer:APQ5': '< 0.1',
                'MDVP:APQ': '< 0.1',
                'NHR': '< 0.2',
                'HNR': '> 20',
                'RPDE': '< 1.0',
                'DFA': '< 2.0',
                'spread1': '-10 - 10',
                'spread2': '-5 - 5',
                'D2': '< 5',
                'PPE': '< 2'
            }
            def get_color(feature, value):

                if feature in ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)']:
                    if 50 <= value <= 250:
                        return "#4CAF50"  
                    elif 250 < value < 270:
                        return "#FFC107"  
                    else:
                        return "#F44336" 
                
                elif feature in ['MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ']:
                    if value < 0.01:
                        return "#4CAF50"
                    elif 0.01 <= value < 0.015:
                        return "#FFC107"
                    else:
                        return "#F44336"
                
                elif feature in ['MDVP:Shimmer', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ']:
                    if value < 0.07:
                        return "#4CAF50"
                    elif 0.07 <= value < 0.09:
                        return "#FFC107"
                    else:
                        return "#F44336"
                
                elif feature in ['NHR', 'RPDE', 'DFA']:
                    if value < 1.0:
                        return "#4CAF50"
                    elif 1.0 <= value < 1.5:
                        return "#FFC107"
                    else:
                        return "#F44336"
                elif feature == 'HNR':
                    if value >= 20:
                        return "#4CAF50"
                    elif 15 <= value < 20:
                        return "#FFC107"
                    else:
                        return "#F44336"
                
                elif feature == 'spread1':
                    if -10 < value < 10:
                        return "#4CAF50"
                    elif value <= -10 or value >= 10:
                        return "#F44336"
                    else:
                        return "#FFC107"
                elif feature == 'spread2':
                    if -5 < value < 5:
                        return "#4CAF50"
                    elif value <= -5 or value >= 5:
                        return "#F44336"
                    else:
                        return "#FFC107"
                elif feature == 'D2':
                    if 0 < value < 5:
                        return "#4CAF50"
                    elif 5 <= value < 7:
                        return "#FFC107"
                    else:
                        return "#F44336"
                elif feature == 'PPE':
                    if 0 <= value < 2:
                        return "#4CAF50"
                    elif 2 <= value < 3:
                        return "#FFC107"
                    else:
                        return "#F44336"
                
                else:
                    return "#9E9E9E"
                
            for group_name, group_features in groups.items():
                group_data = {f: parkinsons_inputs[f] for f in group_features if f in parkinsons_inputs}
                group_df = pd.DataFrame(list(group_data.items()), columns=['Feature', 'Value'])
                group_df['Color'] = group_df.apply(lambda row: get_color(row['Feature'], row['Value']), axis=1)
                group_df['Normal Range'] = group_df['Feature'].map(normal_ranges)

                st.markdown(f"### {group_name}")
                if group_name == 'Nonlinear Measures':
                    min_y = -10.0
                else:
                    min_y = 0.0

                max_y = group_df['Value'].max() * 1.1  
                group_df['Value'] = group_df['Value'].clip(lower=min_y, upper=max_y)    

                base = alt.Chart(group_df).mark_bar().encode(
                    x=alt.X('Feature:N', axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y('Value:Q', scale=alt.Scale(domain=[min_y, max_y])),  
                    color=alt.Color('Color:N', scale=None),
                    tooltip=['Feature', 'Value', 'Normal Range']
                ).properties(width=600, height=300)

                text = alt.Chart(group_df).mark_text(
                    align='center',
                    baseline='bottom',
                    dy=-5,
                    fontSize=12,
                    color='black'
                ).encode(
                    x='Feature:N',
                    y='Value:Q',
                    text=alt.Text('Value:Q', format='.4f')
                )

                range_text = alt.Chart(group_df).mark_text(
                    align='center',
                    baseline='bottom',
                    dy=-25,
                    fontSize=11,
                    color='#555'
                ).encode(
                    x='Feature:N',
                    y='Value:Q',
                    text='Normal Range:N'
                )

                chart = (base + text + range_text)

                st.altair_chart(chart, use_container_width=True)
    
            advice_list = []

            for feature, value in parkinsons_inputs.items():
                if feature in ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)']:
                    if 50 <= value <= 250:
                        mdpv_class = "success"
                        mdvp_comment = "✅ Your voice frequency is within the normal range."
                    elif 250 < value < 270:
                        mdpv_class = "warning"
                        mdvp_comment = "⚠️ Your voice frequency is slightly high."
                        advice_list.append("Your voice frequency is slightly high. Avoid straining your voice.")
                    else:
                        mdpv_class = "error"
                        mdvp_comment = "Your voice frequency is abnormally high."
                        advice_list.append("❗ Abnormal voice frequency detected. Please consult an ENT specialist.")

                elif feature in ['MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ']:
                    if value < 0.01:
                        mdvp2_class = "success"
                        mdvp2_comment = "✅ Jitter value is normal."
                    elif 0.01 <= value < 0.015:
                        mdvp2_class = "warning"
                        mdvp2_comment = "Jitter value is borderline."
                        advice_list.append("⚠️ There may be irregularities in your voice vibrations. Try to rest your voice.")
                    else:
                        mdvp2_class = "error"
                        mdvp2_comment = "Jitter value is high."
                        advice_list.append("❗ High irregularity detected in your voice vibrations. A specialist check is recommended.")

                elif feature in ['MDVP:Shimmer', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ']:
                    if value < 0.07:
                        shimmer_class = "success"
                        shimmer_comment = "✅ Voice amplitude is normal."
                    elif 0.07 <= value < 0.09:
                        shimmer_class = "warning"
                        shimmer_comment = "Voice amplitude is slightly high."
                        advice_list.append("⚠️ Your voice amplitude is slightly high. Make sure to rest your voice.")
                    else:
                        shimmer_class = "error"
                        shimmer_comment = "Voice amplitude is abnormally high."
                        advice_list.append("❗ Abnormal voice amplitude detected. A specialist check is recommended.")


            if risk_prob >= 0.8:
                advice_list.append("❗ Very high risk of Parkinson’s. Please consult a neurologist.")
            elif risk_prob >= 0.5:
                advice_list.append("⚠️ Moderate risk of Parkinson’s. Monitoring is recommended.")
            else:
                advice_list.append("✅ Low risk of Parkinson’s. Keep up your healthy lifestyle!")

            if not advice_list:
                advice_list.append("👏 All indicators are normal. Congratulations!")

        
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
                color: {'green' if prediction == 0 else 'red'};
                font-size: 20px;
                margin-bottom: 20px;
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
            <h2>📝 Parkinson Report 📝</h2>
            <h3>{'✅ No Parkinson Disease Detected ✅' if prediction == 0 else '⚠️ Parkinson Disease Detected ⚠️'}</h3>


            <div class="section">
            {f'<p style="margin:0; padding:0;"><b>Ad:</b> {user_name}</p>' if user_name else ""}
            {f'<p style="margin:0; padding:0;"><b>Cinsiyet:</b> {sex}</p>' if sex else ""}
            </div>

                            <h3>Health Status</h3>

            <ul>                 
                <li><span class="{mdpv_class}">{mdvp_comment}</span></li>   
                <li><span class="{mdvp2_class}">{mdvp2_comment}</span></li>
                <li><span class="{shimmer_class}">{shimmer_comment}</span></li>
            </ul>
            
            <div class="section">
                <h3 style="margin:0; padding:0;">Personalized Recommendations for {user_name}</h3>
                    <ul>
             {advice_html}
                    </ul>
            </div>

            <h3>General Health Recommendations</h3>
            <ul>
                <li>Do not neglect regular neurological check-ups.</li>
                <li>Stress management and a healthy lifestyle can reduce the risk of Parkinson's.</li>
                <li>Participation in regular exercise and social activities is important.</li>
                <li>Maintain a balanced diet rich in antioxidants and omega-3 fatty acids.</li>
                <li>Ensure quality sleep to support brain health and overall well-being.</li>
                <li>Avoid exposure to environmental toxins and harmful chemicals when possible.</li>
                <li>Engage in cognitive activities such as reading, puzzles, or learning new skills to keep your mind active.</li>
            </ul>

            </body>
            </html>
            """
            
            b64_report = base64.b64encode(report_html.encode()).decode()
            href = f'data:text/html;base64,{b64_report}'

            disease_type = "Parkinson"  

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
            file_name = f"{safe_user_name}_parkinsons_report.html"

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
                    Parkinson’s Report 📥
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
            st.error(f"An error occured: {e}")


