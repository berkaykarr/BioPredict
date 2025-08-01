import streamlit as st

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "user_gender" not in st.session_state:
    st.session_state.user_gender = "Select"

col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 style='font-weight: 700;'> <b>What is your</b> Name?</h3>", unsafe_allow_html=True)
    st.session_state.user_name = st.text_input(
        label="", 
        value=st.session_state.user_name, 
        placeholder="Enter your name..."
    )

with col2:
    st.markdown("<h3 style='font-weight: 700;'> <b>What is your</b> Gender?</h3>", unsafe_allow_html=True)
    st.session_state.user_gender = st.selectbox(
        label="",
        options=["Select", "Female", "Male"],
        index=["Select", "Female", "Male"].index(st.session_state.user_gender)
    )

user_name = st.session_state.user_name.strip()

user_gender = st.session_state.user_gender

if user_name and user_gender in ["Female", "Male"]:
    if user_gender == "Female":
        text_color = "#FF69B4" 
    else: 
        text_color = "#1E90FF"

    st.markdown(f"""
        <h1 style="
            font-size: 48px;
            font-weight: 900;
            color: {text_color};
            text-align: center;
            margin-top: 10px;
            margin-bottom: 5px;
            font-family: 'Arial Black', sans-serif;
            animation: fadeIn 1.2s ease-in-out;
        ">
        Welcome, {user_name} 
        </h1>
    """, unsafe_allow_html=True)
else:
    st.warning("Please enter your Name and Gender to personalize your experience.")

    st.markdown(f"""
        <h1 style="
            font-size: 42px;
            font-weight: 800;
            color: #3A3B7A;
            text-align: center;
            margin-top: 10px;
            margin-bottom: 5px;
            font-family: 'Arial Black', sans-serif;
        ">
        Welcome to BioPredict 
        </h1>
    """, unsafe_allow_html=True)

st.markdown("""
    <p style="
        text-align: center;
        font-size: 20px;
        font-weight: 500;
        color: #3A3B7A;
        margin-bottom: 30px;
        animation: fadeIn 2.2s ease-in;
    ">
    Empowering your health decisions with <span style="color:#D156B7; font-weight:600;">AI-powered</span> predictions.
    </p>
""", unsafe_allow_html=True)

with st.expander("What does this app do?", expanded=True):
    st.markdown("""
        <div style="
            background-color: #F9F9FC;
            padding: 25px;
            border-radius: 14px;
            font-size: 17px;
            line-height: 1.75;
            color: #1C1C1C;
            border-left: 6px solid #6C63FF;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            animation: fadeIn 1s ease-in-out;
        ">
        <h3 style="color:#3A3B7A;">What is <strong>BioPredict</strong>?</h3>
        
        <p><strong>BioPredict</strong> is an AI-powered health prediction tool that helps you estimate your risk for 5 major diseases:</p>

        <ul>
            <li>❤️ <strong>Heart Disease</strong> – Based on blood pressure, cholesterol and lifestyle.</li>
            <li>🩸 <strong>Diabetes</strong> – Using glucose, BMI, age and more.</li>
            <li>🧠 <strong>Parkinson’s</strong> – Early detection through movement and behavior patterns.</li>
            <li>🧬 <strong>Thyroid Cancer</strong> – Detects malignancy risk based on medical indicators.</li>
            <li>🧓 <strong>Alzheimer’s</strong> – Cognitive and demographic evaluation.</li>
        </ul>

        <p>✅ Easy to use — no technical knowledge required. Just enter your data and see results instantly.</p>

        <p>📄 After prediction, a detailed <strong>Health Report</strong> is generated and saved under the <em>Reports</em> section. You can view, track, or download it anytime.</p>

        <p>💡 <em>Our mission:</em> To bring accessible, elegant and powerful AI health tools to everyone.</p>
        </div>

        <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; margin-top: 40px; animation: fadeIn 3s;">
        <a href="#Heart Disease Predict" style="
            background: linear-gradient(90deg, #D156B7, #FF82B2);
            color: white;
            padding: 14px 30px;
            font-size: 18px;
            font-weight: bold;
            border-radius: 12px;
            text-decoration: none;
            box-shadow: 3px 3px 10px rgba(0,0,0,0.2);
            transition: 0.3s;
        ">
         Start Predicting Now 🚀
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

st.markdown("""
    <p style="
        text-align: center;
        font-size: 16px;
        font-style: italic;
        color: #555;
        margin-top: 50px;
    ">
    Developed by <b>Berkay Karadeniz</b>
    </p>
""", unsafe_allow_html=True)
