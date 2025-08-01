import os
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

st.set_page_config(page_title="BioPredict", page_icon="💓", layout="wide", initial_sidebar_state="expanded")


if "reports" not in st.session_state:
    st.session_state["reports"] = []
if "menu" not in st.session_state:
    st.session_state["menu"] = "Main Page"

with st.sidebar:
    logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
    image = Image.open(logo_path)    
    st.image(image, use_column_width=True)
    st.markdown(
        """
        <h1 style="
            font-size: 48px; 
            font-weight: 900; 
            background: linear-gradient(90deg, #00b3b3, #e3428b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            color: transparent;
            text-align: center;
            margin-bottom: 10px;
            font-family: 'Arial Black', sans-serif;
            text-shadow: 1.5px 1.5px 3px rgba(0, 0, 0, 0.4);
        ">
        BioPredict
        </h1>
        """,
        unsafe_allow_html=True
    )
    selected = option_menu(
        menu_title="",
        options=["Main Page", "Heart Disease Predict", "Diabetes Predict", "Parkinson's Predict", "Thyroid Predict", "Alzheimer's Predict", "My Reports"],
        icons=["balloon", "heart", "activity", "cpu", "file-earmark-medical", "person-bounding-box", "clipboard-data"],
        default_index=[
            "Main Page", "Heart Disease Predict", "Diabetes Predict", "Parkinson's Predict",
            "Thyroid Predict", "Alzheimer's Predict", "My Reports"
        ].index(st.session_state["menu"]),
        key="menu"
    )

if selected == "Main Page":
    exec(open("MainPage.py", encoding="utf-8").read())
elif selected == "Heart Disease Predict":
    exec(open("HeartDiseasePage.py", encoding="utf-8").read())
elif selected == "Diabetes Predict":
    exec(open("DiabetesPage.py", encoding="utf-8").read())
elif selected == "Parkinson's Predict":
    exec(open("ParkinsonsPage.py", encoding="utf-8").read())
elif selected == "Thyroid Predict":
    exec(open("ThyroidPage.py", encoding="utf-8").read())  
elif selected == "Alzheimer's Predict":
    exec(open("Alzheimers.py", encoding="utf-8").read())
elif selected == "My Reports":
    exec(open("ReportsPage.py", encoding="utf-8").read())
