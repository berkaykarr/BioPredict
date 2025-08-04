# 🧬 BioPredict - Health Risk Prediction App

BioPredict is a web-based machine learning application that predicts potential health risks based on user-provided biometric and lifestyle data. Built with **Streamlit**, it offers an interactive interface, visual insights, and a downloadable health report.

## Features

- 📊 **Health Data Form** – Users input biometric and lifestyle data.
- 🧠 **Machine Learning Prediction** – Real-time risk prediction using trained models.
- 📈 **Visual Insights** – Graphical analysis of input data and prediction results.
- 📄 **Custom PDF Report** – Downloadable HTML report summarizing user data and model outputs.
- 🔒 **User-friendly & Private** – No data is stored; all predictions are local and temporary.

## Technologies Used

- Python
- Streamlit
- Scikit-learn / XGBoost / GradientBoostingClassifier / Support Vector Classification
- Matplotlib / Plotly
- Pandas, NumPy

## How to Run Locally

```bash
git clone https://github.com/berkaykarr/BioPredict.git
cd BioPredict
pip install -r requirements.txt
streamlit run Home.py
