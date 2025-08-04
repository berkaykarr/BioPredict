# BioPredict - Health Risk Prediction App

BioPredict is a web-based machine learning application that predicts the likelihood of five major health conditions — heart disease, diabetes, Parkinson’s, thyroid disorders, and Alzheimer’s — based on user-provided biometric and test data. Built with Streamlit, it features an intuitive interface where users can input their lab results and lifestyle information. With a single click on the "Predict" button, the app delivers a detailed and downloadable health risk report powered by trained machine learning models.

⚠️ This tool is developed for educational and demonstrational purposes only. The predictions do not replace professional medical advice. Always consult a healthcare provider for clinical evaluation and diagnosis.
## 🎥 Demo Video

[![Watch the Demo](https://img.youtube.com/vi/qewoPROoLhc/0.jpg)](https://youtu.be/qewoPROoLhc)

Click the image above to watch a walkthrough of the BioPredict app.


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
