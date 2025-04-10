# ❤️ Heartly - Heart Attack Detection & SOS Alert System

**Heartly** is a smart health monitoring web application that predicts the risk of a heart attack using a Machine Learning model. The app also features an AI-powered chatbot "Heartly" (powered by Gemini AI) for real-time assistance and includes an integrated SOS alert system via SMS and Email for emergency situations.

---

## 🚀 Features

- 🔍 **Heart Attack Risk Prediction** using a trained ML model (Random Forest Classifier)
- 🧠 **AI Chatbot - Heartly** powered by Gemini AI for guidance, FAQs, and lifestyle recommendations
- 🚨 **SOS Alert System** that sends emergency messages via SMS and Email
- 🏥 **Health Tips** and **Medication Recommendations**
- 📊 **Interactive Visuals** for understanding health data
- 📱 Built using **Streamlit**, deployed on the cloud for easy access

---

## 🧠 Machine Learning Model

- Trained on a heart disease dataset with features like age, cholesterol, chest pain type, etc.
- Model used: **Random Forest Classifier**
- Accuracy: ~85%
- Preprocessing includes scaling and encoding for optimal performance

---

## 📦 Tech Stack

- **Frontend/UI:** Streamlit
- **Backend:** Python, scikit-learn
- **Chatbot:** Gemini AI integration (via API)
- **Deployment:** Streamlit Cloud / Google Colab
- **Alert System:** `smtplib`, `twilio` or any other API for SMS/Email

---

## 🧪 Installation & Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/heartly-app.git
cd heartly-app

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
