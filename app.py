import streamlit as st
import joblib
import numpy as np
import streamlit as st
from PIL import Image
import google.generativeai as genai
def apply_custom_theme():
    st.markdown(
        """
        <style>
        :root {
            --primary-color: #BD5E8E;  /* Raspberry/Rose */
            --secondary-color: #5D2A6D; /* Medium purple */
            --accent-color: #D4AF37;    /* Gold accent */
            --background-color: #F8F5FA; /* Very light purple-tinted background */
            --card-color: #FFFFFF;      /* White for cards */
            --text-color: #333333;      /* Dark text for contrast */
            --font: 'Montserrat', sans-serif;
        }
        body {
            font-family: var(--font);
            color: var(--text-color);
            background-color: var(--background-color);
        }
        .stApp {
            background-color: var(--background-color);
            background-image: linear-gradient(135deg, #F8F5FA 25%, #F0EBF7 100%);
        }
        h1 {
            color: var(--primary-color);
            font-weight: 600;
        }
        h2, h3, h4, h5, h6 {
            color: var(--secondary-color);
            font-weight: 500;
        }
        .sidebar .sidebar-content {
            background-image: linear-gradient(180deg, #F0EBF7 0%, #E8E1F2 100%);
            border-right: 1px solid rgba(189, 94, 142, 0.2);
        }
        .css-1lcbmhc {
            background-image: linear-gradient(180deg, #F0EBF7 0%, #E8E1F2 100%);
        }
        /* Main content area */
        .css-ke7pzk {
            background-color: transparent;
        }
        /* Container styling with subtle rose border */
        .css-1r6slb0, .css-12w0qpk, .stAlert {
            background-color: var(--card-color);
            border: 1px solid rgba(189, 94, 142, 0.25);
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }
        /* Button styling */
        .stButton>button {
            background: linear-gradient(45deg, #BD5E8E, #CB648F);
            color: white;
            border: none;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(189, 94, 142, 0.3);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(45deg, #CB648F, #DA6A96);
            box-shadow: 0 4px 12px rgba(189, 94, 142, 0.4);
            transform: translateY(-2px);
        }
        /* Input fields with rose accent */
        input, select, textarea {
            background-color: #FFFFFF !important;
            color: #333333 !important;
            border: 1px solid rgba(189, 94, 142, 0.3) !important;
            border-radius: 6px !important;
        }
        input:focus, select:focus, textarea:focus {
            border: 1px solid rgba(189, 94, 142, 0.8) !important;
            box-shadow: 0 0 0 2px rgba(189, 94, 142, 0.2) !important;
        }
        /* Slider color */
        .stSlider div[data-baseweb="slider"] div {
            background-color: var(--primary-color) !important;
        }
        /* Progress bar */
        .stProgress div {
            background-color: var(--primary-color) !important;
        }
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #F0EBF7;
            border-radius: 8px;
            padding: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: var(--text-color);
            border-radius: 6px;
        }
        .stTabs [aria-selected="true"] {
            background-color: white;
            border-bottom: 2px solid var(--primary-color) !important;
        }
        /* Dropdown menu */
        div[data-baseweb="select"] > div {
            background-color: white !important;
            border: 1px solid rgba(189, 94, 142, 0.3) !important;
        }
        /* Gold accents for certain elements */
        .stCheckbox label span p {
            color: var(--secondary-color) !important;
        }
        .stExpander {
            border-left: 1px solid var(--accent-color) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

 # Load and display the image
image = Image.open("h1.jpg")
st.image(image, caption="Your Image",use_container_width=True)
def get_gemini_response(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # Replace with an available model if necessary
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client
import streamlit as st
import google.generativeai as genai

# Load Gemini AI API Key
GEMINI_API_KEY = "AIzaSyDnI_MLvTZb38YX9aBwfT69Aq2_67TMHmE"
genai.configure(api_key=GEMINI_API_KEY)

# Emergency Contact Details
EMERGENCY_EMAIL = "emergency@example.com"  # Replace with the recipient's email
TWILIO_ACCOUNT_SID = "ACc418eadad41fa6405eba9b8e5057e664"
TWILIO_AUTH_TOKEN = "05fd4e0093b0b3b1c4a913edc39cf50c"
TWILIO_PHONE_NUMBER = "+17753805206"  # Replace with Twilio number
EMERGENCY_PHONE = " +917822966398"  # Replace with actual emergency contact number

def send_email_alert(name, risk_level):
    sender_email = "dnyaneshshinde645@gmail.com"  # Use your email
    sender_password = "tdqw nwar bjcl alpe"  # Use an app password

    subject = "🚨 EMERGENCY: Possible Heart Attack Risk!"
    body = f"Patient {name} is detected with a high heart attack risk level: {risk_level}. Immediate action needed!"

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = EMERGENCY_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, EMERGENCY_EMAIL, msg.as_string())
        server.quit()
        st.success("🚀 Emergency email sent successfully!")
    except Exception as e:
        st.error(f"⚠ Error sending email: {e}")

def send_sms_alert(name, risk_level):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    message = f"🚨 Emergency Alert: {name} HAS HEART ATTACK WITH RISK- ({risk_level})! Immediate help needed!"

    try:
        client.messages.create(body=message, from_=TWILIO_PHONE_NUMBER, to=EMERGENCY_PHONE)
        st.success("📲 Emergency SMS sent successfully!")
    except Exception as e:
        st.error(f"⚠ Error sending SMS: {e}")

st.title("🚑 Heart Attack Prediction with SOS Alert")

# Input fields
name = st.text_input("Enter Patient Name")
risk_level = st.selectbox("Risk Level", ["Low", "Medium", "High"])

# SOS Button
if st.button("🚨 Send Emergency Alert"):
    if risk_level == "High":
        send_email_alert(name, risk_level)
        send_sms_alert(name, risk_level)
    else:
        st.warning("⚠ Alert only triggers for High Risk cases.")



# Load the trained Random Forest model and scaler
def load_model_and_scaler():
    try:
        rf_model = joblib.load('random_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return rf_model, scaler
    except FileNotFoundError as e:
        st.error(f"Error: Model or scaler file not found. Please ensure 'random_forest_model.pkl' and 'scaler.pkl' are in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.stop()

# Function for heart attack risk prediction
def predict_heart_attack_risk(user_input, scaler, model):
    """Predict heart attack risk and provide detailed advice."""

    # Scale the user input
    try:
        scaled_input = scaler.transform(np.array(user_input).reshape(1, -1))
    except Exception as e:
        st.error(f"Error scaling input data: {e}. Please check your input values.")
        return None, None, None, None, None, None, None  # Return None values to prevent further errors

    # Make prediction
    try:
        prediction = model.predict(scaled_input)
    except Exception as e:
        st.error(f"Error during prediction: {e}. Please ensure the model is compatible with the input data.")
        return None, None, None, None, None, None, None  # Return None values to prevent further errors

    # Define messages based on prediction
    if prediction == 1:
        risk_level = "High"
        message = (
            "You are at high risk of a heart attack. Immediate consultation with a cardiologist is strongly advised. "
            "It is crucial to take this seriously and act promptly to mitigate potential health risks. 🚨"
        )

        precautions = [
            "1. Consult a Cardiologist Immediately: Schedule an appointment for a thorough evaluation. 👩‍⚕",
            "2. Modify Diet: Switch to a heart-healthy diet, reducing saturated and trans fats 🍔 and increasing fiber 🥦.",
            "3. Start Light Exercise: If cleared by a doctor, begin with low-impact activities. 🚶‍♀",
            "4. Strictly Adhere to Medications: Take all prescribed medications exactly as directed. 💊",
            "5. Monitor Symptoms Closely: Keep a close watch on any chest pain 💔, shortness of breath 😮‍💨, or unusual fatigue 😴."
        ]

        guidance = [
            "1. Follow Medication Plan: Adhere to your prescribed medication schedule without alterations. ⏰",
            "2. Adopt a Balanced Lifestyle: Focus on diet 🥗, stress management 🧘‍♀, and regular moderate activity 🚴.",
            "3. Schedule Regular Check-ups: Frequent check-ups help in monitoring your condition effectively. 📅",
            "4. Manage Stress Levels: Employ techniques like meditation 🧘 or yoga to lower stress. 😌",
            "5. Involve a Support System: Engage family and friends for emotional and practical support. 🫂"
        ]

        exercise = [
            "1. Consult Your Doctor: Get a tailored exercise plan from your healthcare provider. 👨‍⚕",
            "2. Start Slowly: Begin with very gentle activities such as walking 🚶 or light stretching.",
            "3. Build Gradually: Incrementally increase exercise intensity and duration over time. 📈",
            "4. Choose Enjoyable Activities: Opt for exercises you find pleasurable and motivating. 😄",
            "5. Listen To Your Body: Do not ignore any pain or discomfort; adjust accordingly. 🙏"
        ]

        diet = [
            "1. Prioritize Heart-Healthy Foods: Emphasize fruits 🍎, vegetables 🥦, lean proteins 🍗, and whole grains 🌾.",
            "2. Limit Unhealthy Fats: Minimize saturated and trans fats to protect your arteries. 🍟",
            "3. Reduce Sodium Intake: Lower sodium to manage blood pressure effectively. 🧂",
            "4. Stay Hydrated: Drink plenty of water to support overall cardiovascular function. 💧",
            "5. Avoid Processed Foods: Reduce or eliminate processed foods high in sugars and fats. 🍩"
        ]

        medications = [
            "1. Stick to Prescriptions: Strictly follow prescribed medication dosages and timings. 💊",
            "2. Understand Each Medication: Know the purpose and potential side effects of each medicine. ℹ",
            "3. Regular Review: Review all medications with your healthcare provider regularly. 👩‍⚕",
            "4. Do Not Self-Medicate: Avoid taking any other medications without consulting your doctor. 🚫",
            "5. Report Side Effects: Promptly report any side effects to your doctor. 🗣"
        ]

    else:
        risk_level = "Low"
        message = (
            "Your heart attack risk appears to be low. ✅ However, it is essential to maintain a healthy lifestyle 🏃‍♀ to ensure long-term cardiovascular health. ❤️"
        )

        precautions = [
            "1. Continue Regular Check-ups: Maintain routine appointments with your primary care physician. 👩‍⚕",
            "2. Monitor Health Metrics: Keep track of blood pressure, cholesterol, and blood sugar levels. 📊",
            "3. Maintain a Healthy Lifestyle: Continue with a balanced diet 🥗 and regular exercise routine 🏋‍♀.",
            "4. Stay Informed: Be proactive in learning about heart health and risk factors. 📚",
            "5. Plan for Emergencies: Have a plan in place in case of any sudden health issues. 📅"
        ]

        guidance = [
            "1. Maintain Balanced Diet: Ensure a diverse intake of fruits 🍎, vegetables 🥦, and whole grains 🌾.",
            "2. Stay Active Daily: Engage in at least 150 minutes of moderate aerobic exercise per week. 🏃",
            "3. Practice Stress Reduction: Use techniques like mindfulness 🧘 or yoga to manage stress. 😌",
            "4. Moderate Alcohol Intake: Adhere to recommended limits for alcohol consumption. 🍺",
            "5. Avoid Smoking: Refrain from smoking 🚭 to maintain optimal heart health. ❤️"
        ]

        exercise = [
            "1. Mix Up Activities: Include cardio 🏃, strength training 💪, and flexibility exercises.",
            "2. Be Consistent: Make physical activity a regular part of your daily routine. 📅",
            "3. Enjoy Your Workouts: Select activities that you find enjoyable and motivating. 😄",
            "4. Set Achievable Goals: Establish realistic fitness goals tailored to your ability. 🎯",
            "5. Listen to Your Body's Signals: Adjust your activity level based on how you feel. 🙏"
        ]

        diet = [
            "1. Focus on Whole Foods: Limit processed foods, emphasizing fruits 🍎, vegetables 🥦, and whole grains 🌾.",
            "2. Stay Hydrated: Drink plenty of water throughout the day. 💧",
            "3. Control Portions: Practice mindful eating to maintain a healthy weight. ⚖",
            "4. Plan Your Meals: Prepare meals in advance to make healthier choices. 🍱",
            "5. Seek Nutritional Advice: Consult a nutritionist for personalized guidance if needed. 👩‍⚕"
        ]

        medications = [
            "1. Consult Your Doctor Regularly: Discuss any health concerns or medication questions with them. 👩‍⚕",
            "2. Prioritize Prevention: Focus on lifestyle changes that can prevent heart issues. 💪",
            "3. Review Annually: Review your medications and health status annually with your doctor. 📅",
            "4. Stay Aware of Changes: Note any changes in how you feel after starting or stopping medications and report them promptly. 📝",
            "5. Be Informed: Educate yourself about your health conditions and medications. 📚"
        ]

    return risk_level, message, precautions, guidance, exercise, diet, medications

# Set up Streamlit app
st.title("Heart Attack Risk Prediction-❤️")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page:", ["Home", "About App & Modules", "Symptoms Information"])

if page == "Home":

    # Load model and scaler
    rf_model, scaler = load_model_and_scaler()

    # Check if model and scaler loaded successfully
    if not rf_model or not scaler:
        st.error("Model and scaler could not be loaded.")
        st.stop()

    # Collect input features from the user through Streamlit widgets
    st.sidebar.header("Patient Information")

    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)

    # Define the options for the selectbox
    sex_options = {"Female": 0, "Male": 1}
    sex_label = st.sidebar.selectbox("Select Sex", options=list(sex_options.keys()))
    sex = sex_options[sex_label]  # Set the value based on selection

    cp = st.sidebar.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
    trestbps = st.sidebar.number_input("Resting Blood Pressure (trestbps)", min_value=90, max_value=250, value=120)
    chol = st.sidebar.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600, value=200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "False" if x == 0 else "True")
    restecg = st.sidebar.selectbox("Resting ECG (restecg)", options=[0, 1, 2])
    thalach = st.sidebar.number_input("Max Heart Rate Achieved (thalach)", min_value=60,max_value=220,value=150)
    exang = st.sidebar.selectbox("Exercise-Induced Angina (exang)", options=[0 , 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.sidebar.number_input("ST Depression (oldpeak)", min_value=0., max_value=10., value=0.)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment (slope)", options=[0 , 1 , 2])
    ca = st.sidebar.number_input("Number of Major Vessels Colored by Fluoroscopy (ca)", min_value=0 , max_value=4 , value=0)
    thal = st.sidebar.selectbox("Thalassemia (thal)", options=[0 , 1 , 2 , 3])

    # Create a button to trigger the prediction
    if st.sidebar.button("Predict"):
        # Prepare the user input for prediction
        user_input = [age , sex , cp , trestbps , chol , fbs , restecg ,
                      thalach, exang ,oldpeak,
                      slope ,
                      ca ,
                      thal]

        # Call the prediction function
        risk_level , message , precautions , guidance , exercise , diet , medications = predict_heart_attack_risk(user_input , scaler , rf_model)

        # Check if the prediction was successful
        if risk_level is not None:
            # Display the prediction results
             # Correcting the Heading Here
            st.subheader("Prediction Results")

            # Using Markdown to make Risk Level more attractive
            if risk_level == "High":
                st.markdown(f"<h3 style='color:red;'>Risk Level: <span style='font-weight: bold; color: red;'>{risk_level}</span></h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color:green;'>Risk Level: <span style='font-weight: bold; color: green;'>{risk_level}</span></h3>", unsafe_allow_html=True)

            st.write(message)

            st.subheader("RECOMMENDATIONS:")

            st.markdown("PRECAUTIONS-")
            for rec in precautions:
                st.write(rec)

            st.markdown("---")

            st.markdown("GUIDANCE-")
            for rec in guidance:
                st.write(rec)

            st.markdown("---")

            st.markdown("EXERCISE-")
            for rec in exercise:
                st.write(rec)

            st.markdown("---")

            st.markdown("DIET-")
            for rec in diet:
                st.write(rec)

            st.markdown("---")

            st.markdown("MEDICATIONS")
            for rec in medications:
                st.write(rec)

elif page == "About App & Modules":
    # About App Section
    st.header("About This Application")
    image = Image.open("hq720.jpg")
    st.image(image, caption="Your Image",use_container_width=True)
    st.write("""
     This section showcases the performance of various machine learning models trained to predict the likelihood of heart disease. The models are evaluated based on metrics like *Accuracy, **Precision, **Recall, and **F1-Score*. Below, we explain each evaluation parameter and how it relates to model performance. 📈

    ### Evaluating Parameters:
    To assess the quality of a model, we use the following metrics:

    1. *Accuracy*: The proportion of correct predictions made by the model out of all predictions. 
        - A higher accuracy means the model is making more correct predictions overall.
    2. *Precision*: The proportion of true positive predictions (correctly predicted heart attack cases) out of all positive predictions made by the model.
        - Higher precision means the model is good at predicting positive cases and not making too many false positives.
    3. *Recall*: The proportion of true positive predictions (correct heart attack cases) out of all actual positive cases in the dataset.
        - A higher recall means the model is good at identifying heart attack cases and not missing too many.
    4. *F1-Score*: The harmonic mean of precision and recall, offering a balance between the two.
        - A higher F1-Score indicates that both precision and recall are high, meaning the model is both accurate and reliable.

    ### Confusion Matrix for Random Forest and Decision Tree:
    We also evaluate models using the *Confusion Matrix*, which helps understand the breakdown of predictions into:
    - *True Positives (TP)*: Correct predictions of a heart attack.
    - *True Negatives (TN)*: Correct predictions of no heart attack.
    - *False Positives (FP)*: Incorrect predictions of a heart attack when there is none.
    - *False Negatives (FN)*: Incorrect predictions of no heart attack when there is one.
    
    The confusion matrix shows how well the model distinguishes between the two classes (heart attack vs. no heart attack).

    ### Performance Metrics of Random Forest:
    *Accuracy*: 83.70%  
    Random Forest's confusion matrix shows that:
    
    - For *Class 0 (No Heart Attack)*, the model has a precision of 0.79 and a recall of 0.79, indicating a reasonable balance between predicting no heart attack and avoiding false positives.
    - For *Class 1 (Heart Attack)*, the precision and recall are both higher at 0.87, meaning the model is quite effective at identifying actual heart attack cases.
    
    Overall, the Random Forest model performs well with a *macro average* of 0.83 for precision, recall, and F1-score, and a *weighted average* of 0.84, showing good overall performance across both classes.

    ### Performance Metrics of Decision Tree:
    *Accuracy*: 83.70%  
    Decision Tree’s confusion matrix shows:
    
    - For *Class 0 (No Heart Attack)*, the precision is 0.88, but recall drops to 0.67, meaning that the model is good at predicting no heart attack but misses some cases.
    - For *Class 1 (Heart Attack)*, the precision is 0.82 and recall is 0.94, indicating that the model is very effective at identifying true heart attack cases, but may sometimes misclassify a few no-heart-attack cases as heart attack cases.

    The Decision Tree model's *macro average* is 0.81 for recall and 0.82 for F1-score, with a *weighted average* of 0.84, suggesting it performs well but with a slight bias towards predicting heart attacks.

    ### Performance Comparison of All Models:
    Below is the comparison of the performance of all the models used in this study:

    | Model                        | Accuracy (%) | Precision | Recall | F1 Score |
    |------------------------------|--------------|-----------|--------|----------|
    | Logistic Regression           | 82.22        | 0.82      | 0.82   | 0.82     |
    | Naive Bayes                   | 78.52        | 0.79      | 0.79   | 0.79     |
    | *Random Forest*             | *83.70*    | *0.84*  | *0.84* | *0.84* |
    | Extreme Gradient Boost        | 61.48        | 0.38      | 0.61   | 0.47     |
    | K-Nearest Neighbour           | 82.22        | 0.82      | 0.82   | 0.82     |
    | *Decision Tree*             | *83.70*    | *0.84*  | *0.84* | *0.83* |
    | Support Vector Machine        | 83.70        | 0.84      | 0.84   | 0.83     |

    ### Insights:
    - The *Random Forest, **Decision Tree, and **Support Vector Machine* models achieved the highest *accuracy (83.70%)*. This shows their strong ability to predict heart attack risk.
    - *Random Forest* stands out with a balanced performance across precision, recall, and F1-Score, making it a reliable model for heart attack prediction.
    - While *Extreme Gradient Boost* has a much lower accuracy (61.48%) and precision (0.38), it struggles with predicting heart attack cases accurately and is not ideal for this task.
    - *Logistic Regression* and *K-Nearest Neighbour* perform similarly to Random Forest, but slightly less accurate and precise.

    ### Conclusion:
    The *Random Forest* and *Decision Tree* models both perform exceptionally well and are our top picks for predicting heart attack risk. They balance accuracy, precision, recall, and F1-Score, providing a reliable prediction system for heart disease risk assessment. 🔍💓

    We encourage users to use these predictions and visualizations to make informed decisions about their health. With further improvements in data collection and model training, we aim to increase the accuracy of these models for better, more personalized health predictions in the future.
    """)

    # About Modules Section
    st.subheader("Modules Used")
    image = Image.open("istockphoto-1515913422-612x612.jpg")
    st.image(image, caption="Your Image",use_container_width=True)
    st.write("""
    - Risk Prediction Module: This module employs a Random Forest classifier trained on historical patient data.
      It predicts whether an individual is at high or low risk for heart attacks based on their input parameters.

    - Recommendation Module: After assessing risk levels,
      this module provides tailored recommendations regarding lifestyle changes,
      dietary adjustments,
      medication adherence,
      exercise plans,
      etc., aimed at improving cardiovascular health.

    - User Interface Module: This application features an intuitive user interface built using Streamlit,
      allowing users easy navigation through different sections including predictions,
      recommendations,
      educational content about symptoms,
      etc.

    - Data Handling Module: This module manages data preprocessing steps such as scaling inputs before feeding them into the model.
      It ensures that all inputs are standardized according to what the model expects.

    - Visualization Module: Although not implemented yet,
      this module can be used in future versions for visualizing user data trends over time or comparing different metrics.
    """)

elif page == "Symptoms Information":
    # Symptoms Information Section
    st.header("Heart Attack Symptoms")
    image = Image.open("190206-heart-attack-warning-signs-infographic-aha.webp")
    st.image(image, caption="Your Image",use_container_width=True)
    st.write("""
    Recognizing the symptoms of a heart attack is crucial for timely intervention.

    Common symptoms include:

    - Chest Pain or Discomfort: Often described as pressure or squeezing sensation.

    - Shortness of Breath: May occur with or without chest discomfort.

    - Pain or Discomfort in Other Areas: Such as arms (especially left arm), shoulder blades,
      neck jaw back stomach.

    - Nausea/Vomiting: Some individuals may experience stomach upset along with other symptoms.

    - Lightheadedness/Fainting: Feeling dizzy or faint can also indicate an issue related to heart health.

    If you experience any of these symptoms especially if they last more than few minutes seek immediate medical attention!

    Remember that symptoms can vary between individuals especially between men & women!
    """)
# Gemini AI Integration Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("HEARTLY ❤️")
ai_prompt = st.sidebar.text_area("Ask HEARTLY about WHAT DO YOU WANT TO ASK:")

if st.sidebar.button("🔍 ASK ME"):
    if ai_prompt:
        response = get_gemini_response(ai_prompt)
        st.sidebar.write("* HEARTLY Response:*")
        st.sidebar.write(response)
    else:
        st.sidebar.warning("⚠ Please enter a question.")
# Add a disclaimer at the bottom of each page
st.markdown(
    "<hr style='border:2px solid gray'>", unsafe_allow_html=True)  # Adding horizontal line before disclaimer
st.markdown(
    "<strong>Disclaimer:</strong> This app provides general predictions based on user input but should not be used as substitute for professional medical advice."
    + "<br>Consult with qualified healthcare provider regarding any health concerns.", unsafe_allow_html=True)
