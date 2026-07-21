# Bengaluru Real Estate Price Predictor

An AI-powered web application that estimates real estate market values in Bengaluru using tuned machine learning models.

## 🔗 Live Application Link
[**Click here to view the live app!**](https://bengaluru-real-estate-ai-nmzgaskfyn7ruycdej3hfl.streamlit.app/)

---

## 📌 Project Overview
Predicting property prices in volatile housing markets is challenging due to pricing anomalies and non-linear variables. This project provides an automated machine learning framework trained on 13,000+ listings in Bengaluru. It bypasses market noise to deliver accurate cost evaluations based on square footage, room counts, and geographic location.

---

## 🏗️ System Architecture & Data Flow


```text
[User Input (Streamlit UI Widgets)] 
          │
          ▼
[Pre-Processing Pipeline (StandardScaler)]
          │
          ▼
[Tuned Ridge Regression Model (.pickle)]
          │
          ▼
[Real-Time Output Rendered on UI]

---
```

# Local installation and setup

## Clone the repository:

git clone [https://github.com/Manik-netizenai.git](https://github.com/Manik-netizen/bengaluru-real-estate-ai.git) 
cd bengaluru-real-estate-ai

## Install dependencies:

pip install -r requirements.txt

## Launch the Streamlit application:

streamlit run app.py

## Access UI: The app will automatically open in your default web browser at http://localhost:8501.