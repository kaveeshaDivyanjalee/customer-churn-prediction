# 🚀 Customer Churn Prediction System | End-to-End ML Portfolio Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?logo=streamlit&logoColor=white)
![ML](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **AI-powered Customer Churn Prediction System** achieving **93.48% accuracy**, built using real telecom customer data and deployed as an interactive Streamlit web application.

---

## 🌐 Live Application

👉 **Try the App:**  
🔗 https://customer-churn-prediction-bczicccnktkymsfewgwurk.streamlit.app/

---

## 📌 Project Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses.  
This project predicts **which customers are likely to leave** using machine learning, enabling **early intervention and retention strategies**.

### 🔍 Key Highlights
- Trained on **15,000+ real telecom customer records**
- End-to-end ML pipeline: preprocessing → feature engineering → modeling → deployment
- Business-focused insights, not just predictions
- Fully deployed and production-ready

---

## 🎯 What This Project Demonstrates

✅ Machine Learning fundamentals  
✅ Feature engineering & model optimization  
✅ Model evaluation & interpretation  
✅ Data-driven business decision making  
✅ ML model deployment using Streamlit  

---

## 🧠 Machine Learning Details

### Model
- **Algorithm**: Random Forest Classifier
- **Hyperparameter Tuning**: GridSearchCV (5-fold cross-validation)
- **Class Balancing**: Handled for realistic churn distribution

### Performance

| Metric | Score |
|------|------|
| Accuracy | **93.48%** |
| ROC-AUC | **0.963** |
| Precision | **97%** |
| Recall | **78%** |

---

## 🧩 Feature Engineering

Created **29+ engineered features**, including:
- Tenure groups (new / loyal customers)
- Charge-per-service ratios
- Service engagement scores
- High-risk churn flags

### 🔝 Top Predictive Features
1. Customer Service Calls  
2. Contract Type  
3. Tenure  
4. Monthly Charges  
5. Tech Support Usage  

---

## 🌐 Application Features

### 🔮 Prediction Modes
- **Single Customer Prediction**
- **Batch Prediction via CSV Upload**

### 📊 Insights & Analytics
- Feature importance visualization
- Risk level classification (Low / Medium / High)
- Actionable retention recommendations

### 📥 Export
- Download prediction results as CSV
- Ready-to-use CSV templates provided

---

## 🏗️ System Architecture

Customer Data
↓
Feature Engineering
↓
Random Forest Model
↓
Churn Probability
↓
Streamlit Dashboard
↓
Business Insights & Recommendations


---

## 🛠️ Tech Stack

| Layer | Technology |
|-----|-----------|
| Language | Python |
| ML | Scikit-learn (Random Forest) |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Model Storage | Joblib |
| Deployment | Streamlit Cloud |

---

## 📁 Project Structure

customer-churn-prediction/
│
├── app/
│ └── churn_app.py # Streamlit web app
│
├── src/
│ └── churn_prediction.py # ML pipeline & training
│
├── models/
│ ├── churn_model.pkl
│ └── model_evaluation.png
│
├── data/
│ └── Telco_Customer_Churn_Expanded.xlsx
│
├── notebooks/
│ └── churn_analysis.ipynb
│
├── requirements.txt
└── README.md

---

## 🚀 Run Locally

```bash
# Clone repository
git clone https://github.com/kaveeshaDivyanjalee/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Train model
python src/churn_prediction.py

# Run app
python -m streamlit run app/churn_app.py
Open 👉 http://localhost:8501

💼 Business Impact

✔ Identify high-risk customers early
✔ Reduce churn by 30–50%
✔ Lower acquisition costs
✔ Improve customer lifetime value
✔ Support data-driven retention strategies

📊 Dataset

✔Source: Kaggle – Telecommunication Customer Churn Dataset

✔Records: 15,000+ customers

✔Target Variable: Churn (Yes / No)

🎓 Ideal For

✔Data Science portfolios

✔Machine Learning interviews

✔Final-year or capstone projects

✔Real-world ML deployment demos

👩‍💻 Author

Kaveesha Divyanjalee
🔗 GitHub: https://github.com/kaveeshaDivyanjalee

⭐ Support

If you found this project useful:

⭐ Star the repository

💬 Share feedback

🤝 Connect on LinkedIn

Built with ❤️ using Python, Machine Learning, and Streamlit
