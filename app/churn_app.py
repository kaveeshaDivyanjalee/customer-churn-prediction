"""
Customer Churn Prediction Dashboard
Interactive web application for predicting customer churn
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from churn_prediction import ChurnPredictor

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .high-risk {
        color: #d62728;
        font-weight: bold;
    }
    .low-risk {
        color: #2ca02c;
        font-weight: bold;
    }
    .stButton button {
        width: 100%;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        predictor = ChurnPredictor()
        # Dynamic path resolution - works both locally and in cloud
        model_path = Path(__file__).parent.parent / 'models' / 'churn_model.pkl'
        predictor.load_model(str(model_path))
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please make sure the model is trained first. Run the notebook or churn_prediction.py to create the model.")
        return None

# Initialize
predictor = load_model()

# Header
st.markdown('<h1 class="main-header">📊 Customer Churn Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📋 Navigation")
    page = st.radio(
        "Choose a page:",
        ["🔮 Single Prediction", "📊 Batch Predictions", "📈 Model Insights", "ℹ️ About"]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.metric("Accuracy", "93.48%")
    st.metric("ROC-AUC", "0.9632")
    st.metric("Precision", "97%")
    
    if predictor and hasattr(predictor, 'trained_features'):
        st.markdown("---")
        st.markdown("### 🔧 Model Info")
        st.info(f"Trained with {len(predictor.trained_features)} features")

# Helper function to calculate num_services
def calculate_num_services(df):
    """Calculate number of services from individual service columns"""
    df = df.copy()
    
    # Initialize num_services column
    df['num_services'] = 0
    
    # List of service columns to check
    service_columns = [
        'PhoneService', 'online_security', 'OnlineBackup', 
        'DeviceProtection', 'tech_support', 'StreamingTV', 'StreamingMovies'
    ]
    
    # Count "Yes" for each service
    for col in service_columns:
        if col in df.columns:
            # Count as service if it's "Yes" (ignore "No" and "No internet service"/"No phone service")
            df['num_services'] += (df[col] == 'Yes').astype(int)
    
    # Add MultipleLines as an additional service if PhoneService is Yes
    if 'MultipleLines' in df.columns and 'PhoneService' in df.columns:
        df['num_services'] += ((df['PhoneService'] == 'Yes') & 
                              (df['MultipleLines'] == 'Yes')).astype(int)
    
    # Add internet service as a service if not "No"
    if 'internet_service' in df.columns:
        df['num_services'] += (df['internet_service'] != 'No').astype(int)
    
    # Create 'age' column from SeniorCitizen if not present
    if 'age' not in df.columns and 'SeniorCitizen' in df.columns:
        df['age'] = np.where(
            df['SeniorCitizen'] == 1,
            np.random.randint(65, 80, len(df)),
            np.random.randint(18, 64, len(df))
        )
    
    return df

def prepare_features_for_model(df, predictor):
    """
    Prepare features to match exactly what the model expects.
    Ensures all features are present and in the correct order.
    """
    if predictor is None:
        return df
    
    # First, ensure we have all required features
    df = calculate_num_services(df)
    
    # Apply feature engineering
    df_engineered = predictor.feature_engineering(df)
    
    # Get the features the model was trained with
    if hasattr(predictor, 'trained_features') and predictor.trained_features:
        # Create a new DataFrame with the correct feature order
        final_df = pd.DataFrame(index=df_engineered.index)
        
        for feature in predictor.trained_features:
            if feature in df_engineered.columns:
                final_df[feature] = df_engineered[feature]
            else:
                # Add missing feature with default value
                st.warning(f"Feature '{feature}' not found in input data. Using default value.")
                if feature in ['age', 'tenure_months', 'monthly_charges', 'total_charges', 
                               'num_services', 'customer_service_calls', 'avg_monthly_charge',
                               'charge_per_service', 'service_engagement', 'SeniorCitizen', 'high_risk']:
                    final_df[feature] = 0  # Numerical/binary defaults
                else:
                    final_df[feature] = 'No'  # Categorical defaults
        
        # Reorder columns to match training order
        final_df = final_df[predictor.trained_features]
        return final_df
    
    return df_engineered

# ============================================
# PAGE 1: Single Prediction
# ============================================
if page == "🔮 Single Prediction":
    st.header("🔮 Predict Individual Customer Churn")
    st.markdown("Enter customer details to predict churn probability")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📋 Personal Information")
        customer_id = st.text_input("Customer ID", "CUST-001")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure_months = st.slider("Tenure (months)", 0, 72, 12)
        
    with col2:
        st.subheader("📞 Services & Internet")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        
    with col3:
        st.subheader("💳 Billing & Entertainment")
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, 5.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0, 100.0)
        customer_service_calls = st.slider("Customer Service Calls", 0, 15, 2)
    
    # Predict button
    if st.button("🎯 Predict Churn", type="primary", use_container_width=True):
        if predictor is None:
            st.error("Model not loaded. Please train the model first.")
        else:
            try:
                # Create dataframe with ALL features the model expects
                customer_data = pd.DataFrame({
                    'customer_id': [customer_id],
                    'gender': [gender],
                    'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
                    'Partner': [partner],
                    'Dependents': [dependents],
                    'tenure': [tenure_months],
                    'PhoneService': [phone_service],
                    'MultipleLines': [multiple_lines],
                    'InternetService': [internet_service],
                    'OnlineSecurity': [online_security],
                    'OnlineBackup': [online_backup],
                    'DeviceProtection': [device_protection],
                    'TechSupport': [tech_support],
                    'StreamingTV': [streaming_tv],
                    'StreamingMovies': [streaming_movies],
                    'Contract': [contract_type],
                    'PaperlessBilling': [paperless_billing],
                    'PaymentMethod': [payment_method],
                    'MonthlyCharges': [monthly_charges],
                    'TotalCharges': [total_charges],
                    'customer_service_calls': [customer_service_calls]
                })
                
                # Rename columns to match training data
                customer_data = customer_data.rename(columns={
                    'tenure': 'tenure_months',
                    'Contract': 'contract_type',
                    'PaymentMethod': 'payment_method',
                    'InternetService': 'internet_service',
                    'OnlineSecurity': 'online_security',
                    'TechSupport': 'tech_support',
                    'MonthlyCharges': 'monthly_charges',
                    'TotalCharges': 'total_charges'
                })
                
                # Prepare features to match exactly what the model expects
                customer_data_prepared = prepare_features_for_model(customer_data, predictor)
                
                # Debug info (optional - can be commented out)
                with st.expander("🔍 Debug Info (Feature Preparation)"):
                    st.write("Original features:", list(customer_data.columns))
                    if hasattr(predictor, 'trained_features'):
                        st.write("Model expects:", predictor.trained_features)
                    st.write("Prepared features:", list(customer_data_prepared.columns))
                
                # Preprocess
                X, _ = predictor.preprocess_data(customer_data_prepared, is_training=False)
                
                # Predict
                prediction = predictor.model.predict(X)[0]
                probability = predictor.model.predict_proba(X)[0, 1]
                
                # Display results
                st.markdown("---")
                st.header("📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Churn Prediction",
                        "WILL CHURN ⚠️" if prediction == 1 else "WON'T CHURN ✅",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Churn Probability",
                        f"{probability*100:.2f}%",
                        delta=f"{(probability-0.5)*100:+.1f}% vs baseline"
                    )
                
                with col3:
                    risk_level = "HIGH RISK 🔴" if probability > 0.7 else "MEDIUM RISK 🟡" if probability > 0.4 else "LOW RISK 🟢"
                    st.metric("Risk Level", risk_level)
                
                # Progress bar
                st.markdown("### Churn Probability Meter")
                st.progress(float(probability))
                
                # Show calculated num_services
                num_services_val = int(customer_data_prepared['num_services'].iloc[0]) if 'num_services' in customer_data_prepared.columns else 0
                st.info(f"📊 **Services Summary**: Customer has {num_services_val} active services")
                
                # Recommendations
                st.markdown("---")
                st.header("💡 Recommendations")
                
                if prediction == 1:
                    st.error("⚠️ **HIGH CHURN RISK - Immediate Action Required**")
                    
                    recommendations = []
                    
                    if customer_service_calls > 5:
                        recommendations.append("🔴 **Critical**: Customer has made many service calls. Assign dedicated support rep.")
                    
                    if contract_type == "Month-to-month":
                        recommendations.append("📝 **Contract**: Offer incentive to upgrade to yearly contract (15-20% discount).")
                    
                    if tenure_months < 12:
                        recommendations.append("🆕 **New Customer**: Implement onboarding program to improve experience.")
                    
                    if monthly_charges > 80:
                        recommendations.append("💰 **High Charges**: Review pricing and show value proposition.")
                    
                    if internet_service == "Fiber optic" and online_security == "No":
                        recommendations.append("🛡️ **Security**: Offer free online security trial for 3 months.")
                    
                    if tech_support == "No":
                        recommendations.append("🔧 **Tech Support**: Offer complimentary tech support for first 3 months.")
                    
                    if not recommendations:
                        recommendations.append("📞 Proactive outreach to understand concerns and offer retention package.")
                    
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                else:
                    st.success("✅ **LOW CHURN RISK - Customer is likely to stay**")
                    st.markdown("- Continue excellent service")
                    st.markdown("- Consider upselling additional services")
                    st.markdown("- Ask for referrals")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Make sure your model is trained with all the required features.")
                
                # Detailed error info
                with st.expander("🔍 Show Detailed Error"):
                    st.code(str(e))

# ============================================
# PAGE 2: Batch Predictions
# ============================================
elif page == "📊 Batch Predictions":
    st.header("📊 Batch Prediction - Upload Multiple Customers")
    
    st.markdown("""
    Upload a CSV file with customer data to predict churn for multiple customers at once.
    
    **Required columns** (matching the training data):
    """)
    
    # Show required columns in a cleaner format
    required_cols = [
        'customer_id', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure_months', 'PhoneService', 'MultipleLines', 'internet_service',
        'online_security', 'OnlineBackup', 'DeviceProtection', 'tech_support',
        'StreamingTV', 'StreamingMovies', 'contract_type', 'PaperlessBilling',
        'payment_method', 'monthly_charges', 'total_charges', 'customer_service_calls'
    ]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Required Columns:**")
        for col in required_cols[:11]:
            st.markdown(f"- `{col}`")
    with col2:
        st.markdown("**Required Columns (cont.):**")
        for col in required_cols[11:]:
            st.markdown(f"- `{col}`")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Ensure required columns exist
            missing_columns = [col for col in required_cols if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.info("Please download the sample template below and format your data accordingly.")
            else:
                st.success(f"✅ Loaded {len(df)} customers")
                
                # Show preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(10))
                
                # Predict button
                if st.button("🎯 Predict All", type="primary"):
                    if predictor is None:
                        st.error("Model not loaded. Please train the model first.")
                    else:
                        with st.spinner("Making predictions..."):
                            try:
                                # Prepare features to match exactly what the model expects
                                df_prepared = prepare_features_for_model(df, predictor)
                                
                                # Preprocess
                                X, _ = predictor.preprocess_data(df_prepared, is_training=False)
                                
                                # Predict
                                predictions = predictor.model.predict(X)
                                probabilities = predictor.model.predict_proba(X)[:, 1]
                                
                                # Add results to dataframe
                                results_df = df.copy()
                                results_df['churn_prediction'] = predictions
                                results_df['churn_probability'] = probabilities
                                results_df['risk_level'] = pd.cut(
                                    probabilities,
                                    bins=[0, 0.3, 0.7, 1.0],
                                    labels=['Low', 'Medium', 'High']
                                )
                                
                                # Add num_services to results for transparency
                                if 'num_services' in df_prepared.columns:
                                    results_df['num_services'] = df_prepared['num_services']
                                
                                # Show results
                                st.success("✅ Predictions completed!")
                                
                                # Summary metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Total Customers", len(results_df))
                                
                                with col2:
                                    churn_count = (predictions == 1).sum()
                                    st.metric("Predicted Churners", churn_count)
                                
                                with col3:
                                    churn_rate = churn_count / len(results_df) * 100
                                    st.metric("Churn Rate", f"{churn_rate:.1f}%")
                                
                                with col4:
                                    high_risk = (probabilities > 0.7).sum()
                                    st.metric("High Risk", high_risk)
                                
                                # Results table
                                st.subheader("📊 Detailed Results")
                                
                                # Select columns to display
                                display_cols = ['customer_id', 'tenure_months', 'monthly_charges']
                                if 'num_services' in results_df.columns:
                                    display_cols.append('num_services')
                                display_cols.extend(['churn_prediction', 'churn_probability', 'risk_level'])
                                
                                st.dataframe(
                                    results_df[display_cols].style.format({
                                        'churn_probability': '{:.2%}',
                                        'monthly_charges': '${:.2f}'
                                    })
                                )
                                
                                # Download button
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results CSV",
                                    data=csv,
                                    file_name="churn_predictions.csv",
                                    mime="text/csv"
                                )
                                
                                # Visualization
                                st.subheader("📈 Churn Risk Distribution")
                                
                                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                                
                                # Risk level distribution
                                risk_counts = results_df['risk_level'].value_counts()
                                colors = ['green', 'orange', 'red']
                                axes[0].bar(risk_counts.index, risk_counts.values, color=colors)
                                axes[0].set_title('Customers by Risk Level')
                                axes[0].set_ylabel('Count')
                                
                                # Probability distribution
                                axes[1].hist(probabilities, bins=30, edgecolor='black', alpha=0.7)
                                axes[1].set_title('Churn Probability Distribution')
                                axes[1].set_xlabel('Probability')
                                axes[1].set_ylabel('Count')
                                axes[1].axvline(0.5, color='red', linestyle='--', label='Threshold')
                                axes[1].legend()
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                            except Exception as e:
                                st.error(f"Error during prediction: {str(e)}")
                                st.info("Make sure your data is properly formatted and the model is trained.")
                                
                                # Detailed error info
                                with st.expander("🔍 Show Detailed Error"):
                                    st.code(str(e))
                
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
    
    # Download template
    st.markdown("---")
    st.subheader("📥 Download Sample Template")
    
    # Create sample data with all required columns
    sample_data = pd.DataFrame({
        'customer_id': ['CUST-001', 'CUST-002', 'CUST-003'],
        'gender': ['Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0],
        'Partner': ['Yes', 'Yes', 'No'],
        'Dependents': ['No', 'Yes', 'No'],
        'tenure_months': [6, 48, 2],
        'PhoneService': ['Yes', 'Yes', 'No'],
        'MultipleLines': ['Yes', 'No', 'No phone service'],
        'internet_service': ['Fiber optic', 'DSL', 'No'],
        'online_security': ['No', 'Yes', 'No internet service'],
        'OnlineBackup': ['No', 'Yes', 'No internet service'],
        'DeviceProtection': ['No', 'Yes', 'No internet service'],
        'tech_support': ['No', 'Yes', 'No internet service'],
        'StreamingTV': ['Yes', 'No', 'No internet service'],
        'StreamingMovies': ['Yes', 'Yes', 'No internet service'],
        'contract_type': ['Month-to-month', 'Two year', 'Month-to-month'],
        'PaperlessBilling': ['Yes', 'No', 'Yes'],
        'payment_method': ['Electronic check', 'Credit card (automatic)', 'Electronic check'],
        'monthly_charges': [120.0, 75.0, 95.0],
        'total_charges': [720.0, 3600.0, 190.0],
        'customer_service_calls': [7, 2, 8]
    })
    
    csv = sample_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=csv,
        file_name="customer_churn_template.csv",
        mime="text/csv"
    )

# ============================================
# PAGE 3: Model Insights
# ============================================
elif page == "📈 Model Insights":
    st.header("📈 Model Performance & Insights")
    
    if predictor is None:
        st.error("Model not loaded. Please train the model first.")
    else:
        # Performance metrics
        st.subheader("🎯 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", "93.48%", "Excellent")
        
        with col2:
            st.metric("ROC-AUC", "0.9632", "Near Perfect")
        
        with col3:
            st.metric("Precision", "97%", "Very High")
        
        with col4:
            st.metric("Recall", "78%", "Good")
        
        st.markdown("---")
        
        # Feature importance
        st.subheader("🔑 Feature Importance")
        st.markdown("**Top factors influencing customer churn:**")
        
        if predictor.feature_importance is not None:
            feature_importance = predictor.feature_importance.head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(feature_importance)), feature_importance['importance'], color='steelblue')
            ax.set_yticks(range(len(feature_importance)))
            ax.set_yticklabels(feature_importance['feature'])
            ax.set_xlabel('Importance')
            ax.set_title('Top 10 Most Important Features')
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show feature importance table
            with st.expander("📋 View All Features"):
                st.dataframe(predictor.feature_importance)
        else:
            st.info("Feature importance not available. Please train the model first.")
        
        st.markdown("---")
        
        # Model info
        st.subheader("🤖 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Details:**")
            st.markdown(f"- Algorithm: Random Forest")
            st.markdown(f"- Number of trees: {predictor.model.n_estimators if hasattr(predictor.model, 'n_estimators') else 'N/A'}")
            st.markdown(f"- Max depth: {predictor.model.max_depth if hasattr(predictor.model, 'max_depth') else 'N/A'}")
        
        with col2:
            st.markdown("**Feature Details:**")
            st.markdown(f"- Total features: {len(predictor.feature_importance) if predictor.feature_importance is not None else 'N/A'}")
            if hasattr(predictor, 'trained_features'):
                st.markdown(f"- Expected features: {len(predictor.trained_features)}")
        
        st.markdown("---")
        
        # Business insights
        st.subheader("💼 Business Insights")
        
        insights = [
            {
                "title": "🔴 Customer Service Calls",
                "insight": "The #1 predictor of churn! Customers with frequent service calls are at highest risk.",
                "action": "Improve first-call resolution, train support staff, implement proactive outreach."
            },
            {
                "title": "⚠️ Contract Type Importance",
                "insight": "Month-to-month contracts have significantly higher churn rates compared to yearly contracts.",
                "action": "Offer incentives for customers to switch to longer-term contracts."
            },
            {
                "title": "💰 Monthly Charges Impact",
                "insight": "High monthly charges correlate with churn, especially without perceived value.",
                "action": "Review pricing tiers, offer loyalty discounts, communicate value clearly."
            },
            {
                "title": "📞 Service Quality Indicators",
                "insight": "Customers without tech support, online security, or device protection are more likely to churn.",
                "action": "Bundle essential services and offer them as value-added features."
            },
            {
                "title": "👨‍👩‍👧‍👦 Demographic Factors",
                "insight": "Senior citizens without partners and with paperless billing show different churn patterns.",
                "action": "Create targeted retention programs for different demographic segments."
            }
        ]
        
        for insight in insights:
            with st.expander(insight['title']):
                st.markdown(f"**Insight**: {insight['insight']}")
                st.markdown(f"**Recommended Action**: {insight['action']}")

# ============================================
# PAGE 4: About
# ============================================
elif page == "ℹ️ About":
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### 🎯 Customer Churn Prediction System
    
    This application uses machine learning to predict which customers are likely to churn (cancel their service).
    
    #### 🤖 Model Details
    - **Algorithm**: Random Forest Classifier
    - **Training Data**: Real telecom customer data (Telco Customer Churn dataset)
    - **Features**: 29+ engineered features
    - **Performance**: 93.48% accuracy, 0.9632 ROC-AUC
    - **Optimization**: GridSearchCV with hyperparameter tuning
    
    #### 📊 Key Features
    ✅ **Single Customer Prediction**: Enter individual customer details for instant churn prediction  
    ✅ **Batch Processing**: Upload CSV files for bulk predictions  
    ✅ **Feature Engineering**: Advanced feature creation from raw data  
    ✅ **Actionable Insights**: Business recommendations based on predictions  
    ✅ **Model Transparency**: Feature importance and performance metrics  
    ✅ **Data Export**: Download prediction results for further analysis  
    
    #### 🛠️ Technology Stack
    - **Backend**: Python, Scikit-learn, Pandas, NumPy
    - **Frontend**: Streamlit
    - **Visualization**: Matplotlib, Seaborn
    - **Model**: Random Forest with hyperparameter optimization
    
    #### 📈 Business Impact
    - Identify at-risk customers before they leave
    - Prioritize retention efforts on high-value customers
    - Reduce churn by 30-50% with proactive interventions
    - Save millions in customer acquisition costs
    
    #### 👨‍💻 Developer
    Built as a demonstration of end-to-end machine learning and deployment capabilities.
    
    ---
    
    ### 📞 Support
    For questions or issues, please contact your system administrator.
    
    **Last Updated**: February 2025  
    **Version**: 2.1.0
    """)
    
    st.markdown("---")
    
    # Data requirements
    st.subheader("📋 Data Requirements")
    st.markdown("""
    The model expects the following features:
    
    1. **Demographic Information**: Gender, SeniorCitizen, Partner, Dependents
    2. **Account Information**: Tenure, Contract type, Paperless billing, Payment method
    3. **Services**: Phone service, Internet service, Multiple lines, Security services
    4. **Billing**: Monthly charges, Total charges
    5. **Support**: Customer service calls
    
    **Note**: The model requires all these features to make accurate predictions.
    """)
    
    # Troubleshooting
    with st.expander("🔧 Troubleshooting Guide"):
        st.markdown("""
        ### Common Issues and Solutions:
        
        **1. Model not loading:**
        - Make sure `churn_model.pkl` exists in the `models/` folder
        - Run `python src/churn_prediction.py` to train a new model
        
        **2. Feature mismatch errors:**
        - The model expects specific features in specific order
        - Use the sample template for batch predictions
        - For single predictions, ensure all fields are filled
        
        **3. Prediction errors:**
        - Check that all input values are valid
        - Ensure categorical values match the expected options
        - Try retraining the model with the latest code
        
        **4. Performance issues:**
        - Reduce hyperparameter tuning for faster training
        - Use smaller datasets for testing
        - Ensure sufficient system resources
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Customer Churn Prediction System v2.1 | Powered by Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)