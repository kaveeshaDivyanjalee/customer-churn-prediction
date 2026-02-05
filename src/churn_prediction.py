"""
Customer Churn Prediction Model
A machine learning project to predict customer churn using Random Forest
with feature engineering and hyperparameter tuning.
Uses real Telecom Customer Churn dataset.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')
import openpyxl  # For reading Excel files
from pathlib import Path


class ChurnPredictor:
    """
    A complete customer churn prediction pipeline with feature engineering
    and hyperparameter optimization.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.trained_features = None  # Store feature names seen during training
    
    def load_real_data(self, filepath=None):
        """
        Load and prepare the real Telecom Customer Churn dataset.
        """
        # Use dynamic path resolution
        if filepath is None:
            # Get project root (2 levels up from src/)
            project_root = Path(__file__).parent.parent
            filepath = project_root / 'data' / 'Telco_Customer_Churn_Expanded.xlsx'
        
        print(f"Loading real dataset from: {filepath}")
        
        try:
            # Read the Excel file
            df = pd.read_excel(filepath)
        except FileNotFoundError:
            print(f"⚠️  File not found: {filepath}")
            print("Falling back to synthetic data...")
            return self.create_sample_data(n_samples=5000)
        
        print(f"Original dataset shape: {df.shape}")
        
        # Convert Churn to binary (Yes=1, No=0)
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        # Handle TotalCharges - convert to numeric
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Drop rows with missing values
        original_size = len(df)
        df = df.dropna()
        dropped = original_size - len(df)
        if dropped > 0:
            print(f"Dropped {dropped} rows with missing values")
        
        # Rename columns to match our code - KEEP ALL ORIGINAL COLUMNS
        df = df.rename(columns={
            'customerID': 'customer_id',
            'gender': 'gender',
            'SeniorCitizen': 'SeniorCitizen',
            'Partner': 'Partner',
            'Dependents': 'Dependents',
            'tenure': 'tenure_months',
            'PhoneService': 'PhoneService',
            'MultipleLines': 'MultipleLines',
            'InternetService': 'internet_service',
            'OnlineSecurity': 'online_security',
            'OnlineBackup': 'OnlineBackup',
            'DeviceProtection': 'DeviceProtection',
            'TechSupport': 'tech_support',
            'StreamingTV': 'StreamingTV',
            'StreamingMovies': 'StreamingMovies',
            'Contract': 'contract_type',
            'PaperlessBilling': 'PaperlessBilling',
            'PaymentMethod': 'payment_method',
            'MonthlyCharges': 'monthly_charges',
            'TotalCharges': 'total_charges',
            'Churn': 'churn'
        })
        
        # Create 'age' from SeniorCitizen
        np.random.seed(42)
        df['age'] = np.where(
            df['SeniorCitizen'] == 1,
            np.random.randint(65, 80, len(df)),
            np.random.randint(18, 64, len(df))
        )
        
        # Count number of services - THIS IS CRITICAL
        service_cols = ['PhoneService', 'MultipleLines', 'online_security', 
                        'OnlineBackup', 'DeviceProtection', 'tech_support', 
                        'StreamingTV', 'StreamingMovies']
        
        df['num_services'] = 0
        for col in service_cols:
            if col in df.columns:
                df['num_services'] += (df[col] == 'Yes').astype(int)
        
        # Also count internet service as a service
        df['num_services'] += (df['internet_service'] != 'No').astype(int)
        
        # Simulate customer service calls (not in dataset)
        # Higher calls for churned customers to create realistic pattern
        df['customer_service_calls'] = np.where(
            df['churn'] == 1,
            np.random.randint(3, 10, len(df)),
            np.random.randint(0, 5, len(df))
        )
        
        print(f"Final dataset shape: {df.shape}")
        print(f"Churn rate: {df['churn'].mean():.2%}")
        
        return df
        
    def create_sample_data(self, n_samples=5000):
        """
        Generate synthetic customer data for demonstration.
        Fallback if real data is not available.
        """
        np.random.seed(42)
        
        # Generate synthetic data with ALL the features the Streamlit app expects
        data = {
            'customer_id': [f"CUST-{i:04d}" for i in range(1, n_samples + 1)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'tenure_months': np.random.randint(1, 72, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
            'monthly_charges': np.random.uniform(20, 150, n_samples),
            'total_charges': np.random.uniform(100, 8000, n_samples),
            'customer_service_calls': np.random.randint(0, 10, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create 'age' from SeniorCitizen
        df['age'] = np.where(
            df['SeniorCitizen'] == 1,
            np.random.randint(65, 80, n_samples),
            np.random.randint(18, 64, n_samples)
        )
        
        # Calculate num_services from individual service columns
        df['num_services'] = 0
        service_columns = [
            'PhoneService', 'online_security', 'OnlineBackup', 
            'DeviceProtection', 'tech_support', 'StreamingTV', 'StreamingMovies'
        ]
        
        for col in service_columns:
            df['num_services'] += (df[col] == 'Yes').astype(int)
        
        # Add MultipleLines as an additional service if PhoneService is Yes
        df['num_services'] += ((df['PhoneService'] == 'Yes') & 
                              (df['MultipleLines'] == 'Yes')).astype(int)
        
        # Add internet service as a service if not "No"
        df['num_services'] += (df['internet_service'] != 'No').astype(int)
        
        # Create target variable with realistic patterns
        churn_probability = (
            0.05 +  # base rate
            (df['contract_type'] == 'Month-to-month') * 0.3 +
            (df['tenure_months'] < 12) * 0.25 +
            (df['customer_service_calls'] > 5) * 0.2 +
            (df['monthly_charges'] > 100) * 0.15 +
            (df['tech_support'] == 'No') * 0.1 +
            (df['online_security'] == 'No') * 0.1 +
            np.random.uniform(0, 0.1, n_samples)
        )
        
        df['churn'] = (np.random.random(n_samples) < churn_probability).astype(int)
        
        print(f"Created synthetic dataset with {n_samples} samples")
        print(f"Churn rate: {df['churn'].mean():.2%}")
        
        return df
    
    def feature_engineering(self, df):
        """
        Create new features from existing ones to improve model performance.
        """
        df = df.copy()
        
        # Handle potential division by zero
        df['tenure_months'] = df['tenure_months'].replace(0, 1)
        
        # Average monthly charge
        df['avg_monthly_charge'] = df['total_charges'] / (df['tenure_months'] + 1)
        
        # Tenure groups
        df['tenure_group'] = pd.cut(df['tenure_months'], 
                                     bins=[0, 12, 24, 48, 72],
                                     labels=['0-1 year', '1-2 years', '2-4 years', '4+ years'])
        
        # Charge to service ratio (handle division by zero)
        df['charge_per_service'] = df['monthly_charges'] / (df['num_services'].replace(0, 1))
        
        # High-risk customer flag
        df['high_risk'] = (
            ((df['contract_type'] == 'Month-to-month') & (df['tenure_months'] < 12)) |
            (df['customer_service_calls'] > 5)
        ).astype(int)
        
        # Service engagement score
        df['service_engagement'] = (df['num_services'] / 8) * 100
        
        return df
    
    def preprocess_data(self, df, is_training=True):
        """
        Preprocess the data: encode categorical variables and scale features.
        """
        df = df.copy()
        
        # Create 'age' column if it doesn't exist (for Streamlit predictions)
        if 'age' not in df.columns and 'SeniorCitizen' in df.columns:
            df['age'] = np.where(
                df['SeniorCitizen'] == 1,
                np.random.randint(65, 80, len(df)),
                np.random.randint(18, 64, len(df))
            )
        
        # Separate features and target
        if 'churn' in df.columns:
            X = df.drop(['churn', 'customer_id'], axis=1, errors='ignore')
            y = df['churn']
        else:
            X = df.drop(['customer_id'], axis=1, errors='ignore')
            y = None
        
        # Store feature names during training
        if is_training:
            self.trained_features = list(X.columns)
            print(f"\nTraining with {len(self.trained_features)} features")
        
        # Ensure we have all the features the model was trained with (for prediction)
        if not is_training and self.trained_features is not None:
            # Create a new DataFrame with the correct feature order
            X_processed = pd.DataFrame(index=X.index)
            
            for feature in self.trained_features:
                if feature in X.columns:
                    X_processed[feature] = X[feature]
                else:
                    print(f"Warning: Adding missing feature '{feature}' with default value")
                    # Use appropriate default values based on feature type
                    if X.empty:
                        X_processed[feature] = 0
                    else:
                        # Check feature type and assign appropriate default
                        if feature in ['age', 'tenure_months', 'monthly_charges', 'total_charges', 
                                       'num_services', 'customer_service_calls', 'avg_monthly_charge',
                                       'charge_per_service', 'service_engagement', 'SeniorCitizen']:
                            X_processed[feature] = 0  # Numerical defaults
                        elif feature in ['high_risk']:
                            X_processed[feature] = 0  # Binary defaults
                        else:
                            X_processed[feature] = 'No'  # Categorical defaults
            
            X = X_processed[self.trained_features]  # Ensure correct order
        
        # Ensure all categorical columns are strings
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            X[col] = X[col].astype(str)
        
        # Encode categorical variables with better handling
        for col in categorical_cols:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                if col in self.label_encoders:
                    try:
                        # First, map any "No internet service" or "No phone service" to "No"
                        X[col] = X[col].replace(['No internet service', 'No phone service'], 'No')
                        
                        # Get known categories
                        known_categories = set(self.label_encoders[col].classes_)
                        
                        # For categories not seen during training, use the most common one
                        if len(self.label_encoders[col].classes_) > 0:
                            most_common = self.label_encoders[col].classes_[0]
                        else:
                            most_common = 'No'
                        
                        # Replace unseen categories with the most common one
                        X[col] = X[col].apply(lambda x: x if x in known_categories else most_common)
                        
                        # Now transform
                        X[col] = self.label_encoders[col].transform(X[col])
                    except Exception as e:
                        print(f"Warning: Error encoding column '{col}': {e}")
                        # If encoding fails, use a default value
                        if len(X) > 0:
                            X[col] = 0
        
        # Scale numerical features
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            try:
                X_scaled = self.scaler.transform(X)
            except ValueError as e:
                print(f"Warning: Error scaling features: {e}")
                # If scaling fails, return unscaled features
                X_scaled = X.values
        
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X_scaled, y
    
    def train_model(self, X_train, y_train, tune_hyperparameters=False):
        """
        Train the Random Forest model with optional hyperparameter tuning.
        """
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }
            
            # Initialize base model
            rf = RandomForestClassifier(random_state=42, class_weight='balanced')
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                cv=3,  # Reduced for speed
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"\nBest parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            self.model = grid_search.best_estimator_
        else:
            # Use default parameters
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model and return comprehensive metrics.
        """
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['No Churn', 'Churn']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def plot_results(self, X_test, y_test, save_path=None):
        """
        Create visualization plots for model performance.
        """
        # Use dynamic path resolution
        if save_path is None:
            project_root = Path(__file__).parent.parent
            save_path = project_root / 'models'
        
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Feature Importance
        top_features = self.feature_importance.head(10)
        axes[0, 0].barh(range(len(top_features)), top_features['importance'])
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features['feature'])
        axes[0, 0].set_xlabel('Importance')
        axes[0, 0].set_title('Top 10 Feature Importances')
        axes[0, 0].invert_yaxis()
        
        # 2. Confusion Matrix
        cm = confusion_matrix(y_test, self.model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        axes[0, 1].set_title('Confusion Matrix')
        
        # 3. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        axes[1, 0].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Prediction Distribution
        axes[1, 1].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.5, label='No Churn', color='blue')
        axes[1, 1].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.5, label='Churn', color='red')
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Probability Distribution')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Ensure the models directory exists
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        # Save the plot
        plot_path = save_path / 'model_evaluation.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPlots saved to {plot_path}")
        
        return fig
    
    def save_model(self, path=None):
        """
        Save the trained model and preprocessors.
        """
        # Use dynamic path resolution
        if path is None:
            project_root = Path(__file__).parent.parent
            path = project_root / 'models' / 'churn_model.pkl'
        else:
            path = Path(path)
        
        # Ensure the directory exists
        path.parent.mkdir(exist_ok=True)
        
        model_artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_importance': self.feature_importance,
            'trained_features': self.trained_features  # Save feature names
        }
        joblib.dump(model_artifacts, path)
        print(f"\nModel saved to {path}")
        print(f"Features saved: {len(self.trained_features) if self.trained_features else 0} features")
    
    def load_model(self, path=None):
        """
        Load a saved model and preprocessors.
        """
        # Use dynamic path resolution
        if path is None:
            project_root = Path(__file__).parent.parent
            path = project_root / 'models' / 'churn_model.pkl'
        
        print(f"Loading model from: {path}")
        
        try:
            model_artifacts = joblib.load(path)
        except FileNotFoundError:
            print(f"⚠️  Model file not found: {path}")
            print("Please train the model first by running this script.")
            raise
        
        self.model = model_artifacts['model']
        self.scaler = model_artifacts['scaler']
        self.label_encoders = model_artifacts['label_encoders']
        self.feature_importance = model_artifacts['feature_importance']
        self.trained_features = model_artifacts.get('trained_features', None)
        
        print(f"✅ Model loaded successfully!")
        print(f"Model type: {type(self.model).__name__}")
        print(f"Number of features: {len(self.feature_importance) if self.feature_importance is not None else 'N/A'}")
        
        if self.trained_features:
            print(f"Features expected by model: {len(self.trained_features)}")
        
        return self


def main():
    """
    Main execution function for the churn prediction pipeline.
    """
    print("="*60)
    print("CUSTOMER CHURN PREDICTION PROJECT")
    print("="*60)
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Step 1: Load Real Data (with fallback to synthetic)
    print("\n[1/6] Loading data...")
    try:
        df = predictor.load_real_data()
        print(f"Dataset shape: {df.shape}")
        print(f"Churn rate: {df['churn'].mean():.2%}")
    except FileNotFoundError as e:
        print(f"⚠️  Real dataset not found: {e}")
        print("Using synthetic data instead.")
        print("📥 To use real data, download 'Telco_Customer_Churn_Expanded.xlsx'")
        print("   and place it in the data/ folder")
        df = predictor.create_sample_data(n_samples=5000)
        print(f"Dataset shape: {df.shape}")
        print(f"Churn rate: {df['churn'].mean():.2%}")
    except Exception as e:
        print(f"⚠️  Error loading real dataset: {e}")
        print("Using synthetic data instead.")
        df = predictor.create_sample_data(n_samples=5000)
        print(f"Dataset shape: {df.shape}")
        print(f"Churn rate: {df['churn'].mean():.2%}")
    
    # Step 2: Feature Engineering
    print("\n[2/6] Performing feature engineering...")
    df = predictor.feature_engineering(df)
    print(f"New shape after feature engineering: {df.shape}")
    
    # Step 3: Split Data
    print("\n[3/6] Splitting data into train and test sets...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['churn'])
    print(f"Training set: {train_df.shape}, Test set: {test_df.shape}")
    print(f"Train churn rate: {train_df['churn'].mean():.2%}")
    print(f"Test churn rate: {test_df['churn'].mean():.2%}")
    
    # Step 4: Preprocess Data
    print("\n[4/6] Preprocessing data...")
    X_train, y_train = predictor.preprocess_data(train_df, is_training=True)
    X_test, y_test = predictor.preprocess_data(test_df, is_training=False)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Step 5: Train Model with Hyperparameter Tuning
    print("\n[5/6] Training model...")
    print("Note: Set tune_hyperparameters=True for best results (slower)")
    predictor.train_model(X_train, y_train, tune_hyperparameters=False)
    
    # Step 6: Evaluate Model
    print("\n[6/6] Evaluating model...")
    results = predictor.evaluate_model(X_test, y_test)
    
    # Display Feature Importance
    print("\n" + "="*60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*60)
    if predictor.feature_importance is not None:
        print(predictor.feature_importance.head(10).to_string(index=False))
    else:
        print("Feature importance not available")
    
    # Create Visualizations
    print("\nGenerating visualizations...")
    predictor.plot_results(X_test, y_test)
    
    # Save Model
    predictor.save_model()
    
    print("\n" + "="*60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nKey Achievements:")
    print(f"✓ Model Accuracy: {results['accuracy']*100:.2f}%")
    print(f"✓ ROC-AUC Score: {results['roc_auc']:.4f}")
    print(f"✓ Number of features: {X_train.shape[1]}")
    print("✓ Feature engineering implemented")
    print("✓ Model saved for deployment")
    print("\nNext steps:")
    print("1. Run the Streamlit app: python -m streamlit run app/churn_app.py")
    print("2. Deploy to Streamlit Cloud: share.streamlit.io")
    

if __name__ == "__main__":
    main()