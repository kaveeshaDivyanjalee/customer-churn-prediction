"""
Quick Demo: Customer Churn Prediction
A faster version without hyperparameter tuning for demonstration purposes.
"""

import sys
sys.path.append('.')
from churn_prediction import ChurnPredictor
from sklearn.model_selection import train_test_split

def quick_demo():
    """
    Run a quick demonstration of the churn prediction model.
    This version skips hyperparameter tuning for faster execution.
    """
    print("="*60)
    print("CUSTOMER CHURN PREDICTION - QUICK DEMO")
    print("="*60)
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Create data
    print("\n[1/5] Loading data...")
    df = predictor.create_sample_data(n_samples=3000)
    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate: {df['churn'].mean():.2%}")
    
    # Feature engineering
    print("\n[2/5] Performing feature engineering...")
    df = predictor.feature_engineering(df)
    print(f"Features after engineering: {df.shape[1]}")
    
    # Split data
    print("\n[3/5] Splitting data...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['churn'])
    
    # Preprocess
    print("\n[4/5] Preprocessing data...")
    X_train, y_train = predictor.preprocess_data(train_df, is_training=True)
    X_test, y_test = predictor.preprocess_data(test_df, is_training=False)
    
    # Train model (WITHOUT hyperparameter tuning for speed)
    print("\n[5/5] Training model (using default parameters for speed)...")
    predictor.train_model(X_train, y_train, tune_hyperparameters=False)
    
    # Evaluate
    print("\nEvaluating model...")
    results = predictor.evaluate_model(X_test, y_test)
    
    # Show feature importance
    print("\n" + "="*60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*60)
    print(predictor.feature_importance.head(10).to_string(index=False))
    
    # Create visualizations
    print("\nGenerating visualizations...")
    predictor.plot_results(X_test, y_test)
    
    # Save model
    predictor.save_model()
    
    print("\n" + "="*60)
    print("QUICK DEMO COMPLETED!")
    print("="*60)
    print(f"\n✓ Model Accuracy: {results['accuracy']*100:.2f}%")
    print(f"✓ ROC-AUC Score: {results['roc_auc']:.4f}")
    print("\n💡 For full hyperparameter tuning, run: python churn_prediction.py")
    

if __name__ == "__main__":
    quick_demo()
