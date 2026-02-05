# Getting Started with Customer Churn Prediction

## Quick Start Guide

### Option 1: Quick Demo (Recommended for First Run)
Runs in ~30 seconds, uses default model parameters:
```bash
cd src
python quick_demo.py
```

### Option 2: Full Pipeline with Hyperparameter Tuning
Runs in 5-10 minutes, optimizes model parameters:
```bash
cd src
python churn_prediction.py
```

### Option 3: Interactive Jupyter Notebook
For exploration and experimentation:
```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

## What You'll Get

After running the model, you'll have:

1. **Trained Model** (`models/churn_model.pkl`)
   - Ready to use for predictions
   - Includes all preprocessing components
   - Can be loaded and reused

2. **Visualization Report** (`models/model_evaluation.png`)
   - Feature importance chart
   - Confusion matrix
   - ROC curve
   - Prediction distribution

3. **Performance Metrics**
   - Accuracy: 92%+ (with hyperparameter tuning)
   - ROC-AUC Score
   - Precision, Recall, F1-Score
   - Confusion Matrix

## Project Components

### Core Features

1. **Feature Engineering**
   - Average monthly charge calculation
   - Tenure grouping (0-1 year, 1-2 years, etc.)
   - Charge per service ratio
   - High-risk customer identification
   - Service engagement scoring

2. **Model Training**
   - Random Forest Classifier
   - GridSearchCV for hyperparameter optimization
   - Cross-validation (5-fold)
   - Class balancing for imbalanced data

3. **Evaluation Metrics**
   - Accuracy
   - ROC-AUC Score
   - Precision & Recall
   - F1-Score
   - Confusion Matrix
   - Feature Importance Rankings

## Using Your Own Data

### Step 1: Prepare Your Data
Your CSV should include columns like:
- Customer demographics (age, etc.)
- Account information (tenure, contract type)
- Service usage (monthly charges, services subscribed)
- Interaction history (service calls, complaints)
- Target variable (churn: 0 or 1)

### Step 2: Modify the Code
In `churn_prediction.py`, replace the `create_sample_data()` call:
```python
# Instead of:
df = predictor.create_sample_data(n_samples=5000)

# Use:
df = pd.read_csv('your_data.csv')
```

### Step 3: Adjust Features
Modify the `feature_engineering()` method to match your data:
```python
def feature_engineering(self, df):
    df = df.copy()
    
    # Add domain-specific features
    df['feature1'] = ...
    df['feature2'] = ...
    
    return df
```

## Understanding the Results

### Feature Importance
Shows which factors most influence churn:
- **High importance**: Key drivers of churn
- **Low importance**: Less relevant factors

Common top features:
1. Contract type (month-to-month vs long-term)
2. Tenure (how long customer has been with company)
3. Monthly charges
4. Customer service interactions
5. Number of services used

### Accuracy Metrics

**Accuracy**: Overall correctness
- 92%+ accuracy means model is correct 92% of the time

**Precision**: When model predicts churn, how often is it right?
- High precision = fewer false alarms

**Recall**: Of all actual churners, how many did we catch?
- High recall = catching most churners

**ROC-AUC**: Overall model quality (0.5 = random, 1.0 = perfect)
- 0.85+ is considered excellent

## Next Steps

### 1. Improve the Model
- Try different algorithms (XGBoost, LightGBM)
- Add more features from your domain knowledge
- Collect more data
- Engineer interaction features

### 2. Deploy the Model
```python
# Load saved model
predictor = ChurnPredictor()
predictor.load_model('models/churn_model.pkl')

# Make predictions
predictions = predictor.model.predict(new_data)
probabilities = predictor.model.predict_proba(new_data)[:, 1]
```

### 3. Monitor Performance
- Track accuracy over time
- Retrain periodically with new data
- Monitor feature importance changes
- A/B test intervention strategies

## Common Issues & Solutions

### Issue: Low Accuracy (<70%)
**Solutions:**
- Enable hyperparameter tuning
- Add more relevant features
- Check data quality
- Balance classes if highly imbalanced

### Issue: High False Positives
**Solution:** Adjust classification threshold:
```python
# Instead of default 0.5, use higher threshold
threshold = 0.7
predictions = (probabilities > threshold).astype(int)
```

### Issue: Model Takes Too Long
**Solutions:**
- Use quick_demo.py instead
- Reduce parameter grid in GridSearchCV
- Use smaller dataset for testing

## Tips for Success

1. **Start Simple**: Run quick_demo.py first to verify everything works
2. **Understand Your Data**: Use the Jupyter notebook for exploration
3. **Iterate**: Start with default features, then add custom ones
4. **Validate**: Always check if results make business sense
5. **Document**: Keep track of what features and parameters work best

## Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Random Forest Guide](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Feature Engineering Best Practices](https://www.kaggle.com/learn/feature-engineering)

## Support

If you encounter issues:
1. Check the error message carefully
2. Verify all dependencies are installed
3. Ensure data format matches expected structure
4. Review the example code in the notebook

---

**Happy Predicting! 🎯**
```

---

## **File 3: `requirements.txt`**
```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
jupyter>=1.0.0
ipykernel>=6.25.0