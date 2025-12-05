# Loan Default Prediction – Credit Risk Modeling (XGBoost)

## 1. Project Overview

This project predicts the probability of loan default using real-world credit data (233k+ rows, 41 features).  
It includes:

- Full preprocessing pipeline (imputation, encoding, scaling)
- XGBoost and LightGBM models
- Threshold tuning based on business cost
- Cost matrix optimization to reduce financial losses
- SHAP explainability
- Streamlit application for deployment

---

## 2. Project Structure

```
Project2_Loan_Default/
│
├── app/
│   └── app.py
│
├── models/
│   ├── loan_default_xgb_pipeline.pkl
│   ├── best_threshold.txt
│   └── feature_names.json
│
├── notebooks/
│   └── 01_eda_baseline.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── assets/
│   ├── threshold_curve.png
│   ├── business_cost_curve.png
│   └── shap_summary.png
│
└── README.md
```

---

## 3. Visual Insights

### Threshold Tuning Curve
![Threshold Curve](assets/threshold_curve.png)

### Business Cost Matrix Curve
![Business Cost Curve](assets/business_cost_curve.png)

### SHAP Summary Plot
![SHAP Summary](assets/shap_summary.png)

### Streamlit App UI
![Streamlit UI](assets/streamlit_app.png)

---

## 4. Key Features

- End-to-end ML Pipeline using ColumnTransformer and Pipeline
- Handling missing values and categorical encoding
- XGBoost as primary model
- Business-driven threshold tuning instead of default 0.5
- Cost-sensitive decision optimization
- SHAP explainability for model transparency
- Streamlit UI for easy deployment

---

## 5. Model Performance

| Model | AUC |
|-------|-----|
| Logistic Regression | ~0.63 |
| LightGBM | ~0.66 |
| XGBoost | ~0.663 |

### Business Threshold Optimization
- Optimal threshold: **0.08**
- Minimum financial loss: **₹3.57 Crores**

---

## 6. Run Streamlit App

```
cd app
streamlit run app.py
```

---

## 7. Making Predictions (Example)

```
prob = model.predict_proba(input_df)[:, 1][0]
final_prediction = int(prob >= BEST_THRESHOLD)
```

---

## 8. Business Insights

- High LTV increases default probability  
- Low credit score strongly indicates risk  
- High number of recent enquiries is a key defaulter signal  
- Short credit history increases risk  
- Overdue and delinquent accounts are highly predictive  
- Salaried customers generally show lower risk than self-employed  

---

## 9. Future Improvements

- Hyperparameter tuning with Optuna
- Ensemble of XGBoost + LightGBM
- Deploy Streamlit app on Streamlit Cloud
- Add automated retraining pipeline
- Add monitoring dashboard

---

## 10. Author

**Veera Bokkisam**  
Machine Learning Engineer – Finance Domain

---
## ⭐ If you found this project useful, please ⭐ star the repo! 



---
