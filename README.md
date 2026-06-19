# Bengaluru House Price Prediction

## Overview
ML-powered real estate price prediction system achieving **98.7% R²** using Gradient Boosting. Built with scikit-learn, XGBoost, LightGBM, and Streamlit.

## Live Demo
- **Dashboard:** https://ranavarnit-real-estate-price-prediction.streamlit.app

## Features
- **7 ML Models** compared: Linear Regression, Ridge, Lasso, Random Forest, XGBoost, LightGBM, Gradient Boosting
- **Feature Engineering:** price_per_sqft, sqft_per_bath, location encoding
- **Interactive Dashboard:** Enter property details and get instant price estimate
- **Best Model:** Gradient Boosting (R² = 0.9870, MAPE = 1.94%)

## Tech Stack
- Python, scikit-learn, XGBoost, LightGBM
- Streamlit, Plotly
- Pandas, NumPy

## Quick Start
```bash
pip install -r requirements.txt
python real_estate_project1_starter.py  # Train model
streamlit run app.py  # Launch dashboard
```
## Results

### Model Performance Comparison

| Model | R² Score | RMSE | MAPE |
|-------|----------|------|------|
| **Gradient Boosting** | **0.9870** | **27.05** | **1.94%** |
| LightGBM | 0.9870 | 27.05 | 1.94% |
| XGBoost | 0.9870 | 27.05 | 1.94% |
| Random Forest | 0.9854 | 29.29 | 2.11% |
| Ridge | 0.9853 | 29.37 | 2.12% |
| Lasso | 0.9853 | 29.37 | 2.12% |
| Linear Regression | 0.9853 | 29.37 | 2.12% |

**Best Model:** Gradient Boosting (R² = 0.9870, MAPE = 1.94%)

## Author: Varnit Rana | varnit10@gmail.com
