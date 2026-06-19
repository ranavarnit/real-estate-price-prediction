# real_estate_project1_starter.py
# Real Estate AI Project 1: House Price Prediction System
# Complete starter code with best practices

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# For deployment
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

class RealEstatePricePredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.best_model = None
        self.metrics = {}

    def load_data(self, filepath):
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format")
        print(f"Dataset loaded: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df

    def exploratory_analysis(self, df):
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        print(f"\nDataset Shape: {df.shape}")
        print(f"\nMissing Values:")
        print(df.isnull().sum().sort_values(ascending=False).head(10))
        print(f"\nNumeric Columns Summary:")
        print(df.describe())
        return df

    def feature_engineering(self, df):
        df = df.copy()

        # Handle total_sqft - convert range strings to average
        if 'total_sqft' in df.columns:
            def convert_sqft_to_num(x):
                if pd.isna(x):
                    return np.nan
                tokens = str(x).split('-')
                if len(tokens) == 2:
                    try:
                        return (float(tokens[0]) + float(tokens[1])) / 2
                    except:
                        return np.nan
                try:
                    return float(x)
                except:
                    return np.nan

            df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
            df['total_sqft'].fillna(df['total_sqft'].median(), inplace=True)

        # Fill missing values for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)

        # Fill missing values for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)

        # Create new features
        if 'total_sqft' in df.columns and 'bath' in df.columns:
            df['sqft_per_bath'] = df['total_sqft'] / (df['bath'] + 1)
        if 'total_sqft' in df.columns and 'bhk' in df.columns:
            df['sqft_per_bhk'] = df['total_sqft'] / df['bhk']
        if 'price' in df.columns and 'total_sqft' in df.columns:
            df['price_per_sqft'] = df['price'] / df['total_sqft']

        # Encode categorical variables
        for col in categorical_cols:
            if col != 'price':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # Log transform target
        if 'price' in df.columns:
            df['price'] = np.log1p(df['price'])

        return df

    def prepare_data(self, df, target_col='price'):
        X = df.drop([target_col], axis=1)
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

    def train_models(self, X_train, X_test, y_train, y_test, feature_names):
        print("\n" + "="*60)
        print("MODEL TRAINING & COMPARISON")
        print("="*60)
        models_config = {
            'LinearRegression': {'model': LinearRegression(), 'params': {}},
            'Ridge': {'model': Ridge(), 'params': {'alpha': [0.1, 1, 10, 100]}},
            'Lasso': {'model': Lasso(), 'params': {'alpha': [0.1, 1, 10, 100]}},
            'RandomForest': {'model': RandomForestRegressor(random_state=42), 'params': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}},
            'GradientBoosting': {'model': GradientBoostingRegressor(random_state=42), 'params': {'n_estimators': [100, 200], 'max_depth': [3, 5]}},
            'XGBoost': {'model': xgb.XGBRegressor(random_state=42), 'params': {'n_estimators': [100, 200], 'max_depth': [3, 6], 'learning_rate': [0.1, 0.01]}},
            'LightGBM': {'model': lgb.LGBMRegressor(random_state=42), 'params': {'n_estimators': [100, 200], 'max_depth': [3, 6], 'learning_rate': [0.1, 0.01]}}
        }
        best_r2 = -float('inf')
        for name, config in models_config.items():
            print(f"\nTraining {name}...")
            if config['params']:
                grid_search = GridSearchCV(config['model'], config['params'], cv=5, scoring='r2', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
            else:
                model = config['model']
                model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_test_orig = np.expm1(y_test)
            y_pred_orig = np.expm1(y_pred)
            r2 = r2_score(y_test_orig, y_pred_orig)
            rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
            mae = mean_absolute_error(y_test_orig, y_pred_orig)
            mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            self.models[name] = {
                'model': model, 'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape,
                'cv_r2_mean': cv_scores.mean(), 'cv_r2_std': cv_scores.std()
            }
            print(f"  R2: {r2:.4f}")
            print(f"  RMSE: {rmse:,.2f}")
            print(f"  MAE: {mae:,.2f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
            if r2 > best_r2:
                best_r2 = r2
                self.best_model = model
                self.best_model_name = name
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = dict(zip(feature_names, self.best_model.feature_importances_))
        print(f"\nBest Model: {self.best_model_name} (R2 = {best_r2:.4f})")
        return self.models

    def plot_model_comparison(self):
        results = pd.DataFrame({
            'Model': list(self.models.keys()),
            'R2': [m['r2'] for m in self.models.values()],
            'RMSE': [m['rmse'] for m in self.models.values()],
            'MAPE': [m['mape'] for m in self.models.values()]
        })
        fig = make_subplots(rows=1, cols=3, subplot_titles=('R2 Score', 'RMSE', 'MAPE (%)'))
        fig.add_trace(go.Bar(x=results['Model'], y=results['R2'], name='R2'), row=1, col=1)
        fig.add_trace(go.Bar(x=results['Model'], y=results['RMSE'], name='RMSE'), row=1, col=2)
        fig.add_trace(go.Bar(x=results['Model'], y=results['MAPE'], name='MAPE'), row=1, col=3)
        fig.update_layout(height=400, showlegend=False, title_text="Model Performance Comparison")
        fig.write_html("model_comparison.html")
        fig.show()
        return results

    def plot_feature_importance(self):
        if self.feature_importance:
            importance_df = pd.DataFrame({
                'Feature': list(self.feature_importance.keys()),
                'Importance': list(self.feature_importance.values())
            }).sort_values('Importance', ascending=True)
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['Feature'], importance_df['Importance'])
            plt.title(f'Feature Importance - {self.best_model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.show()

    def save_model(self, filepath='house_price_model.pkl'):
        joblib.dump({'model': self.best_model, 'scaler': self.scaler, 'label_encoders': self.label_encoders, 'feature_importance': self.feature_importance}, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='house_price_model.pkl'):
        data = joblib.load(filepath)
        self.best_model = data['model']
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_importance = data['feature_importance']
        print("Model loaded successfully")

if __name__ == "__main__":
    print("=" * 60)
    print("REAL ESTATE PRICE PREDICTION SYSTEM")
    print("=" * 60)
    predictor = RealEstatePricePredictor()

    print("\n[1] Loading dataset...")
    df = predictor.load_data("Bengaluru_House_Data.csv")

    print("\n[2] Exploratory Data Analysis...")
    df = predictor.exploratory_analysis(df)

    print("\n[3] Feature engineering...")
    df = predictor.feature_engineering(df)

    print("\n[4] Preparing data...")
    X_train, X_test, y_train, y_test, features = predictor.prepare_data(df)

    print("\n[5] Training models...")
    models = predictor.train_models(X_train, X_test, y_train, y_test, features)

    print("\n[6] Plotting model comparison...")
    predictor.plot_model_comparison()

    print("\n[7] Plotting feature importance...")
    predictor.plot_feature_importance()

    print("\n[8] Saving model...")
    predictor.save_model()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nTo run Streamlit app:")
    print("streamlit run house_price_prediction.py")
