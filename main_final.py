import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Fuel Forecast Analysis",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Original Theme */
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #262730;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
        background: linear-gradient(135deg, #f8f9fa, #f0f2f6);
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        letter-spacing: 1.2px;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, rgba(255, 75, 75, 0.1) 0%, rgba(255, 75, 75, 0) 70%);
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 150px;
        height: 4px;
        background: linear-gradient(90deg, #ff4b4b, #ff7c43);
        border-radius: 4px;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #4e5663;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 400;
        letter-spacing: 0.5px;
        position: relative;
    }
    
    .metric-card {
        background-color: #ffffff;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #ff4b4b;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .section-header {
        font-size: 2rem;
        font-weight: 800;
        color: #262730;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        text-align: center;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 4px;
        background: linear-gradient(90deg, #ff4b4b, #ff7c43);
        border-radius: 4px;
    }
    
    .info-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    .model-section {
        background: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        border: 2px solid #f0f2f6;
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }
    
    .button-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .custom-button {
        background: #ff4b4b;
        color: white;
        padding: 1rem 2rem;
        border: none;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        min-width: 200px;
    }
    
    .custom-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    .secondary-button {
        background: #09ab3b;
        color: white;
        padding: 1rem 2rem;
        border: none;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        min-width: 200px;
    }
    
    .secondary-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        padding: 1rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: none;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 65px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 12px;
        gap: 1rem;
        padding: 15px 25px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        color: #4e5663;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        position: relative;
        overflow: hidden;
    }
    
    .stTabs [data-baseweb="tab"]::before {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 0;
        height: 3px;
        background: linear-gradient(90deg, #ff4b4b, #ff7c43);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff4b4b, #ff7c43);
        color: white;
        box-shadow: 0 8px 20px rgba(255, 75, 75, 0.2);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"]::before {
        width: 100%;
    }
    
    .stTabs [aria-selected="false"]:hover {
        background: #f8f9fa;
        color: #262730;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    
    .stTabs [aria-selected="false"]:hover::before {
        width: 100%;
    }
    
    .welcome-text {
        text-align: center;
        color: #4e5663;
        font-size: 1.1rem;
        margin: 0.5rem 0 1.5rem 0;
        line-height: 1.6;
    }
    
    .stats-container {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #f0f2f6;
    }
    
    /* Data Source Section Styling */
    .data-source-container {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.08);
        text-align: center;
        margin: 2rem 0 3rem 0;
        position: relative;
        overflow: hidden;
        border: none;
    }
    
    .data-source-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #ff4b4b, #ff7c43);
    }
    
    .data-source-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #262730;
        margin-bottom: 1.2rem;
        text-align: center;
        position: relative;
        display: inline-block;
    }
    
    .data-source-title::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, #ff4b4b, #ff7c43);
        border-radius: 3px;
    }
    
    .data-source-icon {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1.2rem;
        color: #ff4b4b;
        text-shadow: 0 2px 10px rgba(255, 75, 75, 0.2);
    }
    
    /* Progress Indicator */
    .progress-container {
        background: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #f0f2f6;
    }
    
    .progress-label {
        display: inline-block;
        color: #ff4b4b;
        margin-top: 8px;
        font-size: 0.9rem;
        width: 100%;
        text-align: center;
    }
    
    /* Progress styling simplified */
    
    /* Card Styling */
    .dashboard-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: none;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .dashboard-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #ff4b4b, #ff7c43);
    }
    
    .dashboard-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .card-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #262730;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        position: relative;
        text-align: center;
    }
    
    .card-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, #ff4b4b, #ff7c43);
        border-radius: 3px;
    }
    
    /* Data Overview Cards */
    .data-card {
        text-align: center;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        z-index: 1;
    }
    
    .data-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: rgba(255, 255, 255, 0.1);
        transform: rotate(30deg);
        z-index: -1;
        transition: all 0.5s ease;
    }
    
    .data-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.2);
    }
    
    .data-card:hover::before {
        transform: rotate(30deg) translateY(-10%);
    }
    
    .data-card-1 {
        background: linear-gradient(135deg, #ff4b4b, #ff7c43);
        border-bottom: 4px solid #e63e3e;
    }
    
    .data-card-2 {
        background: linear-gradient(135deg, #09ab3b, #4cc764);
        border-bottom: 4px solid #078f31;
    }
    
    .data-card-3 {
        background: linear-gradient(135deg, #1f77b4, #5aa7d4);
        border-bottom: 4px solid #1a6698;
    }
    
    .data-card-4 {
        background: linear-gradient(135deg, #ff7c43, #ffaa71);
        border-bottom: 4px solid #e66a33;
    }
    
    .data-value {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.3rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .data-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 500;
    }
    
    .data-trend {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.3rem;
        font-weight: 500;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #4e5663;
        font-size: 0.9rem;
        border-top: 1px solid #f0f2f6;
        margin-top: 3rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease-out forwards;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f0f2f6;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #ff4b4b;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #e63e3e;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Utility functions (copied from main.py) ----------
@st.cache_data
def smape(a, f):
    denom = (np.abs(a) + np.abs(f))
    mask = denom != 0
    return 100 * np.mean(2 * np.abs(a[mask] - f[mask]) / denom[mask])

@st.cache_data
def enhanced_feature_engineering(df, forecasting_mode=False):
    df = df.copy()
    # Price elasticity and volatility
    df['price_elasticity'] = np.where(df['avg_price'] != 0,
                                      df['sales_volume'] / df['avg_price'],
                                      np.nan)
    df['price_volatility'] = df.groupby(['Region', 'Product'])['avg_price'].transform(
        lambda x: x.rolling(4, min_periods=1).std()
    )
    df['price_change'] = df['avg_price'].pct_change()

    # Volume trends and volatility
    df['volume_trend'] = df.groupby(['Region', 'Product'])['sales_volume'].transform(
        lambda x: x.rolling(8, min_periods=1).mean()
    )
    df['volume_change'] = df['sales_volume'].pct_change()
    df['volume_volatility'] = df.groupby(['Region', 'Product'])['sales_volume'].transform(
        lambda x: x.rolling(4, min_periods=1).std()
    )

    # Enhanced seasonal / trend features
    df['seasonal_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['seasonal_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    df['quarterly_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 13)
    df['quarterly_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 13)  # Add quarterly cosine
    df['monthly_sin'] = np.sin(2 * np.pi * df['month'] / 12)  # Add monthly sine
    df['monthly_cos'] = np.cos(2 * np.pi * df['month'] / 12)  # Add monthly cosine
    df['trend_factor'] = df['week_of_year'] / 52
    df['trend_squared'] = (df['week_of_year'] / 52) ** 2

    # Regional and product rolling trends
    df['regional_trend'] = df.groupby('Region')['sales_volume'].transform(
        lambda x: x.rolling(12, min_periods=1).mean()
    )
    df['product_trend'] = df.groupby('Product')['sales_volume'].transform(
        lambda x: x.rolling(12, min_periods=1).mean()
    )

    # Interaction features
    df['price_volume_ratio'] = df['avg_price'] * df['sales_volume']
    df['price_over_volume'] = np.where(df['sales_volume'] != 0,
                                       df['avg_price'] / df['sales_volume'],
                                       np.nan)

    # External factors
    # Remove fixed seed to allow for true randomness in forecasting
    # Only use fixed seed during initial feature engineering, not during forecasting
    if not forecasting_mode:
        np.random.seed(42)  # Keep deterministic behavior for initial data processing
    df['weather_factor'] = np.random.normal(1, 0.1, len(df))
    df['economic_factor'] = np.random.normal(1, 0.05, len(df))
    df['event_factor'] = np.random.normal(1, 0.15, len(df))

    # Additional rolling stats on sales_volume
    for window in [4, 8, 12]:
        df[f'roll{window}_mean'] = df.groupby(['Region', 'Product'])['sales_volume'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'roll{window}_std'] = df.groupby(['Region', 'Product'])['sales_volume'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

    return df

@st.cache_data
def evaluate_preds(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = r2_score(y_true, y_pred)
    smape_val = smape(y_true.values, y_pred)
    return {
        'MAE': mae,
        'MAPE': mape,
        'RMSE': rmse,
        'R2': r2,
        'SMAPE': smape_val
    }

def generate_forecast(res, group_eng, product, forecast_weeks, forecast_method,
                      include_confidence, confidence_level):
    """Replicate the original detailed forecasting logic.

    This function mirrors the extensive forecasting routine previously embedded
    in the training loop. It regenerates time and seasonal features, applies
    product-specific randomisation and handles both direct and recursive
    strategies while optionally producing confidence intervals.
    """

    models = res.get('models', {})
    lgbm = models.get('lgbm')
    rf = models.get('rf')
    metrics = res.get('metrics', {})
    y_test = res.get('y_test', pd.Series(dtype=float))
    imputer = res.get('imputer')
    selector = res.get('selector')
    feature_cols = res.get('feature_cols', [])
    selected_features = res.get('selected_features', [])
    feature_selection_enabled = res.get('feature_selection_enabled', False)
    ensemble_method = res.get('ensemble_method', 'Average')
    lgbm_weight = res.get('lgbm_weight', 0.5)

    if not feature_cols or lgbm is None or rf is None or imputer is None:
        return {
            'dates': [],
            'dates_str': [],
            'values': [],
            'lower_bounds': [],
            'upper_bounds': [],
            'method': forecast_method,
            'horizon': forecast_weeks,
            'confidence_level': confidence_level if include_confidence else None,
        }

    group_eng = group_eng.copy()
    if not pd.api.types.is_datetime64_any_dtype(group_eng['week_start']):
        group_eng['week_start'] = pd.to_datetime(group_eng['week_start'])

    last_data = group_eng.iloc[-1:].copy()
    last_date = pd.to_datetime(last_data['week_start'].iloc[0])

    forecast_dates = [last_date + timedelta(days=7 * i) for i in range(1, forecast_weeks + 1)]
    forecast_values = []
    lower_bounds = []
    upper_bounds = []

    if forecast_method == 'Direct':
        for i in range(forecast_weeks):
            future_data = last_data.copy()
            future_data['week_start'] = forecast_dates[i]
            future_data['week_start'] = pd.to_datetime(future_data['week_start'])
            future_data['month'] = future_data['week_start'].dt.month
            future_data['year'] = future_data['week_start'].dt.year
            future_data['week_of_year'] = future_data['week_start'].dt.isocalendar().week
            future_data['seasonal_sin'] = np.sin(2 * np.pi * future_data['week_of_year'] / 52)
            future_data['seasonal_cos'] = np.cos(2 * np.pi * future_data['week_of_year'] / 52)
            future_data['quarterly_sin'] = np.sin(2 * np.pi * future_data['week_of_year'] / 13)
            future_data['quarterly_cos'] = np.cos(2 * np.pi * future_data['week_of_year'] / 13)
            future_data['monthly_sin'] = np.sin(2 * np.pi * future_data['month'] / 12)
            future_data['monthly_cos'] = np.cos(2 * np.pi * future_data['month'] / 12)
            future_data['trend_factor'] = future_data['week_of_year'] / 52
            future_data['trend_squared'] = (future_data['week_of_year'] / 52) ** 2
            future_data['price_change'] = 0

            is_hobc = product == 'HOBC'
            historical_volatility = metrics.get('RMSE', 0) / y_test.mean() if len(y_test) > 0 and y_test.mean() > 0 else 0.05
            base_variation_factor = max(0.02, historical_volatility * 0.3) * (1 + i * 0.15)
            if is_hobc:
                variation_factor = base_variation_factor * 3.0
                min_absolute_variation = 1000
            else:
                variation_factor = base_variation_factor * 1.5
                min_absolute_variation = 250

            relative_change = np.random.normal(0, variation_factor)
            future_data['volume_change'] = relative_change

            if len(y_test) > 1:
                historical_trend = 1 if y_test.iloc[-1] > y_test.iloc[0] else -1
                trend_strength = min(1.0, abs(y_test.iloc[-1] - y_test.iloc[0]) / (y_test.mean() + 1e-10))
            else:
                historical_trend = 1
                trend_strength = 0.5

            volatility_factor = 0.02 * (i + 1) * (3.0 if is_hobc else 1.0)
            trend_factor = 0.03 * (i + 1) * (3.0 if is_hobc else 1.0)
            regional_product_factor = 0.01 * (i + 1) * (3.0 if is_hobc else 1.0)
            trend_bias = historical_trend * trend_strength * 0.01 * (i + 1)

            future_data['price_volatility'] *= (1 + np.random.normal(trend_bias, volatility_factor))
            future_data['volume_trend'] *= (1 + np.random.normal(trend_bias * 1.5, trend_factor))
            future_data['volume_volatility'] *= (1 + np.random.normal(trend_bias, volatility_factor))
            future_data['regional_trend'] *= (1 + np.random.normal(trend_bias, regional_product_factor))
            future_data['product_trend'] *= (1 + np.random.normal(trend_bias, regional_product_factor))

            rolling_factor = 0.02 * (i + 1) * (3.0 if is_hobc else 1.0)
            for window in [4, 8, 12]:
                if f'roll{window}_mean' in future_data.columns:
                    window_adjusted_factor = rolling_factor * (1.0 + (1.0 / window))
                    window_trend_bias = trend_bias * (1.0 + (1.0 / window))
                    mean_change = np.random.normal(window_trend_bias, window_adjusted_factor)
                    future_data[f'roll{window}_mean'] *= (1 + mean_change)
                    if is_hobc:
                        abs_variation = min_absolute_variation * (1.0 + (0.5 / window))
                        direction_prob = 0.5 + (0.3 * historical_trend)
                        direction = np.random.choice([-1, 1], p=[1-direction_prob, direction_prob]) if historical_trend > 0 else np.random.choice([-1, 1], p=[direction_prob, 1-direction_prob])
                        future_data[f'roll{window}_mean'] += direction * abs_variation
                    elif min_absolute_variation > 0:
                        direction_prob = 0.5 + (0.2 * historical_trend)
                        direction = np.random.choice([-1, 1], p=[1-direction_prob, direction_prob]) if historical_trend > 0 else np.random.choice([-1, 1], p=[direction_prob, 1-direction_prob])
                        future_data[f'roll{window}_mean'] += direction * min_absolute_variation * (0.5 + (0.2 / window))

                if f'roll{window}_std' in future_data.columns:
                    std_factor = rolling_factor * (1.0 + historical_volatility)
                    future_data[f'roll{window}_std'] *= (1 + np.random.normal(0, std_factor))
                    min_std = min_absolute_variation * 0.2 * historical_volatility if is_hobc else min_absolute_variation * 0.1 * historical_volatility
                    if min_std > 0:
                        future_data[f'roll{window}_std'] = future_data[f'roll{window}_std'].clip(lower=min_std)

            X_future = future_data[feature_cols]
            X_future_clean = pd.DataFrame(imputer.transform(X_future), columns=feature_cols, index=X_future.index)
            if feature_selection_enabled and selector is not None:
                X_future_sel = selector.transform(X_future_clean)
            else:
                X_future_sel = X_future_clean.values
            X_future_sel = np.nan_to_num(X_future_sel, nan=0.0, posinf=0.0, neginf=0.0)

            pred_lgbm_future = lgbm.predict(X_future_sel)
            pred_rf_future = rf.predict(X_future_sel)
            if ensemble_method == "Weighted Average":
                ensemble_pred_future = (lgbm_weight * pred_lgbm_future) + ((1 - lgbm_weight) * pred_rf_future)
            else:
                ensemble_pred_future = (pred_lgbm_future + pred_rf_future) / 2

            forecast_values.append(ensemble_pred_future[0])

            if include_confidence:
                mae = metrics.get('MAE', 0)
                rmse = metrics.get('RMSE', 0)
                error_estimate = (0.7 * mae) + (0.3 * rmse)
                z_score = 1.96
                if confidence_level == 0.99:
                    z_score = 2.58
                elif confidence_level == 0.9:
                    z_score = 1.645
                elif confidence_level == 0.85:
                    z_score = 1.44
                elif confidence_level == 0.8:
                    z_score = 1.28
                if is_hobc:
                    step_factor = 1 + (i * 0.2)
                    min_margin = 2000 * (i + 1)
                else:
                    step_factor = 1 + (i * 0.1)
                    min_margin = 0
                margin = max(z_score * error_estimate * step_factor, min_margin)
                forecast_value = max(ensemble_pred_future[0], margin/2)
                forecast_values[-1] = forecast_value
                lower_bounds.append(max(0, forecast_value - margin))
                upper_bounds.append(forecast_value + margin)
            else:
                lower_bounds.append(None)
                upper_bounds.append(None)

    else:  # Recursive
        current_data = last_data.copy()
        prediction_history = []
        for i in range(forecast_weeks):
            current_data['week_start'] = forecast_dates[i]
            current_data['week_start'] = pd.to_datetime(current_data['week_start'])
            current_data['month'] = current_data['week_start'].dt.month
            current_data['year'] = current_data['week_start'].dt.year
            current_data['week_of_year'] = current_data['week_start'].dt.isocalendar().week

            for col in current_data.columns:
                if current_data[col].dtype.kind in 'fc':
                    current_data[col] = current_data[col].replace([np.inf, -np.inf], np.nan)
                    if current_data[col].isna().any():
                        if current_data[col].notna().any():
                            current_data[col] = current_data[col].fillna(current_data[col].mean())
                        else:
                            current_data[col] = current_data[col].fillna(0)

            X_future = current_data[feature_cols]
            X_future = X_future.replace([np.inf, -np.inf], np.nan)
            X_future = X_future.fillna(X_future.mean())
            X_future_clean = pd.DataFrame(imputer.transform(X_future), columns=feature_cols, index=X_future.index)
            X_future_clean = X_future_clean.replace([np.inf, -np.inf], np.nan)
            X_future_clean = X_future_clean.fillna(X_future_clean.mean())
            if feature_selection_enabled and selector is not None:
                X_future_sel = selector.transform(X_future_clean)
            else:
                X_future_sel = X_future_clean.values
            X_future_sel = np.nan_to_num(X_future_sel, nan=0.0, posinf=0.0, neginf=0.0)

            pred_lgbm_future = lgbm.predict(X_future_sel)
            pred_rf_future = rf.predict(X_future_sel)
            if ensemble_method == "Weighted Average":
                ensemble_pred_future = (lgbm_weight * pred_lgbm_future) + ((1 - lgbm_weight) * pred_rf_future)
            else:
                ensemble_pred_future = (pred_lgbm_future + pred_rf_future) / 2

            forecast_value = ensemble_pred_future[0]
            if i > 0 and len(prediction_history) > 0:
                trend_direction = 1 if prediction_history[-1] > prediction_history[0] else -1
                seasonal_factor = (current_data.get('seasonal_sin', 0) + current_data.get('monthly_sin', 0)) / 2
                if product == 'HOBC':
                    rand_factor = np.random.normal(trend_direction * 0.05 + seasonal_factor * 0.03, 0.08)
                    forecast_value = max(100, forecast_value * (1 + rand_factor))
                else:
                    rand_factor = np.random.normal(trend_direction * 0.03 + seasonal_factor * 0.02, 0.05)
                    forecast_value = max(100, forecast_value * (1 + rand_factor))

            forecast_values.append(forecast_value)
            prediction_history.append(forecast_value)
            current_data['sales_volume'] = forecast_value

            if include_confidence:
                mae = metrics.get('MAE', 0)
                rmse = metrics.get('RMSE', 0)
                error_estimate = (0.7 * mae) + (0.3 * rmse)
                z_score = 1.96
                if confidence_level == 0.99:
                    z_score = 2.58
                elif confidence_level == 0.9:
                    z_score = 1.645
                elif confidence_level == 0.85:
                    z_score = 1.44
                elif confidence_level == 0.8:
                    z_score = 1.28
                step_factor = 1 + 0.1 * (i + 1)
                margin = z_score * error_estimate * step_factor
                lower_bounds.append(max(0, forecast_value - margin))
                upper_bounds.append(forecast_value + margin)
            else:
                lower_bounds.append(None)
                upper_bounds.append(None)

    return {
        'dates': forecast_dates,
        'dates_str': [d.strftime('%Y-%m-%d') for d in forecast_dates],
        'values': forecast_values,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'method': forecast_method,
        'horizon': forecast_weeks,
        'confidence_level': confidence_level if include_confidence else None,
    }

# Helper function to convert datetime columns to strings for Streamlit display
def prepare_df_for_display(df):
    """Convert any datetime columns to strings to avoid Arrow conversion issues"""
    if df is None or df.empty:
        return df
    
    df_display = df.copy()
    for col in df_display.columns:
        # Check if column is datetime type
        if pd.api.types.is_datetime64_any_dtype(df_display[col]):
            df_display[col] = df_display[col].dt.strftime('%Y-%m-%d')
        # Check if column is object type and might contain Timestamp objects
        elif df_display[col].dtype == 'object' and len(df_display) > 0:
            # Try to convert the entire column to datetime if it contains any Timestamp objects
            try:
                # Check if any values in the column are Timestamp objects
                sample_values = df_display[col].dropna().head(10).tolist()
                has_timestamp = any(isinstance(x, (pd.Timestamp, datetime)) for x in sample_values)
                
                if has_timestamp:
                    # Convert all values in the column to strings
                    df_display[col] = df_display[col].apply(
                        lambda x: x.strftime('%Y-%m-%d') if isinstance(x, (pd.Timestamp, datetime)) 
                        else str(x) if x is not None else x
                    )
            except Exception:
                # If there's any error, try to handle individual Timestamp objects
                try:
                    df_display[col] = df_display[col].apply(
                        lambda x: x.strftime('%Y-%m-%d') if isinstance(x, (pd.Timestamp, datetime))
                        else x
                    )
                except Exception:
                    # If all else fails, convert the column to strings
                    df_display[col] = df_display[col].astype(str)
    
    return df_display

@st.cache_data
# Function implementation moved directly into the progress bar code in the overall model training section

@st.cache_data
# Function implementation moved directly into the progress bar code in the region-fuel model training section

@st.cache_data(ttl=3600)  # Cache for 1 hour to improve performance
def plot_actual_vs_predicted(test_data, y_test, y_pred, region, product):
    """Plot the Actual vs Predicted values for a specific region-product combination."""
    mask = (test_data['Region'] == region) & (test_data['Product'] == product)
    combo_test = test_data[mask]
    combo_y_test = y_test[mask]
    combo_y_pred = y_pred[mask]

    if len(combo_test) == 0:
        return None

    fig = go.Figure()

    # Ensure week_start is properly formatted for plotting
    x_values = combo_test['week_start']
    
    # Function to format values with appropriate unit labels
    def format_value_with_unit(value):
        """Format numeric values with unit suffixes.

        Converts array-like inputs to scalars so that upstream computations and
        Plotly rendering receive plain ``float`` values.
        """
        if isinstance(value, (np.ndarray, list, tuple, pd.Series)):
            arr = np.asarray(value)
            value = float(arr.ravel()[0]) if arr.size > 0 else 0.0
        else:
            value = float(value)
        if value >= 1_000_000_000:
            return f"{value/1_000_000_000:.2f}B"
        elif value >= 1_000_000:
            return f"{value/1_000_000:.2f}M"
        elif value >= 1_000:
            return f"{value/1_000:.2f}K"
        else:
            return f"{value:.2f}"
    
    # Pre-compute formatted values to avoid repeated calculations
    actual_formatted = [format_value_with_unit(val) for val in combo_y_test]
    pred_formatted = [format_value_with_unit(val) for val in combo_y_pred]
    
    # Actual values plot with formatted hover values
    fig.add_trace(go.Scatter(
        x=x_values,
        y=combo_y_test,
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue', width=2),
        opacity=0.9,
        hovertemplate="Date: %{x}<br>Actual: %{y:.2f} (%{customdata})<extra></extra>",
        customdata=actual_formatted
    ))

    # Predicted values plot with formatted hover values
    fig.add_trace(go.Scatter(
        x=x_values,
        y=combo_y_pred,
        mode='lines+markers',
        name='Predicted',
        line=dict(color='red', width=2, dash='dash'),
        opacity=0.9,
        hovertemplate="Date: %{x}<br>Predicted: %{y:.2f} (%{customdata})<extra></extra>",
        customdata=pred_formatted
    ))

    fig.update_layout(
        title=f"{region} - {product}: Actual vs Predicted Weekly Sales",
        xaxis_title="Date",
        yaxis_title="Weekly Sales Volume (Litres)",
        height=400,
        template="plotly_white",
        showlegend=True
    )

    return fig

def create_metrics_dashboard(metrics):
    """Create a metrics dashboard with cards."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("MAE", f"{metrics['MAE']:.2f}")
    with col2:
        st.metric("MAPE", f"{metrics['MAPE']*100:.2f}%")
    with col3:
        st.metric("RMSE", f"{metrics['RMSE']:.2f}")
    with col4:
        st.metric("R²", f"{metrics['R2']:.3f}")
    with col5:
        st.metric("SMAPE", f"{metrics['SMAPE']:.2f}%")

def create_comprehensive_data_overview(df):
    """Create comprehensive data overview with multiple visualizations."""
    
    # Pre-compute statistics to avoid recalculation
    stats = {
        "total_records": len(df),
        "unique_regions": df['Region'].nunique(),
        "unique_products": df['Product'].nunique(),
        "date_range": f"{df['week_start'].min().strftime('%Y-%m-%d') if not isinstance(df['week_start'].min(), str) else df['week_start'].min()} to {df['week_start'].max().strftime('%Y-%m-%d') if not isinstance(df['week_start'].max(), str) else df['week_start'].max()}"
    }

    # Basic statistics
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        st.metric("Total Records", stats["total_records"])
    with col2:
        st.metric("Regions", stats["unique_regions"])
    with col3:
        st.metric("Products", stats["unique_products"])
    with col4:
        st.metric("Date Range", stats["date_range"])
    
    # Price statistics by fuel type
    st.markdown("### 💰 Price Statistics by Fuel Type")
    price_stats = df.groupby('Product')['avg_price'].agg(['mean', 'min', 'max', 'std']).round(2)
    price_stats.columns = ['Average Price (PKR)', 'Min Price (PKR)', 'Max Price (PKR)', 'Std Dev (PKR)']
    st.dataframe(prepare_df_for_display(price_stats), use_container_width=True)

    # Function to format values with appropriate unit labels
    def format_value_with_unit(value):
        """Format numeric values with unit suffixes, handling array inputs."""
        if isinstance(value, (np.ndarray, list, tuple, pd.Series)):
            arr = np.asarray(value)
            value = float(arr.ravel()[0]) if arr.size > 0 else 0.0
        else:
            value = float(value)
        if value >= 1_000_000_000:
            return f"{value/1_000_000_000:.2f}B"
        elif value >= 1_000_000:
            return f"{value/1_000_000:.2f}M"
        elif value >= 1_000:
            return f"{value/1_000:.2f}K"
        else:
            return f"{value:.2f}"

    def format_tick(value: float) -> str:
        """Format tick labels without decimals for large numbers."""
        if value >= 1_000_000_000:
            return f"{int(value/1_000_000_000)}B"
        elif value >= 1_000_000:
            return f"{int(value/1_000_000)}M"
        elif value >= 1_000:
            return f"{int(value/1_000)}K"
        else:
            return str(int(value))

    # Visualizations
    st.markdown("### 📊 Sales Volume Analysis")

    # Create region and product charts with caching
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def create_region_volume_chart(df, log_x=False):
        # Sales volume by region
        region_volume = df.groupby('Region')['sales_volume'].sum().sort_values(ascending=False)

        # Create a DataFrame for px.bar
        region_df = pd.DataFrame({
            'Region': region_volume.index,
            'sales_volume': region_volume.values,
            'formatted_volume': [format_value_with_unit(val) for val in region_volume.values]
        })

        fig_region = px.bar(
            data_frame=region_df,
            y='Region',
            x='sales_volume',
            orientation='h',
            title='Total Weekly Sales Volume by Region',
            labels={'sales_volume': 'Weekly Sales Volume (Litres)'},
            color='sales_volume',
            color_continuous_scale='Blues',
            custom_data='formatted_volume',
            log_x=log_x
        )
        fig_region.update_traces(
            hovertemplate='Region: %{y}<br>Sales Volume: %{x:.2f} (%{customdata})<extra></extra>'
        )
        fig_region.update_layout(
            height=500,
            width=800,
            template='plotly_white',
            font=dict(size=14),
            title_font_size=18,
            xaxis_title='Weekly Sales Volume (Litres)',
            xaxis_title_font_size=14,
            yaxis_title=None,
            coloraxis_showscale=False
        )
        return fig_region
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def create_product_volume_chart(df):
        # Sales volume by product
        product_volume = df.groupby('Product')['sales_volume'].sum().sort_values(ascending=False)

        # Create a DataFrame for px.bar
        product_df = pd.DataFrame({
            'Product': product_volume.index,
            'sales_volume': product_volume.values,
            'formatted_volume': [format_value_with_unit(val) for val in product_volume.values]
        })
        return product_df

    @st.cache_data(ttl=3600)
    def create_product_chart(product_df, log_y=False):
        fig_product = px.bar(
            data_frame=product_df,
            x='Product',
            y='sales_volume',
            title='Total Sales Volume by Product',
            labels={'sales_volume': 'Sales Volume (Litres)'},
            color='sales_volume',
            color_continuous_scale='Greens',
            custom_data='formatted_volume',
            log_y=log_y
        )
        fig_product.update_traces(
            hovertemplate='Product: %{x}<br>Sales Volume: %{y:.2f} (%{customdata})<extra></extra>'
        )
        fig_product.update_layout(
            height=500,
            width=800,
            template='plotly_white',
            font=dict(size=14),
            title_font_size=18,
            xaxis_title_font_size=14,
            yaxis_title='Sales Volume (Litres)',
            yaxis_title_font_size=14,
            coloraxis_showscale=False
        )
        if log_y:
            max_val = product_df['sales_volume'].max()
            tick_vals = [9e7] + [1e8 * i for i in range(1, int(max_val / 1e8) + 2)]
            tick_text = [format_tick(v) for v in tick_vals]
            fig_product.update_yaxes(tickvals=tick_vals, ticktext=tick_text)
        fig_product.update_xaxes(tickangle=45)
        return fig_product

    @st.cache_data(ttl=3600)
    def create_region_product_chart(df, log_y=False):
        rp_data = df.groupby(['Region', 'Product'])['sales_volume'].sum().reset_index()
        rp_data['formatted_volume'] = rp_data['sales_volume'].apply(format_value_with_unit)
        fig_rp = px.bar(
            rp_data,
            x='Region',
            y='sales_volume',
            color='Product',
            color_discrete_sequence=['#1f77b4', '#2ca02c', '#7f7f7f'],
            barmode='group',
            title='Sales Volume by Region and Product',
            labels={'sales_volume': 'Sales Volume (Litres)'},
            custom_data='formatted_volume',
            log_y=log_y
        )
        fig_rp.update_traces(
            hovertemplate='Region: %{x}<br>Product: %{fullData.name}<br>Sales Volume: %{y:.2f} (%{customdata})<extra></extra>'
        )
        fig_rp.update_layout(
            height=600,
            width=800,
            template='plotly_white',
            yaxis_title='Sales Volume (Litres)'
        )
        if log_y:
            max_val = rp_data['sales_volume'].max()
            tick_vals = [9e7] + [1e8 * i for i in range(1, int(max_val / 1e8) + 2)]
            tick_text = [format_tick(v) for v in tick_vals]
            fig_rp.update_yaxes(tickvals=tick_vals, ticktext=tick_text)
        fig_rp.update_xaxes(tickangle=45)
        return fig_rp

    # Toggle for normal and lagged values
    view_mode = st.radio("View Mode", ["Normal", "Lagged"], horizontal=True)

    # Icon-based view selection
    if 'volume_view' not in st.session_state:
        st.session_state.volume_view = 'Region'

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🌍", help="By Region"):
            st.session_state.volume_view = 'Region'
    with col2:
        if st.button("🛢️", help="By Product"):
            st.session_state.volume_view = 'Product'
    with col3:
        if st.button("🌍🛢️", help="Region vs Product"):
            st.session_state.volume_view = 'Region vs Product'

    if st.session_state.volume_view == 'Region':
        fig_region = create_region_volume_chart(df, log_x=(view_mode == "Lagged"))
        st.plotly_chart(fig_region, use_container_width=True)
    elif st.session_state.volume_view == 'Product':
        product_df = create_product_volume_chart(df)
        fig_product = create_product_chart(product_df, log_y=(view_mode == "Lagged"))
        st.plotly_chart(fig_product, use_container_width=True)
    else:
        fig_rp = create_region_product_chart(df, log_y=(view_mode == "Lagged"))
        st.plotly_chart(fig_rp, use_container_width=True)

    # Time series analysis
    st.markdown("### 📈 Monthly Sales Volume Trend")
    
    @st.cache_data(ttl=3600)
    def create_monthly_sales_chart(df, log_y=False):
        if not pd.api.types.is_datetime64_any_dtype(df['week_start']):
            df = df.copy()
            df['week_start'] = pd.to_datetime(df['week_start'], errors='coerce')

        df_monthly = (
            df.groupby([df['week_start'].dt.to_period('M'), 'Product'])['sales_volume']
            .sum()
            .reset_index()
        )
        df_monthly['week_start'] = df_monthly['week_start'].astype(str)
        df_monthly['formatted_volume'] = df_monthly['sales_volume'].apply(format_value_with_unit)

        fig_monthly = px.line(
            df_monthly,
            x='week_start',
            y='sales_volume',
            color='Product',
            color_discrete_sequence=['#1f77b4', '#2ca02c', '#7f7f7f'],
            title='Monthly Sales Volume Trend by Fuel Type',
            labels={'week_start': 'Month', 'sales_volume': 'Monthly Sales Volume (Litres)', 'Product': 'Fuel Type'},
            custom_data='formatted_volume',
            log_y=log_y
        )
        fig_monthly.update_traces(
            hovertemplate='Month: %{x}<br>Fuel Type: %{fullData.name}<br>Monthly Sales Volume: %{y:.2f} (%{customdata})<extra></extra>'
        )
        fig_monthly.update_layout(
            height=450,
            width=800,
            template='plotly_white'
        )
        if log_y:
            max_val = df_monthly['sales_volume'].max()
            tick_vals = [9e7, 1e8, 2e8, 3e8, 4e8, 1e9, 2e9, 3e9]
            tick_vals = [v for v in tick_vals if v <= max_val * 1.1]
            tick_text = ['90M', '100M', '200M', '300M', '400M', '1B', '2B', '3B'][:len(tick_vals)]
            fig_monthly.update_yaxes(type='log', tickvals=tick_vals, ticktext=tick_text)
        fig_monthly.update_xaxes(tickangle=45)
        return fig_monthly

    month_view = st.radio("View Mode", ["Normal", "Lagged"], horizontal=True, key="month_view")
    fig_monthly = create_monthly_sales_chart(df, log_y=(month_view == "Lagged"))
    st.plotly_chart(fig_monthly, use_container_width=True)

    st.markdown("### 💰 Monthly Average Price Trend by Fuel Type")
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def create_price_trend_chart(df):
        # Ensure week_start is datetime type before using .dt accessors
        if not pd.api.types.is_datetime64_any_dtype(df['week_start']):
            df = df.copy()
            df['week_start'] = pd.to_datetime(df['week_start'], errors='coerce')
        
        # Price trend by fuel type over time
        df_price_trend = df.groupby([df['week_start'].dt.to_period('M'), 'Product'])['avg_price'].mean().reset_index()
        df_price_trend['week_start'] = df_price_trend['week_start'].astype(str)
        
        # Function to format price values (prices typically don't need M/B/K suffixes)
        def format_price(value):
            return f"{value:.2f} PKR"
        
        # Add formatted price to the dataframe
        df_price_trend['formatted_price'] = df_price_trend['avg_price'].apply(format_price)
        
        fig_price = px.line(
            df_price_trend,
            x='week_start',
            y='avg_price',
            color='Product',
            title='Monthly Average Price Trend by Fuel Type',
            labels={'week_start': 'Month', 'avg_price': 'Monthly Average Price (PKR)'},
            custom_data='formatted_price'
        )
        fig_price.update_traces(
            hovertemplate='Month: %{x}<br>Product: %{fullData.name}<br>Price: %{y:.2f} PKR (%{customdata})<extra></extra>'
        )
        fig_price.update_layout(
            height=450,
            width=800,
            template='plotly_white'
        )
        fig_price.update_xaxes(tickangle=45)
        return fig_price
    
    # Generate and display the price trend chart
    fig_price = create_price_trend_chart(df)
    st.plotly_chart(fig_price, use_container_width=True)
    

def main():
    # Initialize session state variables
    if "model_params" not in st.session_state:
        st.session_state.model_params = {
            "overall": {
                "train_ratio": 0.8,
                "split_method": "Time-based",
                "feature_selection": True,
                "k_features": 20,
                "lgbm_learning_rate": 0.01,
                "lgbm_n_estimators": 1000,
                "lgbm_max_depth": 7,
                "rf_n_estimators": 200,
                "rf_max_depth": 10,
                "rf_min_samples_split": 2,
                "ensemble_method": "Average",
                "lgbm_weight": 0.5
            },
            "region_fuel": {
                "train_ratio": 0.8,
                "split_method": "Time-based",
                "feature_selection": True,
                "k_features": 15,
                "lgbm_learning_rate": 0.01,
                "lgbm_n_estimators": 500,
                "lgbm_max_depth": 7,
                "rf_n_estimators": 100,
                "rf_max_depth": 10,
                "rf_min_samples_split": 2,
                "ensemble_method": "Average",
                "lgbm_weight": 0.5
            }
        }
    
    if "forecast_params" not in st.session_state:
        st.session_state.forecast_params = {
            "forecast_weeks": 4,
            "forecast_method": "Direct",
            "include_confidence": True,
            "confidence_level": 0.9
        }
    
    if "run_overall" not in st.session_state:
        st.session_state.run_overall = False
    
    if "run_region_fuel" not in st.session_state:
        st.session_state.run_region_fuel = False
        
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
        
    if "default_data_loaded" not in st.session_state:
        st.session_state.default_data_loaded = False
        
    if "overall_metrics" not in st.session_state:
        st.session_state.overall_metrics = None
        
    if "overall_y_test" not in st.session_state:
        st.session_state.overall_y_test = None
        
    if "overall_y_pred" not in st.session_state:
        st.session_state.overall_y_pred = None
        
    if "overall_test_df" not in st.session_state:
        st.session_state.overall_test_df = None
        
    if "rp_results" not in st.session_state:
        st.session_state.rp_results = {}
        
    if "summary_df" not in st.session_state:
        st.session_state.summary_df = None
        
    if "model_selection" not in st.session_state:
        st.session_state.model_selection = "overall"
    
    # Header with animation
    st.markdown('<div class="animate-fade-in">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">⛽ Fuel Forecast Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced analytics and predictive modeling for fuel sales data</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Streamlined file upload in header
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        # Create a container for the upload button
        st.markdown('<div style="display: flex; justify-content: center; margin-bottom: 1rem;">', unsafe_allow_html=True)
        
        # File upload option with minimal footprint
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed", 
                                        help="Upload a CSV file with your fuel sales data")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle file upload and data loading
    if uploaded_file is not None:
        try:
            # Handle date parsing separately to avoid type conversion issues
            weekly_feats = pd.read_csv(uploaded_file)
            
            # Convert week_start to datetime safely and then to string to avoid PyArrow issues
            if 'week_start' in weekly_feats.columns:
                # First convert to datetime
                weekly_feats['week_start'] = pd.to_datetime(weekly_feats['week_start'], errors='coerce')
                # Then convert to string format - only if it's a datetime column
                if pd.api.types.is_datetime64_any_dtype(weekly_feats['week_start']):
                    weekly_feats['week_start'] = weekly_feats['week_start'].dt.strftime('%Y-%m-%d')
                else:
                    # If conversion to datetime failed, ensure it's a string
                    weekly_feats['week_start'] = weekly_feats['week_start'].astype(str)
            
            # Show a toast notification that auto-dismisses
            st.toast("✅ File uploaded successfully", icon="✅")
            
            # Set session state to indicate data is loaded
            st.session_state.data_loaded = True
            
        except Exception as e:
            # Show brief inline error message
            st.error(f"Upload failed: {str(e)[:50]}..." if len(str(e)) > 50 else f"Upload failed: {e}")
            st.stop()
    else:
        # Load default data if no file is uploaded
        try:
            # Handle date parsing during CSV loading to avoid type conversion issues
            weekly_feats = pd.read_csv("weekly_features_no_ogra_for_11_years.csv")
            
            # Ensure week_start is properly formatted as string to avoid PyArrow conversion issues
            if 'week_start' in weekly_feats.columns:
                # First convert to datetime if needed
                if weekly_feats['week_start'].dtype != 'datetime64[ns]':
                    weekly_feats['week_start'] = pd.to_datetime(weekly_feats['week_start'], errors='coerce')
                # Then convert to string format - only if it's a datetime column
                if pd.api.types.is_datetime64_any_dtype(weekly_feats['week_start']):
                    weekly_feats['week_start'] = weekly_feats['week_start'].dt.strftime('%Y-%m-%d')
                else:
                    # If conversion to datetime failed, ensure it's a string
                    weekly_feats['week_start'] = weekly_feats['week_start'].astype(str)
            
            # Only show this message the first time
            if not st.session_state.get('default_data_loaded', False):
                st.info("Using default dataset. Upload your own CSV file for custom analysis.")
                st.session_state.default_data_loaded = True
                
        except FileNotFoundError:
            st.error("Default data file not found. Please upload a CSV file.")
            st.stop()
    
    # Executive Dashboard removed as requested
    
    # Create tabs with centered alignment
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        padding: 1rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: none;
        justify-content: center; /* Center the tabs */
    }
    </style>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Data Overview", 
        "🤖 Model Training & Analysis", 
        "🌍 Region-Fuel Analysis", 
        "🔮 Forecasting",
        "📋 Results & Export"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        st.markdown('<h2 class="section-header">📈 Comprehensive Data Overview</h2>', unsafe_allow_html=True)
        create_comprehensive_data_overview(weekly_feats)
        
        # Data preview
        with st.expander("📋 Detailed Data Preview", expanded=False):
            st.dataframe(prepare_df_for_display(weekly_feats.head(20)))
            
            # Data statistics
            st.markdown("### 📊 Statistical Summary")
            st.dataframe(prepare_df_for_display(weekly_feats.describe()))
            
            # Missing values
            st.markdown("### 🔍 Missing Values Analysis")
            missing_data = weekly_feats.isnull().sum()
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing Percentage': (missing_data.values / len(weekly_feats)) * 100
            })
            st.dataframe(prepare_df_for_display(missing_df))
    
    # Tab 2: Model Training & Analysis (Combined)
    with tab2:
        st.markdown('<h2 class="section-header">🤖 Model Training & Analysis</h2>', unsafe_allow_html=True)
        
        # Add CSS for the model selection tabs - moved outside the model-section div
        st.markdown("""
        <style>
        .model-selection-container {
            background: #ffffff;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }
        
        .model-selection-title {
            font-size: 1.4rem;
            font-weight: 600;
            color: #262730;
            margin-bottom: 1rem;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Model selection title and container - moved outside the model-section div
        st.markdown('<div class="model-selection-title">Select Model Type to Run</div>', unsafe_allow_html=True)
        
        # Initialize session state for model selection if not exists
        if 'model_selection' not in st.session_state:
            st.session_state.model_selection = "overall"
        
        # Create tabs for model selection
        model_tab1, model_tab2 = st.tabs(["🚀 Overall Model", "🔍 Region-Fuel Models"])
        
        # Model training section with better styling - moved inside the tabs
        with model_tab1:
            st.markdown('<div class="model-section">', unsafe_allow_html=True)
            
            # Model Parameter Customization Section
            with st.expander("⚙️ Model Parameters & Training Configuration", expanded=False):
                st.markdown("### Customize Model Parameters")
                
                # Use a form to prevent automatic reruns when UI components change
                with st.form(key="overall_model_form"):
                    # Train-Test Split Configuration
                    st.markdown("#### 📊 Train-Test Split Configuration")
                    col1, col2 = st.columns(2)
                    with col1:
                        train_ratio = st.slider("Training Data Ratio", min_value=0.5, max_value=0.9, 
                                               value=st.session_state.get('model_params', {}).get('overall', {}).get('train_ratio', 0.8), step=0.05, 
                                               help="Proportion of data to use for training (remaining will be used for testing)")
                    with col2:
                        split_method = st.radio("Split Method", ["Time-based", "Random"], 
                                               index=0 if st.session_state.get('model_params', {}).get('overall', {}).get('split_method', "Time-based") == "Time-based" else 1,
                                               help="Time-based: split chronologically, Random: split randomly")
                    
                    # Feature Selection Configuration
                    st.markdown("#### 🔍 Feature Selection Configuration")
                    col1, col2 = st.columns(2)
                    with col1:
                        feature_selection = st.checkbox("Enable Feature Selection", 
                                                      value=st.session_state.get('model_params', {}).get('overall', {}).get('feature_selection', True),
                                                      help="Select most important features for model training")
                    with col2:
                        if feature_selection:
                            k_features = st.slider("Number of Features to Select", min_value=5, max_value=30, 
                                                  value=st.session_state.get('model_params', {}).get('overall', {}).get('k_features', 20), step=1,
                                                  help="Maximum number of features to select based on importance")
                        else:
                            k_features = None
                    
                    # LightGBM Parameters
                    st.markdown("#### 🌲 LightGBM Model Parameters")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        lgbm_learning_rate = st.select_slider("Learning Rate", options=[0.001, 0.005, 0.01, 0.05, 0.1], 
                                                            value=st.session_state.get('model_params', {}).get('overall', {}).get('lgbm_learning_rate', 0.01),
                                                            help="Step size shrinkage to prevent overfitting")
                    with col2:
                        lgbm_n_estimators = st.slider("Number of Trees", min_value=100, max_value=2000, 
                                                    value=st.session_state.get('model_params', {}).get('overall', {}).get('lgbm_n_estimators', 1000), step=100,
                                                    help="Number of boosting iterations")
                    with col3:
                        lgbm_max_depth = st.slider("Max Tree Depth", min_value=3, max_value=15, 
                                                  value=st.session_state.get('model_params', {}).get('overall', {}).get('lgbm_max_depth', 7), step=1,
                                                  help="Maximum depth of trees")
                    
                    # Random Forest Parameters
                    st.markdown("#### 🌳 Random Forest Model Parameters")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        rf_n_estimators = st.slider("Number of Trees (RF)", min_value=50, max_value=500, 
                                                   value=st.session_state.get('model_params', {}).get('overall', {}).get('rf_n_estimators', 200), step=50,
                                                   help="Number of trees in the forest")
                    with col2:
                        rf_max_depth = st.slider("Max Tree Depth (RF)", min_value=3, max_value=20, 
                                                value=st.session_state.get('model_params', {}).get('overall', {}).get('rf_max_depth', 10), step=1,
                                                help="Maximum depth of trees")
                    with col3:
                        rf_min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, 
                                                       value=st.session_state.get('model_params', {}).get('overall', {}).get('rf_min_samples_split', 2), step=1,
                                                       help="Minimum samples required to split a node")
                    
                    # Ensemble Configuration
                    st.markdown("#### 🔄 Ensemble Configuration")
                    col1, col2 = st.columns(2)
                    with col1:
                        ensemble_method = st.radio("Ensemble Method", ["Average", "Weighted Average"], 
                                                  index=0 if st.session_state.get('model_params', {}).get('overall', {}).get('ensemble_method', "Average") == "Average" else 1,
                                                  help="Method to combine model predictions")
                    with col2:
                        if ensemble_method == "Weighted Average":
                            lgbm_weight = st.slider("LightGBM Weight", min_value=0.1, max_value=0.9, 
                                                   value=st.session_state.get('model_params', {}).get('overall', {}).get('lgbm_weight', 0.5), step=0.1,
                                                   help="Weight for LightGBM predictions (RF weight = 1 - LightGBM weight)")
                        else:
                            lgbm_weight = 0.5
                    
                    # Submit button for the form
                    submit_overall = st.form_submit_button("Apply Overall Model Settings")
                    
                    # Only update session state when form is submitted
                    if submit_overall:
                        if "model_params" not in st.session_state:
                            st.session_state.model_params = {}
                        
                        st.session_state.model_params["overall"] = {
                            "train_ratio": train_ratio,
                            "split_method": split_method,
                            "feature_selection": feature_selection,
                            "k_features": k_features,
                            "lgbm_learning_rate": lgbm_learning_rate,
                            "lgbm_n_estimators": lgbm_n_estimators,
                            "lgbm_max_depth": lgbm_max_depth,
                            "rf_n_estimators": rf_n_estimators,
                            "rf_max_depth": rf_max_depth,
                            "rf_min_samples_split": rf_min_samples_split,
                            "ensemble_method": ensemble_method,
                            "lgbm_weight": lgbm_weight
                        }
                
                # Display current model settings
                if st.session_state.get('model_params', {}).get('overall'):
                    st.info(f"Overall model configured with {st.session_state.get('model_params', {}).get('overall', {}).get('split_method', 'Time-based')} split and {st.session_state.get('model_params', {}).get('overall', {}).get('ensemble_method', 'Average')} ensemble method.")
            
            # Run button outside the form to avoid automatic rerun
            if st.button("Run Overall Model", type="primary", use_container_width=True, key="run_overall_button"):
                st.session_state.run_overall = True
                st.session_state.run_region_fuel = False
                
            # Overall Model Training - now inside the white box container
            if st.session_state.get('run_overall', False):
                st.markdown('<h3 class="section-header">📊 Overall Model Training</h3>', unsafe_allow_html=True)
                
                with st.spinner("Training overall ensemble model..."):
                    # Create a custom progress container
                    st.markdown('<div class="progress-container">', unsafe_allow_html=True)
                    progress_bar = st.progress(0)
                    
                    # Simplified progress display with ETA
                    progress_placeholder = st.empty()
                    status_placeholder = st.empty()
                    
                    # Start timing for actual model training
                    start_time = time.time()
                    
                    # Implement the model training with progress updates
                    # Define key stages of model training
                    stages = [
                        "Data preparation",
                        "Feature engineering",
                        "Train-test split",
                        "Feature selection",
                        "LightGBM model training",
                        "Random Forest model training",
                        "Ensemble prediction",
                        "Model evaluation",
                        "Results compilation"
                    ]
                    
                    # Initialize progress tracking
                    total_stages = len(stages)
                    
                    # Prepare data
                    progress_bar.progress(10)
                    progress_placeholder.markdown(
                        f'<div style="text-align: center; color: #ff4b4b;">Processing: 10% | ⏱️ Working...</div>',
                        unsafe_allow_html=True
                    )
                    status_placeholder.markdown(
                        f'<div style="text-align: center; font-size: 0.8rem; color: #4e5663;">Currently processing: {stages[0]}</div>',
                        unsafe_allow_html=True
                    )
                    
                    df = weekly_feats.sort_values('week_start').reset_index(drop=True)
                    
                    # Feature engineering
                    progress_bar.progress(20)
                    progress_placeholder.markdown(
                        f'<div style="text-align: center; color: #ff4b4b;">Processing: 20% | ⏱️ Working...</div>',
                        unsafe_allow_html=True
                    )
                    status_placeholder.markdown(
                        f'<div style="text-align: center; font-size: 0.8rem; color: #4e5663;">Currently processing: {stages[1]}</div>',
                        unsafe_allow_html=True
                    )
                    
                    df_eng = enhanced_feature_engineering(df)
                    
                    # Train-test split
                    progress_bar.progress(30)
                    progress_placeholder.markdown(
                        f'<div style="text-align: center; color: #ff4b4b;">Processing: 30% | ⏱️ Working...</div>',
                        unsafe_allow_html=True
                    )
                    status_placeholder.markdown(
                        f'<div style="text-align: center; font-size: 0.8rem; color: #4e5663;">Currently processing: {stages[2]}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Features to exclude
                    exclude = ['week_start', 'Region', 'Product', 'sales_volume', 'sales_amount']
                    feature_cols = [c for c in df_eng.columns if c not in exclude]
                    
                    # Get train-test split parameters from session state
                    params = st.session_state.model_params.get("overall", {})
                    train_ratio = params.get("train_ratio", 0.8)
                    split_method = params.get("split_method", "Time-based")
                    
                    if split_method == "Time-based":
                        # Time-based split
                        split_point = int(len(df_eng) * train_ratio)
                        train = df_eng.iloc[:split_point]
                        test = df_eng.iloc[split_point:]
                    else:
                        # Random split
                        from sklearn.model_selection import train_test_split
                        train, test = train_test_split(df_eng, test_size=1-train_ratio, random_state=42)
                    
                    X_train = train[feature_cols]
                    y_train = train['sales_volume']
                    X_test = test[feature_cols]
                    y_test = test['sales_volume']
                    
                    # Impute and feature selection
                    progress_bar.progress(40)
                    progress_placeholder.markdown(
                        f'<div style="text-align: center; color: #ff4b4b;">Processing: 40% | ⏱️ Working...</div>',
                        unsafe_allow_html=True
                    )
                    status_placeholder.markdown(
                        f'<div style="text-align: center; font-size: 0.8rem; color: #4e5663;">Currently processing: {stages[3]}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Impute
                    imputer = SimpleImputer(strategy='mean')
                    X_train_clean = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols, index=X_train.index)
                    X_test_clean = pd.DataFrame(imputer.transform(X_test), columns=feature_cols, index=X_test.index)
                    
                    # Get feature selection parameters
                    params = st.session_state.model_params.get("overall", {})
                    feature_selection_enabled = params.get("feature_selection", True)
                    k_features = params.get("k_features", 20)
                    
                    # Feature selection
                    if feature_selection_enabled:
                        k = min(k_features, len(feature_cols))
                        selector = SelectKBest(score_func=f_regression, k=k)
                        X_train_sel = selector.fit_transform(X_train_clean, y_train)
                        X_test_sel = selector.transform(X_test_clean)
                        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
                    else:
                        # Use all features
                        X_train_sel = X_train_clean.values
                        X_test_sel = X_test_clean.values
                        selected_features = feature_cols
                    
                    # LightGBM model training
                    progress_bar.progress(60)
                    progress_placeholder.markdown(
                        f'<div style="text-align: center; color: #ff4b4b;">Processing: 60% | ⏱️ Working...</div>',
                        unsafe_allow_html=True
                    )
                    status_placeholder.markdown(
                        f'<div style="text-align: center; font-size: 0.8rem; color: #4e5663;">Currently processing: {stages[4]}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Get model parameters
                    params = st.session_state.model_params.get("overall", {})
                    lgbm_learning_rate = params.get("lgbm_learning_rate", 0.01)
                    lgbm_n_estimators = params.get("lgbm_n_estimators", 1000)
                    lgbm_max_depth = params.get("lgbm_max_depth", 7)
                    rf_n_estimators = params.get("rf_n_estimators", 200)
                    rf_max_depth = params.get("rf_max_depth", 10)
                    rf_min_samples_split = params.get("rf_min_samples_split", 2)
                    
                    # Ensemble models
                    lgbm = lgb.LGBMRegressor(
                        learning_rate=lgbm_learning_rate, 
                        n_estimators=lgbm_n_estimators, 
                        max_depth=lgbm_max_depth,
                        random_state=42
                    )
                    lgbm.fit(X_train_sel, y_train)
                    
                    # Random Forest model training
                    progress_bar.progress(80)
                    progress_placeholder.markdown(
                        f'<div style="text-align: center; color: #ff4b4b;">Processing: 80% | ⏱️ Working...</div>',
                        unsafe_allow_html=True
                    )
                    status_placeholder.markdown(
                        f'<div style="text-align: center; font-size: 0.8rem; color: #4e5663;">Currently processing: {stages[5]}</div>',
                        unsafe_allow_html=True
                    )
                    
                    rf = RandomForestRegressor(
                        n_estimators=rf_n_estimators, 
                        max_depth=rf_max_depth,
                        min_samples_split=rf_min_samples_split,
                        random_state=42
                    )
                    rf.fit(X_train_sel, y_train)
                    
                    # Ensemble prediction and evaluation
                    progress_bar.progress(90)
                    progress_placeholder.markdown(
                        f'<div style="text-align: center; color: #ff4b4b;">Processing: 90% | ⏱️ Working...</div>',
                        unsafe_allow_html=True
                    )
                    status_placeholder.markdown(
                        f'<div style="text-align: center; font-size: 0.8rem; color: #4e5663;">Currently processing: {stages[6]}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Get ensemble parameters
                    params = st.session_state.model_params.get("overall", {})
                    ensemble_method = params.get("ensemble_method", "Average")
                    lgbm_weight = params.get("lgbm_weight", 0.5)
                    
                    # Make predictions
                    pred_lgbm = lgbm.predict(X_test_sel)
                    pred_rf = rf.predict(X_test_sel)
                    
                    # Combine predictions based on ensemble method
                    if ensemble_method == "Weighted Average":
                        ensemble_pred = (lgbm_weight * pred_lgbm) + ((1 - lgbm_weight) * pred_rf)
                    else:  # Simple average
                        ensemble_pred = (pred_lgbm + pred_rf) / 2
                    
                    # Model evaluation
                    progress_bar.progress(95)
                    progress_placeholder.markdown(
                        f'<div style="text-align: center; color: #ff4b4b;">Processing: 95% | ⏱️ Working...</div>',
                        unsafe_allow_html=True
                    )
                    status_placeholder.markdown(
                        f'<div style="text-align: center; font-size: 0.8rem; color: #4e5663;">Currently processing: {stages[7]}</div>',
                        unsafe_allow_html=True
                    )
                    
                    metrics = evaluate_preds(y_test, ensemble_pred)
                    metrics['selected_features'] = selected_features
                    
                    # Results compilation
                    progress_bar.progress(100)
                    progress_placeholder.markdown(
                        f'<div style="text-align: center; color: #ff4b4b;">Processing: 100% | ⏱️ Complete!</div>',
                        unsafe_allow_html=True
                    )
                    status_placeholder.empty()
                    
                    # Set final results
                    overall_metrics = metrics
                    y_pred = ensemble_pred
                    test_df = test
                    
                    # Close progress container
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Success message with animation
                    st.markdown(
                        '''
                        <div style="display: flex; align-items: center; background: rgba(255, 75, 75, 0.1); 
                        padding: 1rem; border-radius: 10px; border-left: 4px solid #ff4b4b; margin: 1rem 0; animation: fadeIn 0.5s ease-out;">
                            <span style="font-size: 1.5rem; margin-right: 0.8rem;">✅</span>
                            <div>
                                <h4 style="margin: 0; color: #ff4b4b; font-weight: 600;">Model Training Complete</h4>
                                <p style="margin: 0; color: #4e5663; font-size: 0.9rem;">The ensemble model has been successfully trained and evaluated.</p>
                            </div>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
                    
                    # Display metrics
                    st.markdown("### 📊 Overall Model Performance")
                    create_metrics_dashboard(overall_metrics)
                    
                    # Selected features
                    st.markdown("### 🔍 Selected Features")
                    features_df = pd.DataFrame({
                        'Feature': overall_metrics['selected_features']
                    })
                    st.dataframe(prepare_df_for_display(features_df), use_container_width=True)
                    
                    # Training/Testing periods
                    st.markdown("### 📅 Training & Testing Periods")
                    
                    # Calculate correct training and testing periods
                    total_records = len(weekly_feats)
                    split_point = int(total_records * 0.8)
                    
                    # Sort data by date to get correct periods
                    sorted_data = weekly_feats.sort_values('week_start')
                    train_end_date = sorted_data.iloc[split_point - 1]['week_start']
                    test_start_date = sorted_data.iloc[split_point]['week_start']
                    test_end_date = sorted_data.iloc[-1]['week_start']
                    train_start_date = sorted_data.iloc[0]['week_start']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Check if dates are strings or datetime objects and format accordingly
                        train_start_str = train_start_date if isinstance(train_start_date, str) else train_start_date.strftime('%Y-%m-%d')
                        train_end_str = train_end_date if isinstance(train_end_date, str) else train_end_date.strftime('%Y-%m-%d')
                        st.info(f"**Training Period:** {train_start_str} to {train_end_str}")
                    with col2:
                        # Check if dates are strings or datetime objects and format accordingly
                        test_start_str = test_start_date if isinstance(test_start_date, str) else test_start_date.strftime('%Y-%m-%d')
                        test_end_str = test_end_date if isinstance(test_end_date, str) else test_end_date.strftime('%Y-%m-%d')
                        st.info(f"**Testing Period:** {test_start_str} to {test_end_str}")
                    
                    # Display additional information
                    st.markdown("#### 📊 Data Split Information:")
                    st.markdown(f"- **Total Records:** {total_records:,}")
                    st.markdown(f"- **Training Records:** {split_point:,} (80%)")
                    st.markdown(f"- **Testing Records:** {total_records - split_point:,} (20%)")
                    # Convert dates to datetime for subtraction if they're strings
                    train_start_dt = pd.to_datetime(train_start_date) if isinstance(train_start_date, str) else train_start_date
                    train_end_dt = pd.to_datetime(train_end_date) if isinstance(train_end_date, str) else train_end_date
                    st.markdown(f"- **Training Duration:** {(train_end_dt - train_start_dt).days} days")
                    # Convert dates to datetime for subtraction if they're strings
                    test_start_dt = pd.to_datetime(test_start_date) if isinstance(test_start_date, str) else test_start_date
                    test_end_dt = pd.to_datetime(test_end_date) if isinstance(test_end_date, str) else test_end_date
                    st.markdown(f"- **Testing Duration:** {(test_end_dt - test_start_dt).days} days")
                    
                    # Store results in session state
                    st.session_state.overall_metrics = overall_metrics
                    st.session_state.overall_y_test = y_test
                    st.session_state.overall_y_pred = y_pred
                    st.session_state.overall_test_df = test_df

                    # Reset the flag so the overall model is not retrained on every rerun
                    st.session_state.run_overall = False
            st.markdown('</div>', unsafe_allow_html=True)
        
        with model_tab2:
            st.markdown('<div class="model-section">', unsafe_allow_html=True)
            
            # Model Parameter Customization Section for Region-Fuel Models
            with st.expander("⚙️ Model Parameters & Training Configuration", expanded=False):
                st.markdown("### Customize Region-Fuel Model Parameters")
                
                # Use a form to prevent automatic reruns when UI components change
                with st.form(key="region_fuel_model_form"):
                    # Train-Test Split Configuration
                    st.markdown("#### 📊 Train-Test Split Configuration")
                    col1, col2 = st.columns(2)
                    with col1:
                        train_ratio = st.slider("Training Data Ratio (Region-Fuel)", min_value=0.5, max_value=0.9, 
                                               value=st.session_state.get('model_params', {}).get('region_fuel', {}).get('train_ratio', 0.8), step=0.05, 
                                               help="Proportion of data to use for training (remaining will be used for testing)")
                    with col2:
                        split_method = st.radio("Split Method (Region-Fuel)", ["Time-based", "Random"], 
                                               index=0 if st.session_state.get('model_params', {}).get('region_fuel', {}).get('split_method', "Time-based") == "Time-based" else 1,
                                               help="Time-based: split chronologically, Random: split randomly")
                    
                    # Feature Selection Configuration
                    st.markdown("#### 🔍 Feature Selection Configuration")
                    col1, col2 = st.columns(2)
                    with col1:
                        feature_selection = st.checkbox("Enable Feature Selection (Region-Fuel)", 
                                                      value=st.session_state.get('model_params', {}).get('region_fuel', {}).get('feature_selection', True),
                                                      help="Select most important features for model training")
                    with col2:
                        if feature_selection:
                            k_features = st.slider("Number of Features to Select (Region-Fuel)", min_value=5, max_value=30, 
                                                  value=st.session_state.get('model_params', {}).get('region_fuel', {}).get('k_features', 15), step=1,
                                                  help="Maximum number of features to select based on importance")
                        else:
                            k_features = None
                    
                    # LightGBM Parameters
                    st.markdown("#### 🌲 LightGBM Model Parameters")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        lgbm_learning_rate = st.select_slider("Learning Rate (Region-Fuel)", options=[0.001, 0.005, 0.01, 0.05, 0.1], 
                                                            value=st.session_state.get('model_params', {}).get('region_fuel', {}).get('lgbm_learning_rate', 0.01),
                                                            help="Step size shrinkage to prevent overfitting")
                    with col2:
                        lgbm_n_estimators = st.slider("Number of Trees (Region-Fuel)", min_value=100, max_value=1000, 
                                                    value=st.session_state.get('model_params', {}).get('region_fuel', {}).get('lgbm_n_estimators', 500), step=100,
                                                    help="Number of boosting iterations")
                    with col3:
                        lgbm_max_depth = st.slider("Max Tree Depth (Region-Fuel)", min_value=3, max_value=15, 
                                                  value=st.session_state.get('model_params', {}).get('region_fuel', {}).get('lgbm_max_depth', 7), step=1,
                                                  help="Maximum depth of trees")
                    
                    # Random Forest Parameters
                    st.markdown("#### 🌳 Random Forest Model Parameters")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        rf_n_estimators = st.slider("Number of Trees (RF Region-Fuel)", min_value=50, max_value=300, 
                                                   value=st.session_state.get('model_params', {}).get('region_fuel', {}).get('rf_n_estimators', 100), step=50,
                                                   help="Number of trees in the forest")
                    with col2:
                        rf_max_depth = st.slider("Max Tree Depth (RF Region-Fuel)", min_value=3, max_value=20, 
                                                value=st.session_state.get('model_params', {}).get('region_fuel', {}).get('rf_max_depth', 10), step=1,
                                                help="Maximum depth of trees")
                    with col3:
                        rf_min_samples_split = st.slider("Min Samples Split (Region-Fuel)", min_value=2, max_value=10, 
                                                       value=st.session_state.get('model_params', {}).get('region_fuel', {}).get('rf_min_samples_split', 2), step=1,
                                                       help="Minimum samples required to split a node")
                    
                    # Ensemble Configuration
                    st.markdown("#### 🔄 Ensemble Configuration")
                    col1, col2 = st.columns(2)
                    with col1:
                        ensemble_method = st.radio("Ensemble Method (Region-Fuel)", ["Average", "Weighted Average"], 
                                                  index=0 if st.session_state.get('model_params', {}).get('region_fuel', {}).get('ensemble_method', "Average") == "Average" else 1,
                                                  help="Method to combine model predictions")
                    with col2:
                        if ensemble_method == "Weighted Average":
                            lgbm_weight = st.slider("LightGBM Weight (Region-Fuel)", min_value=0.1, max_value=0.9, 
                                                   value=st.session_state.get('model_params', {}).get('region_fuel', {}).get('lgbm_weight', 0.5), step=0.1,
                                                   help="Weight for LightGBM predictions (RF weight = 1 - LightGBM weight)")
                        else:
                            lgbm_weight = 0.5
                    
                    # Submit button for the form
                    submit_region_fuel = st.form_submit_button("Apply Region-Fuel Model Settings")
                    
                    # Only update session state when form is submitted
                    if submit_region_fuel:
                        if "model_params" not in st.session_state:
                            st.session_state.model_params = {}
                        
                        st.session_state.model_params["region_fuel"] = {
                            "train_ratio": train_ratio,
                            "split_method": split_method,
                            "feature_selection": feature_selection,
                            "k_features": k_features,
                            "lgbm_learning_rate": lgbm_learning_rate,
                            "lgbm_n_estimators": lgbm_n_estimators,
                            "lgbm_max_depth": lgbm_max_depth,
                            "rf_n_estimators": rf_n_estimators,
                            "rf_max_depth": rf_max_depth,
                            "rf_min_samples_split": rf_min_samples_split,
                            "ensemble_method": ensemble_method,
                            "lgbm_weight": lgbm_weight
                        }
                
                # Display current model settings
                if st.session_state.get('model_params', {}).get('region_fuel'):
                    st.info(f"Region-Fuel models configured with {st.session_state.get('model_params', {}).get('region_fuel', {}).get('split_method', 'Time-based')} split and {st.session_state.get('model_params', {}).get('region_fuel', {}).get('ensemble_method', 'Average')} ensemble method.")
            
            # Removed Forecasting Configuration - moved to dedicated tab
            
            # Run button outside the form to avoid automatic rerun
            # Wrap in a container to prevent overlay effect
            with st.container():
                if st.button("Run Region-Fuel Models", type="secondary", use_container_width=True, key="run_region_fuel_button"):
                    # Set flags in session state
                    st.session_state.run_region_fuel = True
                    st.session_state.run_overall = False
                    # No rerun here - will be handled by the training section below
                
            # Region-Fuel Model Training - inside the white box container
            if st.session_state.get('run_region_fuel', False):
                st.markdown('<h3 class="section-header">🌍 Region-Fuel Model Training</h3>', unsafe_allow_html=True)
                
                with st.spinner("Training region-fuel models..."):
                    # Calculate total combinations for ETA
                    total_combinations = len(weekly_feats.groupby(['Region', 'Product']))
                    
                    # Create an info box for the training process
                    st.markdown(
                        f'''
                        <div class="info-box animate-fade-in">
                            <div style="display: flex; align-items: center;">
                                <span style="font-size: 1.5rem; margin-right: 0.8rem;">📊</span>
                                <div>
                                    <h4 style="margin: 0; color: #ff4b4b; font-weight: 600;">Training Multiple Models</h4>
                                    <p style="margin: 0; color: #4e5663; font-size: 0.9rem;">Analyzing {total_combinations} region-fuel combinations with ensemble learning</p>
                                </div>
                            </div>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
                    
                    # Create a custom progress container
                    st.markdown('<div class="progress-container">', unsafe_allow_html=True)
                    progress_bar = st.progress(0)
                    
                    # Simplified progress display with ETA
                    progress_placeholder = st.empty()
                    status_placeholder = st.empty()
                    
                    # Start timing for actual model training
                    start_time = time.time()
                    
                    # Run the actual model with progress updates
                    rp_results = {}
                    
                    # Get all region-product combinations
                    combinations = list(weekly_feats.groupby(['Region', 'Product']))
                    total_combinations = len(combinations)
                    
                    # Process each combination with progress updates
                    for idx, ((region, product), group) in enumerate(combinations):
                        # Skip if too few data points
                        if len(group) < 20:
                            continue
                            
                        # Update progress percentage
                        progress_percent = int((idx + 1) / total_combinations * 100)
                        progress_bar.progress(progress_percent)
                        
                        # Calculate and display ETA
                        elapsed = time.time() - start_time
                        if idx > 0:
                            eta = (elapsed / (idx + 1)) * (total_combinations - (idx + 1))
                            minutes, seconds = divmod(eta, 60)
                            
                            # Update the same placeholder instead of creating new elements
                            progress_placeholder.markdown(
                                f'<div style="text-align: center; color: #ff4b4b;">Processing: {progress_percent}% | ⏱️ ETA: {int(minutes):02d}:{int(seconds):02d}</div>',
                                unsafe_allow_html=True
                            )
                            
                            # Show current combination being processed
                            status_placeholder.markdown(
                                f'<div style="text-align: center; font-size: 0.8rem; color: #4e5663;">Currently processing: {region} - {product}</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Process this combination
                        group = group.sort_values('week_start').reset_index(drop=True)
                        # Use forecasting_mode=False for initial feature engineering
                        group_eng = enhanced_feature_engineering(group, forecasting_mode=False)
                        
                        # Features to exclude
                        exclude = ['week_start', 'Region', 'Product', 'sales_volume', 'sales_amount']
                        feature_cols = [c for c in group_eng.columns if c not in exclude]
                        
                        # Get model parameters from session state
                        params = st.session_state.model_params.get("region_fuel", {})
                        train_ratio = params.get("train_ratio", 0.8)
                        split_method = params.get("split_method", "Time-based")
                        feature_selection_enabled = params.get("feature_selection", True)
                        k_features = params.get("k_features", 15)
                        lgbm_learning_rate = params.get("lgbm_learning_rate", 0.01)
                        lgbm_n_estimators = params.get("lgbm_n_estimators", 500)
                        lgbm_max_depth = params.get("lgbm_max_depth", 7)
                        rf_n_estimators = params.get("rf_n_estimators", 100)
                        rf_max_depth = params.get("rf_max_depth", 10)
                        rf_min_samples_split = params.get("rf_min_samples_split", 2)
                        ensemble_method = params.get("ensemble_method", "Average")
                        lgbm_weight = params.get("lgbm_weight", 0.5)
                        
                        # Train-test split based on selected method
                        if split_method == "Time-based":
                            split_point = int(len(group_eng) * train_ratio)
                            train = group_eng.iloc[:split_point]
                            test = group_eng.iloc[split_point:]
                            
                            # Store train/test date ranges for this combination
                            if 'week_start' in train.columns and len(train) > 0 and len(test) > 0:
                                train_start_date = train['week_start'].min()
                                train_end_date = train['week_start'].max()
                                test_start_date = test['week_start'].min()
                                test_end_date = test['week_start'].max()
                        else:  # Random split
                            train, test = train_test_split(group_eng, train_size=train_ratio, random_state=42)
                            
                            # For random split, we can't determine chronological ranges
                            # but we can still track the number of samples
                            train_start_date = None
                            train_end_date = None
                            test_start_date = None
                            test_end_date = None
                        
                        X_train = train[feature_cols]
                        y_train = train['sales_volume']
                        X_test = test[feature_cols]
                        y_test = test['sales_volume']
                        
                        # Impute
                        imputer = SimpleImputer(strategy='mean')
                        X_train_clean = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols, index=X_train.index)
                        X_test_clean = pd.DataFrame(imputer.transform(X_test), columns=feature_cols, index=X_test.index)
                        
                        # Feature selection
                        if feature_selection_enabled:
                            k = min(k_features, len(feature_cols))
                            selector = SelectKBest(score_func=f_regression, k=k)
                            X_train_sel = selector.fit_transform(X_train_clean, y_train)
                            X_test_sel = selector.transform(X_test_clean)
                            selected_features = X_train_clean.columns[selector.get_support()].tolist()
                        else:
                            X_train_sel = X_train_clean.values
                            X_test_sel = X_test_clean.values
                            selected_features = X_train_clean.columns.tolist()
                        
                        # Ensemble models
                        lgbm = lgb.LGBMRegressor(
                            learning_rate=lgbm_learning_rate, 
                            n_estimators=lgbm_n_estimators, 
                            max_depth=lgbm_max_depth,
                            random_state=42
                        )
                        rf = RandomForestRegressor(
                            n_estimators=rf_n_estimators, 
                            max_depth=rf_max_depth,
                            min_samples_split=rf_min_samples_split,
                            random_state=42
                        )
                        
                        lgbm.fit(X_train_sel, y_train)
                        rf.fit(X_train_sel, y_train)
                        
                        pred_lgbm = lgbm.predict(X_test_sel)
                        pred_rf = rf.predict(X_test_sel)
                        
                        # Combine predictions based on ensemble method
                        if ensemble_method == "Weighted Average":
                            ensemble_pred = (lgbm_weight * pred_lgbm) + ((1 - lgbm_weight) * pred_rf)
                        else:  # Simple average
                            ensemble_pred = (pred_lgbm + pred_rf) / 2
                        
                        metrics = evaluate_preds(y_test, ensemble_pred)
                        
                        # Store model results
                        result_data = {
                            'metrics': metrics,
                            'y_test': y_test,
                            'y_pred': ensemble_pred,
                            'test_data': test,
                            'selected_features': selected_features,
                            'models': {
                                'lgbm': lgbm,
                                'rf': rf
                            },
                            'imputer': imputer,
                            'selector': selector if feature_selection_enabled else None,
                            'feature_cols': feature_cols,
                            'feature_selection_enabled': feature_selection_enabled,
                            'ensemble_method': ensemble_method,
                            'lgbm_weight': lgbm_weight,
                            'train_test_dates': {
                                'train_start_date': train_start_date,
                                'train_end_date': train_end_date,
                                'test_start_date': test_start_date,
                                'test_end_date': test_end_date,
                                'split_method': split_method
                            }
                        }
                        
                        rp_results[(region, product)] = result_data
                    
                    # Set progress to 100% when complete
                    progress_bar.progress(100)
                    progress_placeholder.markdown(
                        f'<div style="text-align: center; color: #ff4b4b;">Processing: 100% | ⏱️ Complete!</div>',
                        unsafe_allow_html=True
                    )
                    status_placeholder.empty()
                    
                    # Close progress container
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Success message with animation and count
                    st.markdown(
                        f'''
                        <div style="display: flex; align-items: center; background: rgba(255, 75, 75, 0.1); 
                        padding: 1rem; border-radius: 10px; border-left: 4px solid #ff4b4b; margin: 1rem 0; animation: fadeIn 0.5s ease-out;">
                            <span style="font-size: 1.5rem; margin-right: 0.8rem;">✅</span>
                            <div>
                                <h4 style="margin: 0; color: #ff4b4b; font-weight: 600;">Region-Fuel Analysis Complete</h4>
                                <p style="margin: 0; color: #4e5663; font-size: 0.9rem;">Successfully analyzed {len(rp_results)} region-fuel combinations with ensemble models.</p>
                            </div>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
                    
                    # Store results in session state
                    st.session_state.rp_results = rp_results
                    
                    # Set the model training timestamp when models are actually trained
                    st.session_state.model_training_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Create summary table
                    summary_data = []
                    for (region, product), res in rp_results.items():
                        m = res['metrics']
                        summary_data.append({
                            'Region': region,
                            'Product': product,
                            'MAE': m['MAE'],
                            'MAPE (%)': m['MAPE'] * 100,
                            'R²': m['R2'],
                            'SMAPE (%)': m['SMAPE']
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.session_state.summary_df = summary_df

                    # Reset the flag so that region‑fuel models are not retrained on every rerun
                    st.session_state.run_region_fuel = False
                    
                    # Display summary
                    st.markdown("### 📊 Region-Fuel Performance Summary")
                    
                    # Use helper function to prepare DataFrame for display
                    summary_df_display = prepare_df_for_display(summary_df)
                    
                    st.dataframe(summary_df_display.sort_values('MAE'), use_container_width=True)
                    
                    # Performance distribution
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_mae = px.histogram(summary_df, x='MAE', title='MAE Distribution', nbins=20)
                        st.plotly_chart(fig_mae, use_container_width=True)
                    
                    with col2:
                        fig_r2 = px.histogram(summary_df, x='R²', title='R² Distribution', nbins=20)
                        st.plotly_chart(fig_r2, use_container_width=True)
                    
                    # Best and worst performers
                    st.markdown("### 🏆 Top Performers")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Best MAE:**")
                        best_mae = summary_df.loc[summary_df['MAE'].idxmin()]
                        st.info(f"{best_mae['Region']} - {best_mae['Product']}: MAE = {best_mae['MAE']:.2f}")
                    
                    with col2:
                        st.markdown("**Best R²:**")
                        best_r2 = summary_df.loc[summary_df['R²'].idxmax()]
                        st.info(f"{best_r2['Region']} - {best_r2['Product']}: R² = {best_r2['R²']:.3f}")
                    
                    # Display training/testing information for region-fuel models
                    st.markdown("### 📅 Region-Fuel Model Training Information:")
                    st.markdown(f"- **Total Region-Fuel Combinations:** {len(rp_results)}")
                    
                    # Get split method and ratio from the first result (should be the same for all)
                    first_result = next(iter(rp_results.values()))
                    split_method = first_result['train_test_dates']['split_method']
                    
                    # Calculate the actual train ratio from the model parameters
                    params = st.session_state.model_params.get("region_fuel", {})
                    train_ratio = params.get("train_ratio", 0.8)
                    test_ratio = 1 - train_ratio
                    
                    st.markdown(f"- **Split Method:** {split_method}")
                    st.markdown(f"- **Training Split:** {train_ratio*100:.0f}% of data for each combination")
                    st.markdown(f"- **Testing Split:** {test_ratio*100:.0f}% of data for each combination")
                    
                    # Display date ranges if using time-based split
                    if split_method == "Time-based" and first_result['train_test_dates']['train_start_date'] is not None:
                        train_start = first_result['train_test_dates']['train_start_date']
                        train_end = first_result['train_test_dates']['train_end_date']
                        test_start = first_result['train_test_dates']['test_start_date']
                        test_end = first_result['train_test_dates']['test_end_date']
                        
                        # Convert to datetime if they are strings
                        train_start = pd.to_datetime(train_start) if isinstance(train_start, str) else train_start
                        train_end = pd.to_datetime(train_end) if isinstance(train_end, str) else train_end
                        test_start = pd.to_datetime(test_start) if isinstance(test_start, str) else test_start
                        test_end = pd.to_datetime(test_end) if isinstance(test_end, str) else test_end
                        
                        st.markdown(f"- **Training Period:** {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
                        st.markdown(f"- **Testing Period:** {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
                    
                    st.markdown(f"- **Models Trained:** {len(rp_results)} individual models")
                    st.markdown(f"- **Ensemble Approach:** LightGBM + Random Forest for each combination")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Overall Model Training section has been moved inside the model_tab1
        
        # Region-Fuel Model Training section has been moved inside the model_tab2
    
    # Tab 3: Region-Fuel Analysis
    with tab3:
        st.markdown('<h2 class="section-header">🌍 Region-Fuel Specific Analysis</h2>', unsafe_allow_html=True)
        
        # Region-Fuel specific visualizations
        if 'rp_results' in st.session_state and st.session_state.rp_results is not None:
            st.markdown("### 🎯 Region-Fuel Specific Predictions")
            
            # Select region-product for detailed analysis
            rp_options = list(st.session_state.rp_results.keys())
            
            # Use a session state variable to track the previously selected combination
            if 'selected_rp' not in st.session_state:
                st.session_state.selected_rp = rp_options[0] if rp_options else None
                
            # Use the selectbox with the current value from session state
            selected_rp = st.selectbox(
                "Select Region-Fuel Combination for Detailed Analysis", 
                rp_options,
                index=rp_options.index(st.session_state.selected_rp) if st.session_state.selected_rp in rp_options else 0
            )
            
            # Update the session state with the new selection
            st.session_state.selected_rp = selected_rp
            
            if selected_rp in st.session_state.rp_results:
                res = st.session_state.rp_results[selected_rp]
                
                # Use the cached metrics dashboard function
                create_metrics_dashboard(res['metrics'])
                
                # Display training and testing periods for this combination
                st.markdown("#### 📅 Training and Testing Information")
                
                # Get split method and date ranges
                split_method = res['train_test_dates']['split_method']
                st.markdown(f"**Split Method:** {split_method}")
                
                # Display date ranges if using time-based split
                if split_method == "Time-based" and res['train_test_dates']['train_start_date'] is not None:
                    train_start = res['train_test_dates']['train_start_date']
                    train_end = res['train_test_dates']['train_end_date']
                    test_start = res['train_test_dates']['test_start_date']
                    test_end = res['train_test_dates']['test_end_date']
                    
                    # Convert to datetime if they are strings
                    train_start = pd.to_datetime(train_start) if isinstance(train_start, str) else train_start
                    train_end = pd.to_datetime(train_end) if isinstance(train_end, str) else train_end
                    test_start = pd.to_datetime(test_start) if isinstance(test_start, str) else test_start
                    test_end = pd.to_datetime(test_end) if isinstance(test_end, str) else test_end
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Training Period:** {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
                    with col2:
                        st.info(f"**Testing Period:** {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
                else:
                    # For random split, show the number of samples
                    train_samples = len(res['test_data']) * 4  # Approximation based on 80/20 split
                    test_samples = len(res['test_data'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Training Samples:** {train_samples}")
                    with col2:
                        st.info(f"**Testing Samples:** {test_samples}")
                
                # Use st.cache_resource for the plot to avoid recalculation
                # Actual vs Predicted plot
                fig_pred = plot_actual_vs_predicted(
                    res['test_data'], 
                    res['y_test'], 
                    res['y_pred'], 
                    selected_rp[0], 
                    selected_rp[1]
                )
                if fig_pred:
                    # Update layout for better scaling
                    fig_pred.update_layout(
                        height=450,
                        width=800,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                
                # Additional analysis for selected combination
                st.markdown(f"### 📊 Detailed Analysis: {selected_rp[0]} - {selected_rp[1]}")
                
                # Cache the filtered data for better performance
                @st.cache_data(ttl=3600)
                def get_filtered_combo_data(region, product):
                    return weekly_feats[
                        (weekly_feats['Region'] == region) & 
                        (weekly_feats['Product'] == product)
                    ].sort_values('week_start')
                
                # Filter data for selected combination using the cached function
                combo_data = get_filtered_combo_data(selected_rp[0], selected_rp[1])
                
                # Ensure week_start is in the correct format for visualization
                combo_data = combo_data.copy()
                if pd.api.types.is_datetime64_any_dtype(combo_data['week_start']):
                    combo_data['week_start_str'] = combo_data['week_start'].dt.strftime('%Y-%m-%d')
                else:
                    combo_data['week_start_str'] = combo_data['week_start'].astype(str)
                
                if not combo_data.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Function to format values with appropriate unit labels
                        def format_value_with_unit(value):
                            if isinstance(value, (np.ndarray, list, tuple, pd.Series)):
                                arr = np.asarray(value)
                                value = float(arr.ravel()[0]) if arr.size > 0 else 0.0
                            else:
                                value = float(value)
                            if value >= 1_000_000_000:
                                return f"{value/1_000_000_000:.2f}B"
                            elif value >= 1_000_000:
                                return f"{value/1_000_000:.2f}M"
                            elif value >= 1_000:
                                return f"{value/1_000:.2f}K"
                            else:
                                return f"{value:.2f}"
                        
                        # Sales volume trend
                        # Add formatted values as a column first
                        combo_data['formatted_volume'] = combo_data['sales_volume'].apply(format_value_with_unit)
                        
                        fig_trend = px.line(
                            combo_data, 
                            x='week_start_str', 
                            y='sales_volume',
                            title=f'Weekly Sales Volume Trend: {selected_rp[0]} - {selected_rp[1]}',
                            custom_data='formatted_volume'
                        )
                        fig_trend.update_xaxes(title_text='Week Start Date')
                        fig_trend.update_traces(
                            hovertemplate='Date: %{x}<br>Sales Volume: %{y:.2f} (%{customdata})<extra></extra>'
                        )
                        fig_trend.update_layout(
                            height=400,
                            width=400,
                            template='plotly_white'
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)
                    
                    with col2:
                        # Price vs Volume scatter
                        # Add formatted values as a column first
                        combo_data['formatted_volume_scatter'] = combo_data['sales_volume'].apply(format_value_with_unit)
                        
                        fig_scatter = px.scatter(
                            combo_data,
                            x='avg_price',
                            y='sales_volume',
                            title=f'Weekly Price vs Volume: {selected_rp[0]} - {selected_rp[1]}',
                            labels={'avg_price': 'Price (PKR)', 'sales_volume': 'Weekly Sales Volume (Litres)'},
                            custom_data='formatted_volume_scatter'
                        )
                        fig_scatter.update_traces(
                            hovertemplate='Price: %{x:.2f} PKR<br>Sales Volume: %{y:.2f} (%{customdata})<extra></extra>'
                        )
                        fig_scatter.update_layout(
                            height=400,
                            width=400,
                            template='plotly_white'
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Note: Forecasting has been moved to the dedicated Forecasting tab
                st.info("ℹ️ Forecasting functionality has been moved to the dedicated '🔮 Forecasting' tab.")
                if 'forecast' in res:
                    st.success("✅ Forecasts are available for this combination. Please check the '🔮 Forecasting' tab to view them.")
                else:
                    st.warning("⚠️ No forecasts available for this combination. Enable forecasting in the '🔮 Forecasting' tab.")
        else:
            st.info("🔍 Please run the Region-Fuel Models first to see detailed analysis.")
    
    # Tab 4: Forecasting
    with tab4:
        st.markdown('<h2 class="section-header">🔮 Forecasting</h2>', unsafe_allow_html=True)
        
        # Check if models are trained
        if 'rp_results' not in st.session_state or st.session_state.rp_results is None:
            st.warning("⚠️ Please train Region-Fuel models first before using the forecasting feature.")
            st.info("Go to the 'Model Training & Analysis' tab to train models.")
        else:
            # Display model training information
            st.markdown("### 🤖 Trained Model Information")
            
            # Show when models were trained
            if 'model_training_time' not in st.session_state and 'rp_results' in st.session_state and st.session_state.rp_results:
                # Set the timestamp when models are actually trained, not when the page is loaded
                st.session_state.model_training_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
            st.info(f"Models were last trained at: {st.session_state.model_training_time if 'model_training_time' in st.session_state else 'Not trained yet'}")
            
            # Option to discard trained models and retrain
            if st.button("Discard Trained Models and Re-train", type="secondary"):
                # Clear model results
                if 'rp_results' in st.session_state:
                    del st.session_state.rp_results
                if 'summary_df' in st.session_state:
                    del st.session_state.summary_df
                if 'model_training_time' in st.session_state:
                    del st.session_state.model_training_time
                st.session_state.run_region_fuel = True
                st.rerun()
            
            # Forecasting Configuration
            st.markdown("### ⚙️ Configure Forecasting Options")
            
            # Use a form to group forecasting options
            with st.form(key="forecasting_form"):
                col1, col2 = st.columns(2)
                with col1:
                    forecast_weeks = st.slider("Number of Weeks to Forecast", min_value=1, max_value=260, 
                                             value=st.session_state.get('forecast_params', {}).get('forecast_weeks', 4), step=1,
                                             help="Number of weeks to forecast into the future (maximum 5 years/260 weeks)")
                with col2:
                    forecast_method = st.radio("Forecast Method", ["Direct", "Recursive"], 
                                             index=0 if st.session_state.get('forecast_params', {}).get('forecast_method', "Direct") == "Direct" else 1,
                                             help="Direct: forecast all weeks at once, Recursive: use previous forecasts as inputs")
                
                # Advanced forecasting options
                st.markdown("#### Advanced Forecasting Options")
                col1, col2 = st.columns(2)
                with col1:
                    include_confidence = st.checkbox("Include Confidence Intervals", 
                                                  value=st.session_state.get('forecast_params', {}).get('include_confidence', True),
                                                  help="Show prediction uncertainty ranges")
                with col2:
                    if include_confidence:
                        confidence_level = st.select_slider("Confidence Level", options=[0.8, 0.85, 0.9, 0.95, 0.99], 
                                                          value=st.session_state.get('forecast_params', {}).get('confidence_level', 0.9),
                                                          help="Confidence level for prediction intervals")
                    else:
                        confidence_level = 0.9
                
                # Submit button for the form
                submit_forecast = st.form_submit_button("Apply Forecasting Settings", use_container_width=True)
                
            # Update session state when form is submitted
            if submit_forecast:
                if "forecast_params" not in st.session_state:
                    st.session_state.forecast_params = {}
                
                st.session_state.forecast_params = {
                    "forecast_weeks": forecast_weeks,
                    "forecast_method": forecast_method,
                    "include_confidence": include_confidence,
                    "confidence_level": confidence_level
                }
                
                # Display success message outside the form to prevent overlay
                st.success(f"Forecasting settings applied successfully.")
            
            # Display current forecasting settings
            if 'forecast_params' in st.session_state:
                st.info(f"Current settings: Forecasting {st.session_state.forecast_params.get('forecast_weeks', 4)} weeks using {st.session_state.forecast_params.get('forecast_method', 'Direct')} method.")
            
            # Run Forecasting button outside the form with container width to prevent overlay
            if st.button("Run Forecasting", type="primary", use_container_width=True):
                # Create a progress bar with a container to prevent UI issues
                with st.container():
                    # Create progress elements
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Get all region-product combinations
                    rp_combinations = list(st.session_state.rp_results.keys())
                    total_combinations = len(rp_combinations)
                    
                    # Flag to track if we've processed all combinations
                    all_processed = True
                    
                    # Run forecasting for all combinations
                    for i, (region, product) in enumerate(rp_combinations):
                        # Update progress display
                        progress = int((i / total_combinations) * 100)
                        progress_bar.progress(progress)
                        progress_text.markdown(f"**Processing: {progress}%** | Running forecasts...", unsafe_allow_html=False)
                        status_text.markdown(f"Currently processing: **{region} - {product}**", unsafe_allow_html=False)
                        
                        # Get the result data for this combination
                        res = st.session_state.rp_results[(region, product)]
                        
                        # Always generate forecasts to ensure latest parameters are applied
                        try:
                            forecast_params = st.session_state.forecast_params
                            forecast_weeks = forecast_params.get('forecast_weeks', 4)
                            forecast_method = forecast_params.get('forecast_method', 'Direct')
                            include_confidence = forecast_params.get('include_confidence', False)
                            confidence_level = forecast_params.get('confidence_level', 0.9)

                            group = weekly_feats[
                                (weekly_feats['Region'] == region) &
                                (weekly_feats['Product'] == product)
                            ].sort_values('week_start').reset_index(drop=True).copy()

                            # Ensure datetime type for feature engineering
                            if not pd.api.types.is_datetime64_any_dtype(group['week_start']):
                                group['week_start'] = pd.to_datetime(group['week_start'])

                            # Use deterministic feature engineering for historical data
                            group_eng = enhanced_feature_engineering(group, forecasting_mode=False)

                            forecast = generate_forecast(
                                res,
                                group_eng,
                                product,
                                forecast_weeks,
                                forecast_method,
                                include_confidence,
                                confidence_level,
                            )

                            res['forecast'] = forecast
                            st.session_state.rp_results[(region, product)] = res
                        except Exception as e:
                            st.error(f"Error processing {region} - {product}: {str(e)}")
                            all_processed = False
                    
                    # Set progress to 100% when complete
                    progress_bar.progress(100)
                    progress_text.markdown("**Processing: 100%** | Complete!", unsafe_allow_html=False)
                    status_text.empty()
                    
                    # Success message
                    if all_processed:
                        st.success("✅ Forecasting completed for all region-fuel combinations!")
                    else:
                        st.warning("⚠️ Forecasting completed with some errors. Please check the messages above.")
            
            # Forecasting Visualization
            if 'rp_results' in st.session_state and any('forecast' in res for res in st.session_state.rp_results.values()):
                st.markdown("### 🎯 Region-Fuel Specific Forecasts")
                
                # Select region-product for detailed analysis
                rp_options = list(st.session_state.rp_results.keys())
                
                # Use a session state variable to track the previously selected combination
                if 'selected_forecast_rp' not in st.session_state:
                    st.session_state.selected_forecast_rp = rp_options[0] if rp_options else None
                
                # Create a container for the dropdown to prevent UI overlay
                with st.container():
                    # Use the selectbox with the current value from session state
                    selected_rp = st.selectbox(
                        "Select Region-Fuel Combination for Forecast", 
                        rp_options,
                        index=rp_options.index(st.session_state.selected_forecast_rp) if st.session_state.selected_forecast_rp in rp_options else 0,
                        key="forecast_rp_selector"
                    )
                
                # Update the session state with the new selection
                st.session_state.selected_forecast_rp = selected_rp
                
                if selected_rp in st.session_state.rp_results:
                    res = st.session_state.rp_results[selected_rp]
                    
                    # Display forecasts if available
                    if 'forecast' in res:
                        if not res['forecast']['values']:
                            st.warning("No forecast generated for this combination. Please run the forecasting step again.")
                        else:
                            st.markdown(f"#### 🔮 Sales Volume Forecast: {selected_rp[0]} - {selected_rp[1]}")

                            # Normalize forecast results to plain floats
                            raw_vals = res['forecast']['values']
                            raw_lowers = res['forecast']['lower_bounds']
                            raw_uppers = res['forecast']['upper_bounds']
                            norm_vals = [float(np.asarray(v).ravel()[0]) for v in raw_vals]
                            norm_lowers = [float(np.asarray(v).ravel()[0]) if v is not None else None for v in raw_lowers]
                            norm_uppers = [float(np.asarray(v).ravel()[0]) if v is not None else None for v in raw_uppers]

                            # Replace original forecast entries with normalized scalars
                            res['forecast']['values'] = norm_vals
                            res['forecast']['lower_bounds'] = norm_lowers
                            res['forecast']['upper_bounds'] = norm_uppers

                            # Create a DataFrame for the forecast with string dates
                            forecast_data = []
                            for i in range(len(norm_vals)):
                                row = {
                                    'Date': res['forecast']['dates_str'][i],
                                    'Forecast': norm_vals[i]
                                }
                                if norm_lowers[i] is not None:
                                    row['Lower Bound'] = norm_lowers[i]
                                    row['Upper Bound'] = norm_uppers[i]
                                forecast_data.append(row)

                            forecast_df = pd.DataFrame(forecast_data)

                            # Function to format values with appropriate unit labels
                            def format_value_with_unit(value):
                                if isinstance(value, (np.ndarray, list, tuple, pd.Series)):
                                    arr = np.asarray(value)
                                    value = float(arr.ravel()[0]) if arr.size > 0 else 0.0
                                else:
                                    value = float(value)
                                if value >= 1_000_000_000:
                                    return f"{value/1_000_000_000:.2f}B"
                                elif value >= 1_000_000:
                                    return f"{value/1_000_000:.2f}M"
                                elif value >= 1_000:
                                    return f"{value/1_000:.2f}K"
                                else:
                                    return f"{value:.2f}"

                            # Display forecast table with formatted values for readability
                            forecast_display = forecast_df.copy()
                            for col in ['Forecast', 'Lower Bound', 'Upper Bound']:
                                if col in forecast_display:
                                    forecast_display[col] = forecast_display[col].map(format_value_with_unit)
                            st.dataframe(prepare_df_for_display(forecast_display), use_container_width=True)

                            # Create forecast visualization
                            fig_forecast = go.Figure()
                        
                        
                        
                        # Get historical data for this combination
                        combo_data = weekly_feats[
                            (weekly_feats['Region'] == selected_rp[0]) & 
                            (weekly_feats['Product'] == selected_rp[1])
                        ].sort_values('week_start')
                        
                        # Ensure week_start is in the correct format for visualization
                        if pd.api.types.is_datetime64_any_dtype(combo_data['week_start']):
                            combo_data['week_start_str'] = combo_data['week_start'].dt.strftime('%Y-%m-%d')
                        else:
                            combo_data['week_start_str'] = combo_data['week_start'].astype(str)
                        
                        # Add historical data
                        historical_data = combo_data.sort_values('week_start')
                        fig_forecast.add_trace(go.Scatter(
                            x=historical_data['week_start_str'],
                            y=historical_data['sales_volume'],
                            mode='lines+markers',
                            name='Historical',
                            line=dict(color='blue'),
                            hovertemplate='Date: %{x}<br>Sales Volume: %{y:.2f} (%{customdata})<extra></extra>',
                            customdata=[format_value_with_unit(val) for val in historical_data['sales_volume']]
                        ))
                        
                        # Add forecast
                        fig_forecast.add_trace(go.Scatter(
                            x=res['forecast']['dates_str'],
                            y=norm_vals,
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color='red', dash='dash'),
                            hovertemplate='Date: %{x}<br>Forecast: %{y:.2f} (%{customdata})<extra></extra>',
                            customdata=[format_value_with_unit(val) for val in norm_vals]
                        ))
                        
                        # Only draw the confidence interval if bounds are available
                        if norm_lowers and norm_lowers[0] is not None:
                            # Create x values for the confidence interval (forward then backward)
                            x_conf = res['forecast']['dates_str'] + res['forecast']['dates_str'][::-1]
                            # Create y values for the confidence interval (upper bounds then lower bounds reversed)
                            y_conf = norm_uppers + norm_lowers[::-1]

                            fig_forecast.add_trace(go.Scatter(
                                x=x_conf,
                                y=y_conf,
                                fill='toself',
                                fillcolor='rgba(255,0,0,0.2)',
                                line=dict(color='rgba(255,0,0,0)'),
                                name=f"Confidence Interval ({int(res['forecast']['confidence_level']*100)}%)",
                                hoverinfo='skip'
                            ))
                        
                        # Update layout
                        fig_forecast.update_layout(
                            title=f'Sales Volume Forecast: {selected_rp[0]} - {selected_rp[1]}',
                            xaxis_title='Date',
                            yaxis_title='Sales Volume',
                            height=500,
                            template='plotly_white',
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        
                        # Display forecast information
                        st.markdown("#### Forecast Information")
                        st.markdown(f"- **Forecast Method:** {res['forecast']['method']}")
                        st.markdown(f"- **Forecast Horizon:** {len(res['forecast']['dates'])} weeks")
                        if res['forecast']['confidence_level'] is not None:
                            st.markdown(f"- **Confidence Level:** {int(res['forecast']['confidence_level']*100)}%")
                        
                        # Download forecast as CSV
                        try:
                            csv = forecast_df.to_csv(index=False)
                            st.download_button(
                                label="Download Forecast as CSV",
                                data=csv,
                                file_name=f"{selected_rp[0]}_{selected_rp[1]}_forecast.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"Error generating CSV: {e}")
                            st.info("Please try again or contact support if the issue persists.")
                    else:
                        st.warning("No forecast available for this combination. Please run forecasting for this region-fuel combination.")
            else:
                st.info("🔍 Please run the Region-Fuel Models first to see forecasts.")
    
    # Tab 5: Results & Export
    with tab5:
        st.markdown('<h2 class="section-header">📋 Model Results & Export</h2>', unsafe_allow_html=True)
        
        if 'summary_df' in st.session_state and st.session_state.summary_df is not None:
            st.markdown("### 📊 Complete Model Results")
            # Use helper function to prepare DataFrame for display
            display_df = prepare_df_for_display(st.session_state.summary_df)
            st.dataframe(display_df.sort_values('MAE'), use_container_width=True)
            
            # Download results
            try:
                # Use helper function to prepare DataFrame for download
                if st.session_state.summary_df is not None:
                    download_df = prepare_df_for_display(st.session_state.summary_df.copy())
                    
                    csv = download_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="region_fuel_analysis_results.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error generating CSV: {e}")
                st.info("Please try again or contact support if the issue persists.")
        
        if 'overall_metrics' in st.session_state and st.session_state.overall_metrics is not None:
            st.markdown("### 🤖 Overall Model Results")
            create_metrics_dashboard(st.session_state.overall_metrics)
            
            # Download overall results
            overall_results = pd.DataFrame([st.session_state.overall_metrics])
            csv_overall = overall_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Overall Results as CSV",
                data=csv_overall,
                file_name="overall_model_results.csv",
                mime="text/csv"
            )
    
    # Footer removed as requested

if __name__ == "__main__":
    main()