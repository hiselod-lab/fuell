import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb


def enhanced_feature_engineering(df, forecasting_mode=False, random_state=42):
    """Create deterministic features for model training and forecasting."""
    df = df.copy()
    df['price_elasticity'] = np.where(df['avg_price'] != 0,
                                      df['sales_volume'] / df['avg_price'],
                                      np.nan)
    df['price_volatility'] = df.groupby(['Region', 'Product'])['avg_price'].transform(
        lambda x: x.rolling(4, min_periods=1).std()
    )
    df['price_change'] = df['avg_price'].pct_change()

    df['volume_trend'] = df.groupby(['Region', 'Product'])['sales_volume'].transform(
        lambda x: x.rolling(8, min_periods=1).mean()
    )
    df['volume_change'] = df['sales_volume'].pct_change()
    df['volume_volatility'] = df.groupby(['Region', 'Product'])['sales_volume'].transform(
        lambda x: x.rolling(4, min_periods=1).std()
    )

    df['seasonal_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['seasonal_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    df['quarterly_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 13)
    df['quarterly_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 13)
    df['monthly_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['monthly_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['trend_factor'] = df['week_of_year'] / 52
    df['trend_squared'] = (df['week_of_year'] / 52) ** 2

    df['regional_trend'] = df.groupby('Region')['sales_volume'].transform(
        lambda x: x.rolling(12, min_periods=1).mean()
    )
    df['product_trend'] = df.groupby('Product')['sales_volume'].transform(
        lambda x: x.rolling(12, min_periods=1).mean()
    )

    df['price_volume_ratio'] = df['avg_price'] * df['sales_volume']
    df['price_over_volume'] = np.where(df['sales_volume'] != 0,
                                       df['avg_price'] / df['sales_volume'],
                                       np.nan)

    if forecasting_mode:
        df['weather_factor'] = 1.0
        df['economic_factor'] = 1.0
        df['event_factor'] = 1.0
    else:
        np.random.seed(random_state)
        df['weather_factor'] = np.random.normal(1, 0.1, len(df))
        df['economic_factor'] = np.random.normal(1, 0.05, len(df))
        df['event_factor'] = np.random.normal(1, 0.15, len(df))

    for window in [4, 8, 12]:
        df[f'roll{window}_mean'] = df.groupby(['Region', 'Product'])['sales_volume'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'roll{window}_std'] = df.groupby(['Region', 'Product'])['sales_volume'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

    return df


def evaluate_preds(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = r2_score(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom))
    return {'MAE': mae, 'MAPE': mape, 'RMSE': rmse, 'R2': r2, 'SMAPE': smape}


def train_models(df, feature_cols, params, train_ratio=0.8, split_method='time'):
    if split_method == 'time':
        split_point = int(len(df) * train_ratio)
        train = df.iloc[:split_point]
        test = df.iloc[split_point:]
    else:
        train, test = train_test_split(df, train_size=train_ratio, random_state=42, shuffle=True)

    X_train, y_train = train[feature_cols], train['sales_volume']
    X_test, y_test = test[feature_cols], test['sales_volume']

    imputer = SimpleImputer(strategy='mean')
    X_train_clean = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_clean = pd.DataFrame(imputer.transform(X_test), columns=feature_cols, index=X_test.index)

    selector = None
    if params.get('feature_selection', True):
        k = min(params.get('k_features', 20), len(feature_cols))
        selector = SelectKBest(f_regression, k=k)
        X_train_sel = selector.fit_transform(X_train_clean, y_train)
        X_test_sel = selector.transform(X_test_clean)
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
    else:
        X_train_sel, X_test_sel = X_train_clean.values, X_test_clean.values
        selected_features = feature_cols

    lgbm = lgb.LGBMRegressor(
        learning_rate=params.get('lgbm_learning_rate', 0.01),
        n_estimators=params.get('lgbm_n_estimators', 1000),
        max_depth=params.get('lgbm_max_depth', 7),
        random_state=42
    )
    rf = RandomForestRegressor(
        n_estimators=params.get('rf_n_estimators', 200),
        max_depth=params.get('rf_max_depth', 10),
        min_samples_split=params.get('rf_min_samples_split', 2),
        random_state=42
    )

    lgbm.fit(X_train_sel, y_train)
    rf.fit(X_train_sel, y_train)

    pred_lgbm = lgbm.predict(X_test_sel)
    pred_rf = rf.predict(X_test_sel)
    if params.get('ensemble_method', 'Average') == 'Weighted Average':
        weight = params.get('lgbm_weight', 0.5)
        ensemble_pred = weight * pred_lgbm + (1 - weight) * pred_rf
    else:
        ensemble_pred = (pred_lgbm + pred_rf) / 2

    metrics = evaluate_preds(y_test, ensemble_pred)
    residual_std = float(np.std(y_test - ensemble_pred, ddof=1)) if len(y_test) > 1 else 0.0

    # Refit preprocessing and models on full data
    imputer_full = SimpleImputer(strategy='mean')
    X_full = df[feature_cols]
    X_full_clean = pd.DataFrame(imputer_full.fit_transform(X_full), columns=feature_cols, index=df.index)
    if selector is not None:
        selector_full = SelectKBest(f_regression, k=len(selected_features))
        X_full_sel = selector_full.fit_transform(X_full_clean, df['sales_volume'])
        final_features = [feature_cols[i] for i in selector_full.get_support(indices=True)]
    else:
        selector_full = None
        X_full_sel = X_full_clean.values
        final_features = feature_cols

    lgbm_full = lgb.LGBMRegressor(
        learning_rate=params.get('lgbm_learning_rate', 0.01),
        n_estimators=params.get('lgbm_n_estimators', 1000),
        max_depth=params.get('lgbm_max_depth', 7),
        random_state=42
    )
    rf_full = RandomForestRegressor(
        n_estimators=params.get('rf_n_estimators', 200),
        max_depth=params.get('rf_max_depth', 10),
        min_samples_split=params.get('rf_min_samples_split', 2),
        random_state=42
    )
    lgbm_full.fit(X_full_sel, df['sales_volume'])
    rf_full.fit(X_full_sel, df['sales_volume'])

    model_bundle = {
        'models': {'lgbm': lgbm_full, 'rf': rf_full},
        'imputer': imputer_full,
        'selector': selector_full,
        'feature_cols': feature_cols,
        'selected_features': final_features,
        'ensemble_method': params.get('ensemble_method', 'Average'),
        'lgbm_weight': params.get('lgbm_weight', 0.5),
        'residual_std': residual_std,
        'metrics': metrics
    }
    return model_bundle


def generate_forecast(model_bundle, history, steps, include_confidence=True, confidence_level=0.95):
    models = model_bundle['models']
    lgbm = models['lgbm']
    rf = models['rf']
    imputer = model_bundle['imputer']
    selector = model_bundle['selector']
    feature_cols = model_bundle['feature_cols']
    ensemble_method = model_bundle['ensemble_method']
    lgbm_weight = model_bundle['lgbm_weight']
    residual_std = model_bundle['residual_std']

    history = history.copy().sort_values('week_start')
    if not pd.api.types.is_datetime64_any_dtype(history['week_start']):
        history['week_start'] = pd.to_datetime(history['week_start'])

    forecasts, dates, lower, upper = [], [], [], []
    z_lookup = {0.99: 2.58, 0.95: 1.96, 0.9: 1.645, 0.8: 1.28}
    z = z_lookup.get(confidence_level, 1.96)

    for _ in range(steps):
        next_date = history['week_start'].iloc[-1] + timedelta(days=7)
        new_row = history.iloc[-1:].copy()
        new_row['week_start'] = next_date
        new_row['sales_volume'] = np.nan
        history = pd.concat([history, new_row], ignore_index=True)
        eng = enhanced_feature_engineering(history, forecasting_mode=True)
        X = eng[feature_cols].iloc[-1:]
        X_clean = pd.DataFrame(imputer.transform(X), columns=feature_cols)
        if selector is not None:
            X_sel = selector.transform(X_clean)
        else:
            X_sel = X_clean.values
        pred_lgbm = lgbm.predict(X_sel)
        pred_rf = rf.predict(X_sel)
        if ensemble_method == 'Weighted Average':
            pred = lgbm_weight * pred_lgbm + (1 - lgbm_weight) * pred_rf
        else:
            pred = (pred_lgbm + pred_rf) / 2
        pred = float(pred[0])
        history.loc[history.index[-1], 'sales_volume'] = pred
        forecasts.append(pred)
        dates.append(next_date)
        if include_confidence:
            margin = residual_std * z
            lower.append(pred - margin)
            upper.append(pred + margin)
    return pd.DataFrame({
        'week_start': dates,
        'forecast': forecasts,
        'lower': lower if include_confidence else None,
        'upper': upper if include_confidence else None
    })
