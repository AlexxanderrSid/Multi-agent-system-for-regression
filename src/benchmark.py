import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder

from src.config import RANDOM_STATE


def run_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """5-fold CV по базовым моделям для сравнения с мультиагентной системой."""
    print('\n' + '='*50)
    print('BENCHMARK — базовые модели')
    print('='*50)

    X = df.drop(columns=['target']).copy()
    y = df['target']

    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.select_dtypes(include=[np.number]).fillna(X.median(numeric_only=True))

    kf = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    baselines = {
        'Ridge': Ridge(),
        'RandomForest': RandomForestRegressor(100, random_state=RANDOM_STATE),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
        'LightGBM (defaults)': lgb.LGBMRegressor(n_estimators=100, random_state=RANDOM_STATE, verbose=-1),
    }

    rows = []
    for name, reg in baselines.items():
        sc = cross_val_score(reg, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
        mse = -sc.mean()
        rows.append({'Model': name, 'MSE': mse, 'Std': sc.std(), 'Type': 'Baseline'})
        print(f'  {name:30s}: MSE={mse:.4f} ± {sc.std():.4f}')

    return pd.DataFrame(rows)


def compare(bench: pd.DataFrame, mas_score: float) -> pd.DataFrame:
    mas_row = pd.DataFrame([{
        'Model': 'MultiAgent (Qwen2.5-Coder)',
        'MSE': mas_score, 'Std': 0.0, 'Type': 'MultiAgent'
    }])
    result = pd.concat([bench, mas_row], ignore_index=True)\
               .sort_values('MSE', ascending=True)
    print('\nИтоговое сравнение (MSE, меньше = лучше):')
    print(result.to_string(index=False))
    delta = mas_score - bench['MSE'].min()
    print(f'\n  Разница с лучшим baseline: {delta:+.4f} (отрицательное = улучшение)')
    return result
