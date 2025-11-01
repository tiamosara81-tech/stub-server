# pmu_pipeline.py
# متطلبات: pip install requests pandas scikit-learn xgboost
import requests, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss
import xgboost as xgb
import numpy as np

# 1) تحميل نتائج يومية عبر open-pmu-api (مثال)
def fetch_results(date_str):
    # API مفتوح (مثال من GitHub project open-pmu-api)
    url = f"https://open-pmu-api.vercel.app/results?date={date_str}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

# 2) تحويل JSON إلى DataFrame مبسّط
def build_dataframe(json_results):
    rows = []
    for race in json_results.get('races', []):
        race_id = race.get('id') or race.get('raceId')
        for pos, starter in enumerate(race.get('starters', []), start=1):
            row = {
                'race_id': race_id,
                'horse': starter.get('horse_name'),
                'jockey': starter.get('jockey'),
                'trainer': starter.get('trainer'),
                'draw': starter.get('draw'),
                'weight': starter.get('weight'),
                'fin_pos': starter.get('finishing_position') if starter.get('finishing_position') else 99,
                'odds': float(starter.get('odds')) if starter.get('odds') else np.nan,
                # هنا تضيف ميزات مشتقة لاحقاً
            }
            rows.append(row)
    return pd.DataFrame(rows)

# 3) بناء ميزات بسيطة
def featurize(df):
    # مثال: هدف = الفوز (fin_pos == 1)
    df['target_win'] = (df['fin_pos'] == 1).astype(int)
    # ميزات مبسطة: draw, weight, implied_odds
    df['draw'] = pd.to_numeric(df['draw'], errors='coerce').fillna(0)
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(df['weight'].median())
    df['implied_prob_market'] = 1.0 / df['odds'].replace(0, np.nan)
    df['implied_prob_market'].fillna(df['implied_prob_market'].median(), inplace=True)
    # تحويلات سريعة للفارس/مدرب (count-based)
    for col in ['jockey','trainer','horse']:
        freq = df[col].value_counts().to_dict()
        df[f'{col}_freq'] = df[col].map(freq).fillna(0)
    return df

# 4) نموذج XGBoost لتنبؤ الاحتمال
def train_model(df):
    features = ['draw','weight','implied_prob_market','jockey_freq','trainer_freq','horse_freq']
    X = df[features].values
    y = df['target_win'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {'objective':'binary:logistic','eval_metric':'logloss','seed':42}
    bst = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest,'eval')], early_stopping_rounds=10, verbose_eval=False)
    preds = bst.predict(dtest)
    print("LogLoss:", log_loss(y_test, preds))
    print("Brier:", brier_score_loss(y_test, preds))
    return bst, features

# Example flow
if __name__ == "__main__":
    date = "2025-10-31"   # عدّل إلى التاريخ المطلوب
    j = fetch_results(date)
    df = build_dataframe(j)
    df = featurize(df)
    model, features = train_model(df)
    # حفظ النموذج أو استعماله لاحقاً لتصنيف الاحتمالات لكل سباق
