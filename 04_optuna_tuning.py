"""
ADIM 4.2: HİPERPARAMETRE OPTİMİZASYONU (OPTUNA - BAYESYEN ARAMA)
==============================================================
Neden GridSearch Değil?
  GridSearch körleme her kombinasyonu dener (saatler sürer, verimsizdir).
Neden RandomizedSearch Değil?
  Rastgele dener, geçmiş denemelerden ders çıkarmaz.
Neden Optuna?
  Bayesyen Optimizasyon (TPE Algoritması) kullanır. Başarılı olan
  parametre bölgelerine odaklanır, başarısız bölgeleri terk eder.

Çalıştırma: python 04_optuna_tuning.py
"""
import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")

# Optuna'nın gürültülü ara loglarını sustur
optuna.logging.set_verbosity(optuna.logging.WARNING)

t0 = time.time()
VERI_DIZINI = Path(__file__).resolve().parent / "data"

print("=" * 70)
print("OPTUNA İLE AKILLI HİPERPARAMETRE OPTİMİZASYONU BAŞLIYOR")
print("=" * 70)

# 1) Veriyi Oku ve Böl
df = pd.read_parquet(VERI_DIZINI / "train_master_features.parquet")

tum_kullanicilar = df["user_id"].unique()
np.random.seed(42)
np.random.shuffle(tum_kullanicilar)
train_users = set(tum_kullanicilar[:int(len(tum_kullanicilar) * 0.80)])
val_users = set(tum_kullanicilar[int(len(tum_kullanicilar) * 0.80):])

train_df = df[df["user_id"].isin(train_users)]
val_df = df[df["user_id"].isin(val_users)]

features = [c for c in df.columns if c not in ["user_id", "product_id", "reordered"]]
X_train = train_df[features].fillna(0)
y_train = train_df["reordered"]
X_val = val_df[features].fillna(0)
y_val = val_df["reordered"]

# 2) Optuna Hedef Fonksiyonu (Objective)
deneme_sayaci = 0

def objective(trial):
    global deneme_sayaci
    deneme_sayaci += 1

    # Optuna'nın deneyeceği akıllı parametre aralıkları:
    params = {
        "n_estimators": 200,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 40),
        "subsample": trial.suggest_float("subsample", 0.65, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "early_stopping_rounds": 15,
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    val_preds = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, val_preds)

    print(f"  [Deneme {deneme_sayaci:>2d}/12] AUC: {score:.4f} | max_depth: {params['max_depth']} | lr: {params['learning_rate']:.3f} | subsample: {params['subsample']:.2f}")
    return score

# 3) Optuna Çalışmasını Başlat
print("\n>>> Optuna Denemeleri Başlıyor (12 Akıllı Tur)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=12)

print("\n" + "=" * 70)
print("OPTUNA OPTİMİZASYONU TAMAMLANDI!")
print("=" * 70)
print(f"  • En İyi Doğrulama Skoru (Best Val AUC): {study.best_value:.4f}")
print("  • Bulunan En Optimum Hiperparametreler:")
for param, deger in study.best_params.items():
    if isinstance(deger, float):
        print(f"    - {param:<20}: {deger:.4f}")
    else:
        print(f"    - {param:<20}: {deger}")

print("\n" + "=" * 70)
print("BUYUK GELISIM TABLOSU:")
print("  1. Baseline (Lojistik Regresyon)  : 0.8158 AUC")
print("  2. XGBoost (Elle Secilen)         : 0.8233 AUC")
print(f"  3. XGBoost (Optuna ile Ayarlanan) : {study.best_value:.4f} AUC [EN IYI]")
print(f"  * Toplam Gecen Sure               : {time.time() - t0:.2f} saniye")
print("=" * 70)
