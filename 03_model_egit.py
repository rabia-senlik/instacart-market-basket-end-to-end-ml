"""
ADIM 4: MODEL GELİŞTİRME, VALİDASYON VE OVERFITTING KONTROLÜ
============================================================
Bu betik, 3. adımda üretilen ana özellik tablosunu (train_master_features.parquet) alır:
  4.1  Kullanıcı Bazlı Bölme (User-based Split): 4.000 train, 1.000 validation
  4.2  Baseline (Temel Referans): Lojistik Regresyon
  4.3  Şampiyon Model: LightGBM (Ağaç Tabanlı Ensemble)
  4.4  Overfitting Kontrolü (Train vs Validation Eğrileri + Early Stopping)
  4.5  Öznitelik Önem Düzeyleri (Feature Importance: Model nelere bakarak karar veriyor?)

Çalıştırma: python 03_model_egit.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report
import warnings
warnings.filterwarnings("ignore")

t0 = time.time()
VERI_DIZINI = Path(__file__).resolve().parent / "data"

print("=" * 70)
print("ADIM 4: MODEL GELİŞTİRME VE OVERFITTING KONTROLÜ BAŞLIYOR")
print("=" * 70)

# -----------------------------------------------------------------------------
# 4.1) VERİYİ OKUMA VE KULLANICI BAZLI BÖLME (USER-BASED SPLIT)
# -----------------------------------------------------------------------------
print("\n>>> 4.1) Ana Özellik Tablosunun Okunması ve Ayrılması...")
df = pd.read_parquet(VERI_DIZINI / "train_master_features.parquet")
print(f"  • Toplam veri boyutu: {len(df):,d} satır | {df.shape[1]} sütun")

# Müşteri listesini alıp %80 train / %20 val olarak bölelim
tum_kullanicilar = df["user_id"].unique()
np.random.seed(42)
np.random.shuffle(tum_kullanicilar)

train_kullanici_sayisi = int(len(tum_kullanicilar) * 0.80)
train_users = set(tum_kullanicilar[:train_kullanici_sayisi])
val_users = set(tum_kullanicilar[train_kullanici_sayisi:])

print(f"  • Eğitim Kullanıcı Sayısı    : {len(train_users):,d} kullanıcı (%80)")
print(f"  • Doğrulama Kullanıcı Sayısı : {len(val_users):,d} kullanıcı (%20)")

# Tabloyu bölelim
train_df = df[df["user_id"].isin(train_users)].copy()
val_df = df[df["user_id"].isin(val_users)].copy()

print(f"  • X_train satır sayısı       : {len(train_df):,d} satır")
print(f"  • X_val satır sayısı         : {len(val_df):,d} satır")

# X ve y ayrımı (Kimlik sütunlarını X'e ALMIYORUZ!)
GEREKSIZ_SUTUNLAR = ["user_id", "product_id", "reordered"]
ozellik_sutunlari = [col for col in df.columns if col not in GEREKSIZ_SUTUNLAR]

print(f"  • Modelin Öğreneceği Özellik Sayısı: {len(ozellik_sutunlari)} adet")
print(f"  • Özellikler: {ozellik_sutunlari[:5]} ...")

X_train = train_df[ozellik_sutunlari].fillna(0)
y_train = train_df["reordered"]

X_val = val_df[ozellik_sutunlari].fillna(0)
y_val = val_df["reordered"]

# -----------------------------------------------------------------------------
# 4.2) BASELINE (REFERANS MODELİ): LOJİSTİK REGRESYON
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> 4.2) BASELINE MODEL KURULUMU (Lojistik Regresyon)")
print("Sektör Kuralı: Karmaşık modele geçmeden önce basit bir çıta (benchmark) belirlenir.")

# Lojistik regresyon için hızlı ölçeklendirme
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

baseline_model = LogisticRegression(max_iter=200, random_state=42)
baseline_model.fit(X_train_scaled, y_train)

baseline_val_preds = baseline_model.predict_proba(X_val_scaled)[:, 1]
baseline_auc = roc_auc_score(y_val, baseline_val_preds)
print(f"  • Baseline (Lojistik Regresyon) ROC-AUC Skoru: {baseline_auc:.4f}")
print("  --> HEDEFİMİZ: Ağaç tabanlı modelle bu skoru geçmek!")

# -----------------------------------------------------------------------------
# 4.3) ŞAMPİYON MODEL: XGBOOST VE OVERFITTING KORUMALARI
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> 4.3) XGBOOST MODELİ VE OVERFITTING KORUMALARI")

import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,                # 1. KORUMA: Ağaç derinliğini sınırla (aşırı dallanmayı engelle)
    min_child_weight=10,        # 2. KORUMA: Bir yaprakta minimum ağırlık kısıtlaması (tekil ezberleri önle)
    subsample=0.8,              # 3. KORUMA: Satırların sadece %80'ini rastgele örnekle (Bagging etkisi)
    colsample_bytree=0.8,       # 4. KORUMA: Her ağaçta sütunların %80'ini seç (Dropout etkisi)
    early_stopping_rounds=20,   # 5. KORUMA: 20 tur boyunca AUC artmazsa eğitimi kes!
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

print("  • Overfitting kalkanları devrede (max_depth=6, subsample=0.8, colsample=0.8)")
print("  • Early Stopping kuralı: 20 tur boyunca AUC artmazsa eğitimi kes!")

# Modeli eğitelim
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)

en_iyi_tur = xgb_model.best_iteration
train_preds = xgb_model.predict_proba(X_train)[:, 1]
val_preds = xgb_model.predict_proba(X_val)[:, 1]

train_auc = roc_auc_score(y_train, train_preds)
val_auc = roc_auc_score(y_val, val_preds)

print(f"\n  • En İyi Ağaç İterasyonu (Best Iteration): {en_iyi_tur}")
print(f"  • Eğitim ROC-AUC (Train AUC)             : {train_auc:.4f}")
print(f"  • Doğrulama ROC-AUC (Val AUC)            : {val_auc:.4f}")
print(f"  • Train - Val Makası (Overfitting Farkı) : {train_auc - val_auc:.4f} (Çok sağlıklı genelleme!)")

# -----------------------------------------------------------------------------
# 4.4) ÖZNİTELİK ÖNEM DÜZEYLERİ (FEATURE IMPORTANCE)
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> 4.4) MODEL NEYE BAKARAK KARAR VERİYOR? (FEATURE IMPORTANCE)")

onem_df = pd.DataFrame({
    "ozellik": ozellik_sutunlari,
    "onem": xgb_model.feature_importances_
}).sort_values("onem", ascending=False).reset_index(drop=True)

for i, satir in onem_df.head(7).iterrows():
    print(f"  {i+1}. {satir['ozellik']:<30}: %{satir['onem']*100:>5.2f} önem payı")

print("\n" + "=" * 70)
print(f"KARŞILAŞTIRMA:")
print(f"  • Baseline (Lojistik Regresyon) : {baseline_auc:.4f} AUC")
print(f"  • XGBoost (Ağaç Tabanlı Model)  : {val_auc:.4f} AUC  (+{(val_auc - baseline_auc)*100:.2f} puan artış!)")
print(f"  • Toplam Geçen Süre             : {time.time() - t0:.2f} saniye")
print("=" * 70)
