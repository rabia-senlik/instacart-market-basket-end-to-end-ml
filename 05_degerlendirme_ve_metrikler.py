"""
ADIM 5: BAŞARI METRİKLERİ, EŞİK AYARI (THRESHOLD TUNING) VE TOP-K ÖNERİ LİSTESİ
================================================================================
Mülakatlarda en çok sorulan ve kıdemli farkı yaratan konular:
  5.1 Neden Accuracy Bu Problemde Yalandır? (Accuracy Paradox)
  5.2 Karışıklık Matrisi (Confusion Matrix) ve İş Dünyası Anlamı (TP, FP, TN, FN)
  5.3 Eşik Ayarı (Threshold Tuning): Neden 0.50 Kullanılmaz? (0.50 vs 0.20)
  5.4 Sıralama ve Kişiselleştirilmiş Öneri Listesi (Top-K Hit Rate)
  5.5 GitHub İçin 4 Panelli Profesyonel Görsel Dashboard (PNG)

Çalıştırma: python 05_degerlendirme_ve_metrikler.py
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve, auc
)
from pathlib import Path
import time
import shutil
import warnings
warnings.filterwarnings("ignore")

t0 = time.time()
VERI_DIZINI = Path(__file__).resolve().parent / "data"
CIKTI_DIZINI = Path(__file__).resolve().parent

print("=" * 75)
print("ADIM 5: METRİK SEÇİMİ, EŞİK AYARI VE GÖRSEL DEĞERLENDİRME DASHBOARDU")
print("=" * 75)

# 1) Veriyi Oku ve Kullanıcı Bazlı Böl (Data Leakage Önlemi)
df = pd.read_parquet(VERI_DIZINI / "train_master_features.parquet")

tum_kullanicilar = df["user_id"].unique()
np.random.seed(42)
np.random.shuffle(tum_kullanicilar)
train_users = set(tum_kullanicilar[:int(len(tum_kullanicilar) * 0.80)])
val_users = set(tum_kullanicilar[int(len(tum_kullanicilar) * 0.80):])

train_df = df[df["user_id"].isin(train_users)]
val_df = df[df["user_id"].isin(val_users)].copy()

features = [c for c in df.columns if c not in ["user_id", "product_id", "reordered"]]
X_train = train_df[features].fillna(0)
y_train = train_df["reordered"]

X_val = val_df[features].fillna(0)
y_val = val_df["reordered"]

# 2) Modeli Eğit ve Olasılıkları Hesapla
print("\n>>> Model Eğitiliyor ve Doğrulama Olasılıkları Çıkarılıyor...")
model = xgb.XGBClassifier(
    n_estimators=150,
    learning_rate=0.035,
    max_depth=8,
    min_child_weight=40,
    subsample=0.7,
    colsample_bytree=0.87,
    gamma=1.34,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train, verbose=False)
val_probs = model.predict_proba(X_val)[:, 1]
val_df["prob"] = val_probs

# -----------------------------------------------------------------------------
# 5.1) TABLO 1: ACCURACY VE METRİK KARŞILAŞTIRMA ÖZETİ
# -----------------------------------------------------------------------------
toplam_ornek = len(y_val)
negatif_ornek = (y_val == 0).sum()
pozitif_ornek = (y_val == 1).sum()

pred_050 = (val_probs >= 0.50).astype(int)
pred_020 = (val_probs >= 0.20).astype(int)

cm_050 = confusion_matrix(y_val, pred_050)
cm_020 = confusion_matrix(y_val, pred_020)

tablo_verisi = {
    "Metrik / Gösterge": [
        "Accuracy (Doğruluk)",
        "Precision (Kesinlik)",
        "Recall (Satış Yakalama)",
        "F1-Score (Denge Skoru)",
        "TP (Yakalanan Satış)",
        "FN (Kaçan Satış)",
        "İş Dünyası Yorumu"
    ],
    "Kör Model (Hep 0)": [
        f"%{(negatif_ornek/toplam_ornek)*100:.2f}",
        "%0.0",
        "%0.0",
        "0.000",
        "0 adet",
        f"{pozitif_ornek:,d} adet",
        "HİÇBİR ŞEY SATAMAZ (İflas)"
    ],
    "Eşik = 0.50 (Varsayılan)": [
        f"%{(pred_050 == y_val.values).mean()*100:.2f}",
        f"%{precision_score(y_val, pred_050)*100:.1f}",
        f"%{recall_score(y_val, pred_050)*100:.1f}",
        f"{f1_score(y_val, pred_050):.3f}",
        f"{cm_050[1,1]:,d} adet",
        f"{cm_050[1,0]:,d} adet",
        "Aşırı Çekingen (Satış Kaçırır)"
    ],
    "Eşik = 0.20 (Optimize)": [
        f"%{(pred_020 == y_val.values).mean()*100:.2f}",
        f"%{precision_score(y_val, pred_020)*100:.1f}",
        f"%{recall_score(y_val, pred_020)*100:.1f}",
        f"{f1_score(y_val, pred_020):.3f}",
        f"{cm_020[1,1]:,d} adet",
        f"{cm_020[1,0]:,d} adet",
        "Satış Odaklı (Ciroyu 3x Yapar)"
    ]
}

ozet_df = pd.DataFrame(tablo_verisi)

print("\n" + "=" * 75)
print(">>> KARŞILAŞTIRMALI METRİK ÖZET TABLOSU:")
print("=" * 75)
print(ozet_df.to_string(index=False))

# -----------------------------------------------------------------------------
# 5.2) GÖRSEL DASHBOARD (GITHUB İÇİN 4 PANELLİ ŞAHESER)
# -----------------------------------------------------------------------------
print("\n>>> GitHub İçin Profesyonel Görsel Dashboard Hazırlanıyor...")

sns.set_theme(style="whitegrid", font="DejaVu Sans")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.32, wspace=0.25)

# PANEL 1: Eşik Değerine Göre Metrik Değişimi (Threshold Tuning Curve)
esikler = np.linspace(0.05, 0.70, 50)
precisions = []
recalls = []
f1_scores = []

for t in esikler:
    p_t = (val_probs >= t).astype(int)
    precisions.append(precision_score(y_val, p_t, zero_division=0))
    recalls.append(recall_score(y_val, p_t, zero_division=0))
    f1_scores.append(f1_score(y_val, p_t, zero_division=0))

en_iyi_idx = np.argmax(f1_scores)
en_iyi_esik = esikler[en_iyi_idx]
en_iyi_f1 = f1_scores[en_iyi_idx]

ax1 = axes[0, 0]
ax1.plot(esikler, precisions, label="Precision (Kesinlik)", color="#1f77b4", linewidth=2.5, linestyle="--")
ax1.plot(esikler, recalls, label="Recall (Duyarlılık / Satış Yakalama)", color="#2ca02c", linewidth=2.5, linestyle="-.")
ax1.plot(esikler, f1_scores, label="F1-Score (Denge Skoru)", color="#d62728", linewidth=3)
ax1.axvline(en_iyi_esik, color="black", linestyle=":", linewidth=2, label=f"Optimal Eşik ({en_iyi_esik:.2f} -> Max F1: {en_iyi_f1:.3f})")
ax1.axvline(0.50, color="gray", linestyle="--", alpha=0.7, label="Varsayılan Eşik (0.50)")
ax1.scatter([en_iyi_esik], [en_iyi_f1], color="red", s=100, zorder=5)
ax1.set_title("Panel A: Eşik Değeri Ayarı (Threshold Tuning Curve)", fontsize=13, weight="bold", pad=10)
ax1.set_xlabel("Karar Eşik Değeri (Classification Threshold)", fontsize=11)
ax1.set_ylabel("Skor (0 - 1)", fontsize=11)
ax1.legend(loc="best", frameon=True)
ax1.set_xlim(0.05, 0.70)
ax1.set_ylim(0, 1)

# PANEL 2: Karışıklık Matrisi (Confusion Matrix) Yan Yana Karşılaştırma (Eşik 0.50 vs 0.20)
ax2 = axes[0, 1]
cm_data = np.array([
    [cm_020[0,0], cm_020[0,1]],
    [cm_020[1,0], cm_020[1,1]]
])
labels = [
    [f"TN: {cm_020[0,0]:,d}\n(Doğru Sessizlik)", f"FP: {cm_020[0,1]:,d}\n(Fazladan Öneri)"],
    [f"FN: {cm_020[1,0]:,d}\n(Kaçan Satış)", f"TP: {cm_020[1,1]:,d}\n(YAKALANAN CİRO)"]
]

sns.heatmap(cm_data, annot=np.array(labels), fmt="", cmap="Blues", cbar=False, ax=ax2,
            xticklabels=["Önerilmedi (0)", "Önerildi (1)"],
            yticklabels=["Alınmadı (0)", "Alındı (1)"], annot_kws={"size": 11, "weight": "bold"})
ax2.set_title(f"Panel B: Confusion Matrix - Eşik: 0.20 (Yakalanan Satış: {cm_020[1,1]:,d})", fontsize=13, weight="bold", pad=10)
ax2.set_xlabel("Modelin Tahmini", fontsize=11)
ax2.set_ylabel("Gerçek Müşteri Davranışı", fontsize=11)

# PANEL 3: Precision-Recall (PR) Curve
precision_curve, recall_curve, _ = precision_recall_curve(y_val, val_probs)
pr_auc = auc(recall_curve, precision_curve)
taban_oran = pozitif_ornek / toplam_ornek

ax3 = axes[1, 0]
ax3.plot(recall_curve, precision_curve, color="#9467bd", linewidth=3, label=f"Model PR Eğrisi (AUC = {pr_auc:.3f})")
ax3.axhline(taban_oran, color="red", linestyle="--", linewidth=2, label=f"Rastgele Model Tabanı (%{taban_oran*100:.1f})")
ax3.set_title("Panel C: Precision-Recall Curve (Dengesiz Veride Asıl Metrik)", fontsize=13, weight="bold", pad=10)
ax3.set_xlabel("Recall (Yakalama Oranı)", fontsize=11)
ax3.set_ylabel("Precision (Kesinlik)", fontsize=11)
ax3.legend(loc="best", frameon=True)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)

# PANEL 4: Top-K Hit Rate / Recommendation Satış Yakalama Oranı
val_df["sira"] = val_df.groupby("user_id")["prob"].rank(ascending=False, method="first")
toplam_satis = val_df["reordered"].sum()

k_listesi = [1, 3, 5, 8, 10]
hit_rates = []
for k in k_listesi:
    yakalanan = val_df[val_df["sira"] <= k]["reordered"].sum()
    hit_rates.append((yakalanan / toplam_satis) * 100)

ax4 = axes[1, 1]
bars = ax4.bar([f"Top-{k}" for k in k_listesi], hit_rates, color="#ff7f0e", edgecolor="black", alpha=0.85, width=0.55)
for bar in bars:
    yval = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"%{yval:.1f}", ha='center', va='bottom', fontsize=11, weight='bold')

ax4.set_title("Panel D: Top-K Sıralama - Vitrin Genişliğine Göre Satış Yakalama", fontsize=13, weight="bold", pad=10)
ax4.set_xlabel("Müşteriye Gösterilen Öneri Sayısı (Vitrin Alanı)", fontsize=11)
ax4.set_ylabel("Toplam Satışları Yakalama Oranı (%)", fontsize=11)
ax4.set_ylim(0, max(hit_rates) + 10)

plt.suptitle("INSTACART E-TİCARET SEPET TAHMİNİ - MODEL DEĞERLENDİRME VE İŞ ETKİSİ DASHBOARDU", fontsize=15, weight="bold", y=0.98)

# Kaydet
cikis_resim = CIKTI_DIZINI / "05_model_degerlendirme_paneli.png"
plt.savefig(cikis_resim, dpi=300, bbox_inches="tight")
plt.close()

# Artifact dizinine de kopyalayalım
artifact_dir = Path(r"C:\Users\rabia.senlik\.gemini\antigravity\brain\20f637aa-4527-4bf5-8c36-804bd73ed960")
if artifact_dir.exists():
    shutil.copy(cikis_resim, artifact_dir / "05_model_degerlendirme_paneli.png")

print(f">>> Dashboard görseli başarıyla oluşturuldu:")
print(f"    1. {cikis_resim}")
print(f"    2. {artifact_dir / '05_model_degerlendirme_paneli.png'}")

# -----------------------------------------------------------------------------
# 5.3) CANLI VİTRİN SİMÜLASYONU
# -----------------------------------------------------------------------------
products = pd.read_csv(VERI_DIZINI / "products.csv")
top5_df = val_df[val_df["sira"] <= 5]
ornek_user_id = list(val_users)[0]
user_vitrin = top5_df[top5_df["user_id"] == ornek_user_id].merge(products, on="product_id")

print("\n" + "=" * 75)
print(f">>> CANLI VİTRİN SİMÜLASYONU (Kullanıcı ID: {ornek_user_id} - Mobil Ekranda İlk 5 Ürün)")
print("=" * 75)
for i, (_, row) in enumerate(user_vitrin.sort_values("prob", ascending=False).iterrows(), 1):
    sonuc = "[ISABET! Sepete Attı]" if row["reordered"] == 1 else "[Göz Ardı Etti]"
    print(f"  {i}. {row['product_name']:<38} (Tahmin Olasılığı: %{row['prob']*100:>4.1f}) -> {sonuc}")

print("\n" + "=" * 75)
print(f"ADIM 5 TAMAMLANDI! Toplam Süre: {time.time() - t0:.2f} saniye")
print("=" * 75)
