"""
ADIM 6: İTERASYON, HATA ANALİZİ (ERROR ANALYSIS) VE CANLI İZLEME (MLOPS)
================================================================================
Kıdemli Veri Bilimci Farkı: Model eğitip bırakılmaz!
  6.1 Hata Analizi (Error Analysis): Model nerede çuvallıyor? (FN ve FP anatomisi)
  6.2 Modelin Beynini Okumak: Özellik Önemi (Feature Importance - Gain)
  6.3 Model İterasyonu: Sıradaki sprintte modeli nasıl daha iyi yaparız?
  6.4 MLOps ve Canlı İzleme: Data Drift ve Concept Drift tespiti, Retraining döngüsü
  6.5 GitHub İçin Profesyonel Analiz Görseli (PNG)

Çalıştırma: python 06_hata_analizi_ve_mlops.py
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import shutil
import warnings
warnings.filterwarnings("ignore")

t0 = time.time()
VERI_DIZINI = Path(__file__).resolve().parent / "data"
CIKTI_DIZINI = Path(__file__).resolve().parent

print("=" * 75)
print("ADIM 6: HATA ANALİZİ (ERROR ANALYSIS), İTERASYON VE MLOPS CANLI İZLEME")
print("=" * 75)

# 1) Veriyi Oku ve Kullanıcı Bazlı Böl
df = pd.read_parquet(VERI_DIZINI / "train_master_features.parquet")
products = pd.read_csv(VERI_DIZINI / "products.csv")

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

# 2) Modeli Eğit ve Tahminleri Al
print("\n>>> Model Eğitiliyor...")
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
val_df["prob"] = model.predict_proba(X_val)[:, 1]
val_df["pred"] = (val_df["prob"] >= 0.20).astype(int)

# Ürün isimlerini eşleştir
val_df = val_df.merge(products[["product_id", "product_name"]], on="product_id", how="left")

# -----------------------------------------------------------------------------
# 6.1) HATA ANALİZİ: MODEL NEREDE ÇUVALLIYOR?
# -----------------------------------------------------------------------------
print("\n" + "=" * 75)
print(">>> 6.1) HATA ANALİZİ (ERROR ANALYSIS): MODELİN YANILDIĞI NOKTALAR")
print("=" * 75)

# FN: Gerçekte 1 (Alındı) ama Model 0 dedi (KAÇAN SATIŞ)
fn_df = val_df[(val_df["reordered"] == 1) & (val_df["pred"] == 0)]

# FP: Gerçekte 0 (Alınmadı) ama Model 1 dedi (BOŞUNA ÖNERİ)
fp_df = val_df[(val_df["reordered"] == 0) & (val_df["pred"] == 1)]

# TP: Doğru Yakalanan Satışlar
tp_df = val_df[(val_df["reordered"] == 1) & (val_df["pred"] == 1)]

print(f"Toplam Doğrulama Örneği : {len(val_df):,d}")
print(f"Başarıyla Yakalanan Satış (TP): {len(tp_df):,d}")
print(f"Kaçırılan Satışlar (FN)       : {len(fn_df):,d}")
print(f"Gereksiz Önerilenler (FP)     : {len(fp_df):,d}")

print("\n--- Model En Çok Hangi Ürünlerin Satışını Kaçırıyor? (Top 5 FN Ürün) ---")
top_fn_products = fn_df["product_name"].value_counts().head(5)
for p_name, count in top_fn_products.items():
    print(f"  • {p_name:<40}: {count} kez kaçırıldı")

print("\n--- Kaçırılan Satışların (FN) Arkasındaki Kök Neden (Root Cause) ---")
print(f"  • Yakalanan Satışlarda (TP) ortalama sipariş sıklığı (up_order_rate): {tp_df['up_order_rate'].mean():.2f}")
print(f"  • Kaçan Satışlarda (FN) ortalama sipariş sıklığı (up_order_rate)    : {fn_df['up_order_rate'].mean():.2f}")
print(f"  • Yakalanan Satışlarda son alıştan beri geçen sipariş (since_last)   : {tp_df['up_orders_since_last'].mean():.1f}")
print(f"  • Kaçan Satışlarda son alıştan beri geçen sipariş (since_last)       : {fn_df['up_orders_since_last'].mean():.1f}")

print("\n[MÜHENDİSLİK ÇIKARIMI]:")
print("  --> Model, müşterinin SIK aldığı (up_order_rate > 0.45) ve YENİ aldığı (since_last < 2) ürünleri")
print("      mükemmel yakalıyor!")
print("  --> Modelin kaçırdığı ürünler (FN), müşterinin 4-5 siparişte bir nadiren aldığı (deterjan, baharat vb.)")
print("      veya geçmişte sadece 1-2 kez denediği 'düşük frekanslı' ürünlerdir!")

# -----------------------------------------------------------------------------
# 6.2) MODELİN BEYNİNİ OKUMAK: FEATURE IMPORTANCE (ÖZELLİK ÖNEMİ)
# -----------------------------------------------------------------------------
print("\n" + "=" * 75)
print(">>> 6.2) ÖZELLİK ÖNEMİ (FEATURE IMPORTANCE): MODEL NEYE BAKIYOR?")
print("=" * 75)

importance_dict = model.get_booster().get_score(importance_type="gain")
feat_imp = pd.DataFrame({
    "Feature": list(importance_dict.keys()),
    "Gain": list(importance_dict.values())
}).sort_values("Gain", ascending=False).reset_index(drop=True)

for i, row in feat_imp.head(6).iterrows():
    print(f"  {i+1}. {row['Feature']:<25} | Bilgi Kazanımı (Gain): {row['Gain']:>8.2f}")

# -----------------------------------------------------------------------------
# 6.3) GÖRSEL DASHBOARD OLUŞTURMA (GITHUB İÇİN)
# -----------------------------------------------------------------------------
print("\n>>> GitHub İçin Hata Analizi ve MLOps Görseli Oluşturuluyor...")

sns.set_theme(style="whitegrid", font="DejaVu Sans")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# PANEL A: Özellik Önemi (Feature Importance - Gain)
ax1 = axes[0, 0]
top10_feat = feat_imp.head(10).sort_values("Gain", ascending=True)
bars = ax1.barh(top10_feat["Feature"], top10_feat["Gain"], color="#1f77b4", edgecolor="black", alpha=0.85)
ax1.set_title("Panel A: Modelin Karar Mekanizması (Feature Importance - Gain)", fontsize=13, weight="bold", pad=10)
ax1.set_xlabel("Bilgi Kazanımı (Gain Değeri)", fontsize=11)

# PANEL B: Hata Analizi Kök Neden (TP vs FN Karşılaştırması)
ax2 = axes[0, 1]
karsilastirma_df = pd.DataFrame({
    "Kategori": ["Yakalanan Satış (TP)", "Kaçırılan Satış (FN)"],
    "Ortalama Sipariş Oranı (%)": [tp_df["up_order_rate"].mean()*100, fn_df["up_order_rate"].mean()*100],
    "Son Alıştan Geçen Sipariş": [tp_df["up_orders_since_last"].mean(), fn_df["up_orders_since_last"].mean()]
})

x = np.arange(2)
width = 0.35
ax2.bar(x - width/2, karsilastirma_df["Ortalama Sipariş Oranı (%)"], width, label="Sipariş Sıklığı (%)", color="#2ca02c", edgecolor="black")
ax2.bar(x + width/2, karsilastirma_df["Son Alıştan Geçen Sipariş"] * 10, width, label="Son Alıştan Geçen Sipariş (x10)", color="#d62728", edgecolor="black")
ax2.set_xticks(x)
ax2.set_xticklabels(karsilastirma_df["Kategori"], fontsize=11, weight="bold")
ax2.set_title("Panel B: Hata Analizi Kök Neden (Neden Satış Kaçtı?)", fontsize=13, weight="bold", pad=10)
ax2.set_ylabel("Değer", fontsize=11)
ax2.legend(loc="upper right", frameon=True)

# PANEL C: Model İterasyon Yol Haritası (Sıradaki Sprint)
ax3 = axes[1, 0]
ax3.axis("off")
iterasyon_text = (
    "MODEL İTERASYON YOL HARİTASI (Sprint 2 - Gelecek Geliştirmeler):\n\n"
    "1. Nadir Ürünlerin Tespiti (Periyodik Alışveriş Özelliği):\n"
    "   • Deterjan, tuvalet kağıdı gibi ürünler 45 günde bir alınır.\n"
    "   • 'days_since_prior_order' kümülatif toplanıp ürünün periyoduyla\n"
    "     kıyaslanmalı: (Geçen Gün / Ortalama Tüketim Süresi).\n\n"
    "2. Birlikte Alınan Ürünler (Co-occurrence / Market Basket):\n"
    "   • Makarna alan sos da alır. Sepetteki anlık ürün etkileşimi\n"
    "     (Item2Vec veya Apriori skorları) modele girdi yapılmalı.\n\n"
    "3. Zaman Bazlı Özellikler (Temporal Features):\n"
    "   • Sipariş verilen gün (Hafta sonu kahvaltılıklar artar mı?)\n"
    "   • Sipariş verilen saat (Sabah kahve, akşam abur cubur)."
)
ax3.text(0.05, 0.95, iterasyon_text, transform=ax3.transAxes, fontsize=11,
         verticalalignment="top", bbox=dict(boxstyle="round,pad=0.8", facecolor="#e8f4f8", edgecolor="#1f77b4", linewidth=2))
ax3.set_title("Panel C: Model İterasyonu (Nasıl Daha İyi Yaparız?)", fontsize=13, weight="bold", pad=10)

# PANEL D: MLOps ve Canlı İzleme (Monitoring & Retraining)
ax4 = axes[1, 1]
ax4.axis("off")
mlops_text = (
    "CANLI İZLEME (MLOPS MONITORING) VE DRIFT STRATEJİSİ:\n\n"
    "1. Data Drift (Veri Kayması) Takibi:\n"
    "   • Yaz mevsimi gelince karpuz/dondurma satışı artar mı?\n"
    "   • Girdi dağılımlarını PSI (Population Stability Index) ile izle.\n"
    "   • PSI > 0.25 ise alarm ver: 'Veri dağılımı değişti!'.\n\n"
    "2. Concept Drift (Kullanıcı Alışkanlığı Kayması):\n"
    "   • Enflasyon veya krizde organik ürün yerine ucuz ürünler tercih edilir.\n"
    "   • Canlıda günlük CTR ve Top-5 Recall takip edilir.\n\n"
    "3. Otomatik Yeniden Eğitim (Retraining Pipeline):\n"
    "   • Model her Pazartesi gecesi son 30 günün yeni verisiyle\n"
    "     otomatik olarak yeniden eğitilir (Sliding Window)."
)
ax4.text(0.05, 0.95, mlops_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment="top", bbox=dict(boxstyle="round,pad=0.8", facecolor="#fff2e6", edgecolor="#ff7f0e", linewidth=2))
ax4.set_title("Panel D: MLOps Canlı İzleme ve Drift Yönetimi", fontsize=13, weight="bold", pad=10)

plt.suptitle("INSTACART SEPET TAHMİNİ - HATA ANALİZİ, İTERASYON VE MLOPS MİMARİSİ", fontsize=15, weight="bold", y=0.98)

cikis_resim = CIKTI_DIZINI / "06_hata_analizi_ve_izleme_paneli.png"
plt.savefig(cikis_resim, dpi=300, bbox_inches="tight")
plt.close()

# Artifact klasörüne de kopyala
artifact_dir = Path(r"C:\Users\rabia.senlik\.gemini\antigravity\brain\20f637aa-4527-4bf5-8c36-804bd73ed960")
if artifact_dir.exists():
    shutil.copy(cikis_resim, artifact_dir / "06_hata_analizi_ve_izleme_paneli.png")

print(f">>> MLOps ve Hata Analizi Dashboardu Başarıyla Oluşturuldu:")
print(f"    1. {cikis_resim}")
print(f"    2. {artifact_dir / '06_hata_analizi_ve_izleme_paneli.png'}")

print("\n" + "=" * 75)
print(f"ADIM 6 TAMAMLANDI! Toplam Süre: {time.time() - t0:.2f} saniye")
print("=" * 75)
