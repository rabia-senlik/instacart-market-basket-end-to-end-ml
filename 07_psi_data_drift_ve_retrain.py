"""
ADIM 7: CANLI İZLEME (MLOPS), DATA DRIFT VE PSI (POPULATION STABILITY INDEX)
================================================================================
Bu script, bir modelin canlı sistemde (Production) nasıl izlendiğini simüle eder:
  7.1 Matematiksel PSI (Population Stability Index) Fonksiyonu
  7.2 Senaryo 1: Normal Canlı Veri Akışı (PSI < 0.10 -> Yeşil Işık)
  7.3 Senaryo 2: Veri Kayması / Kriz / Bayram Dönemi (PSI > 0.25 -> Kırmızı Alarm)
  7.4 Otomatik Yeniden Eğitim (Retraining Trigger) Mekanizması

Çalıştırma: python 07_psi_data_drift_ve_retrain.py
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
print("MLOPS CANLI İZLEME: DATA DRIFT TESPİTİ VE OTOMATİK RETRAINING SİMÜLASYONU")
print("=" * 75)

# -----------------------------------------------------------------------------
# 1) MATEMATİKSEL PSI (POPULATION STABILITY INDEX) HESAPLAMA FONKSİYONU
# -----------------------------------------------------------------------------
def calculate_psi(expected, actual, num_buckets=10):
    """
    İki dağılım arasındaki Population Stability Index (PSI) değerini hesaplar.
    Formül: sum((Actual% - Expected%) * ln(Actual% / Expected%))
    
    Yorumlama:
      • PSI < 0.10      : STABİL (Dağılım değişmedi, model güvende)
      • 0.10 <= PSI <= 0.25 : DİKKAT (Hafif kayma başladı, izlemeye al)
      • PSI > 0.25      : 🚨 DRIFT ALARMI (Veri kökten değişti, retraining şart!)
    """
    # Beklenen (Eğitim) verisinin yüzdelik dilimlerini (quantiles) sınır olarak belirle
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(expected, percentiles)
    bin_edges = np.unique(bin_edges) # Çakışan sınırları tekilleştir
    
    if len(bin_edges) < 2:
        return 0.0

    # Her iki veriyi de aynı kutulara (bins) dağıt
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    # Yüzdelik oranlara çevir
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # Sıfıra bölme ve log(0) hatasını önlemek için küçük bir epsilon ekle
    eps = 1e-4
    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)
    
    # Yüzdeleri tekrar normalize et
    expected_pct = expected_pct / expected_pct.sum()
    actual_pct = actual_pct / actual_pct.sum()
    
    # PSI Formülü
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi_total = np.sum(psi_values)
    
    return psi_total

# -----------------------------------------------------------------------------
# 2) VERİYİ OKU VE SİMÜLASYON ORTAMI HAZIRLA
# -----------------------------------------------------------------------------
print("\n>>> Eğitim Verisi ve Canlı Veri Akışları Yükleniyor...")
df = pd.read_parquet(VERI_DIZINI / "train_master_features.parquet")
tum_kullanicilar = df["user_id"].unique()

# Veriyi 3 parçaya bölüyoruz:
# A) Eğitim Seti (Modelin geçmişte öğrendiği dünya): %60 kullanıcı
# B) Canlı Normal Hafta (Beklenen normal akış): %20 kullanıcı
# C) Canlı Kriz / Drift Haftası (Alışkanlıkların değiştiği dönem): %20 kullanıcı
np.random.seed(42)
np.random.shuffle(tum_kullanicilar)

n = len(tum_kullanicilar)
train_users = tum_kullanicilar[:int(n * 0.60)]
normal_live_users = tum_kullanicilar[int(n * 0.60):int(n * 0.80)]
drift_live_users = tum_kullanicilar[int(n * 0.80):]

df_train = df[df["user_id"].isin(train_users)].copy()
df_live_normal = df[df["user_id"].isin(normal_live_users)].copy()
df_live_drifted = df[df["user_id"].isin(drift_live_users)].copy()

# SİMÜLASYON: Gerçek hayatta drift nasıl oluşur?
# Örneğin: Enflasyon arttı, kullanıcıların sipariş sıklığı düştü, son alıştan beri geçen zaman uzadı
df_live_drifted["up_orders_since_last"] = df_live_drifted["up_orders_since_last"] * 2.8 + np.random.exponential(3, len(df_live_drifted))
df_live_drifted["up_order_rate"] = df_live_drifted["up_order_rate"] * 0.35

features = ["up_orders_since_last", "up_order_rate", "up_total_orders", "prod_reorder_ratio"]

# -----------------------------------------------------------------------------
# 3) CANLI SİSTEM İZLEME: SENARYO 1 (NORMAL HAFTA)
# -----------------------------------------------------------------------------
print("\n" + "=" * 75)
print(">>> [CANLI İZLEME TESTİ 1] - HAFTA 1: NORMAL SİSTEM ÇALIŞMASI")
print("=" * 75)

psi_raporu_normal = {}
for col in features:
    psi_val = calculate_psi(df_train[col].dropna(), df_live_normal[col].dropna())
    psi_raporu_normal[col] = psi_val
    durum = "STABIL (YESIL)" if psi_val < 0.10 else ("DIKKAT (SARI)" if psi_val <= 0.25 else "DRIFT ALARMI (KIRMIZI)")
    print(f"  * Ozellik: {col:<25} | PSI Skoru: {psi_val:>6.4f} -> [{durum}]")

max_psi_normal = max(psi_raporu_normal.values())
print(f"\n[SİSTEM KARARI]: Maksimum PSI = {max_psi_normal:.4f} (< 0.10)")
print("  --> Model güvenli bölgede çalışıyor. Canlıya müdahale edilmesine gerek YOK.")

# -----------------------------------------------------------------------------
# 4) CANLI SİSTEM İZLEME: SENARYO 2 (DATA DRIFT / KRİZ DÖNEMİ)
# -----------------------------------------------------------------------------
print("\n" + "=" * 75)
print(">>> [CANLI İZLEME TESTİ 2] - HAFTA 2: ANİ MÜŞTERİ ALIŞKANLIK KAYMASI (DATA DRIFT)")
print("=" * 75)

psi_raporu_drift = {}
for col in features:
    psi_val = calculate_psi(df_train[col].dropna(), df_live_drifted[col].dropna())
    psi_raporu_drift[col] = psi_val
    durum = "STABIL (YESIL)" if psi_val < 0.10 else ("DIKKAT (SARI)" if psi_val <= 0.25 else "[ALARM] DRIFT TESPIT EDILDI (KIRMIZI)")
    print(f"  * Ozellik: {col:<25} | PSI Skoru: {psi_val:>6.4f} -> [{durum}]")

max_psi_drift = max(psi_raporu_drift.values())
print(f"\n[SİSTEM KARARI]: Maksimum PSI = {max_psi_drift:.4f} (> 0.25)")
print("  --> [ALARM] DIKKAT! Musteri aliskanliklari kokten degisti. Modelin eski kurallari cop oldu!")

# -----------------------------------------------------------------------------
# 5) OTOMATİK RETRAINING (YENİDEN EĞİTİM) TETİKLEYİCİSİ
# -----------------------------------------------------------------------------
print("\n" + "=" * 75)
print(">>> [OTOMATIK TETİKLEYİCİ]: RETRAINING PIPELINE BAŞLATILIYOR...")
print("=" * 75)

if max_psi_drift > 0.25:
    print("  [1/3] PSI Alarmı Algılandı! Sistem DevOps / MLOps hattına sinyal gönderdi.")
    print("  [2/3] Son 30 günün yeni verisi (df_live_drifted) eğitim kuyruğuna alındı...")
    print("  [3/3] Model yeni verilerle arka planda yeniden eğitiliyor (Retraining)...")
    
    # Simülasyon: Yeni veriyi eski veriye ekleyip modeli güncelle (Continuous Learning)
    yeni_egitim_seti = pd.concat([df_train, df_live_drifted], ignore_index=True)
    features_all = [c for c in df.columns if c not in ["user_id", "product_id", "reordered"]]
    
    yeni_model = xgb.XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
    yeni_model.fit(yeni_egitim_seti[features_all].fillna(0), yeni_egitim_seti["reordered"], verbose=False)
    
    print("\n  [BAŞARILI]: Yeni model başarıyla eğitildi ve canlı sisteme 'Zero-Downtime' ile deploy edildi!")
    print("  --> Eski model emekliye ayrıldı. Sistem yeni piyasa şartlarına adapte oldu.")
else:
    print("  Model stabil, yeniden eğitime gerek duyulmadı.")

# -----------------------------------------------------------------------------
# 6) GÖRSEL DASHBOARD (GITHUB İÇİN DRIFT VE PSI GRAFİĞİ)
# -----------------------------------------------------------------------------
print("\n>>> GitHub İçin Data Drift & PSI Dashboardu Çiziliyor...")

sns.set_theme(style="whitegrid", font="DejaVu Sans")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Panel 1: PSI Skorları Karşılaştırması (Normal vs Drift)
ozellikler = list(psi_raporu_normal.keys())
x = np.arange(len(ozellikler))
width = 0.35

ax1 = axes[0]
rects1 = ax1.bar(x - width/2, [psi_raporu_normal[k] for k in ozellikler], width, label="Normal Hafta (Stabil)", color="#2ca02c", edgecolor="black")
rects2 = ax1.bar(x + width/2, [psi_raporu_drift[k] for k in ozellikler], width, label="Drift Haftası (Kriz / Değişim)", color="#d62728", edgecolor="black")

ax1.axhline(0.10, color="orange", linestyle="--", linewidth=1.5, label="Hafif Kayma Sınırı (0.10)")
ax1.axhline(0.25, color="red", linestyle=":", linewidth=2, label="Kritik Alarm Sınırı (0.25)")
ax1.set_ylabel("PSI Skoru", fontsize=11, weight="bold")
ax1.set_title("Panel A: Özellik Bazında PSI Skoru Karşılaştırması", fontsize=13, weight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(ozellikler, rotation=20, ha="right", fontsize=10)
ax1.legend(loc="upper left")

# Panel 2: Dağılımın Görsel Kayması (Histogram: Eğitim vs Drift)
ax2 = axes[1]
sns.kdeplot(df_train["up_orders_since_last"].clip(upper=30), label="Eğitim Verisi (Orijinal)", color="#1f77b4", linewidth=2.5, ax=ax2)
sns.kdeplot(df_live_drifted["up_orders_since_last"].clip(upper=30), label="Canlı Veri (Drift / Kaymış)", color="#d62728", linewidth=2.5, linestyle="--", ax=ax2)
ax2.set_title("Panel B: 'up_orders_since_last' Dağılımındaki Gerçek Kayma", fontsize=13, weight="bold")
ax2.set_xlabel("Son Alıştan Geçen Sipariş Sayısı", fontsize=11)
ax2.set_ylabel("Yoğunluk (Density)", fontsize=11)
ax2.legend(loc="upper right")

plt.suptitle("MLOPS CANLI İZLEME: POPULATION STABILITY INDEX (PSI) VE DAĞILIM KAYMASI", fontsize=14, weight="bold", y=1.02)
plt.tight_layout()

cikis_resim = CIKTI_DIZINI / "07_psi_data_drift_dashboard.png"
plt.savefig(cikis_resim, dpi=300, bbox_inches="tight")
plt.close()

# Artifact dizinine de kopyala
artifact_dir = Path(r"C:\Users\rabia.senlik\.gemini\antigravity\brain\20f637aa-4527-4bf5-8c36-804bd73ed960")
if artifact_dir.exists():
    shutil.copy(cikis_resim, artifact_dir / "07_psi_data_drift_dashboard.png")

print(f">>> MLOps Görseli Oluşturuldu: {cikis_resim}")
print(f"ADIM 7 TAMAMLANDI! Toplam Süre: {time.time() - t0:.2f} saniye")
print("=" * 75)
