"""
ADIM 3: ÖZELLİK MÜHENDİSLİĞİ (FEATURE ENGINEERING)
===================================================
Bu betik, 6 farklı CSV dosyasını birleştirip modelin eğitileceği
TEK BİR NİHAİ TABLO (X_train, y_train) üretir.

DATA LEAKAGE KİLİDİ:
Tüm geçmiş alışveriş istatistikleri SADECE 'prior' verisinden hesaplanır.
Gelecekteki 'train' sepetinin bilgisi ASLA geçmiş özelliklerine karıştırılmaz!

Türetilen Özellik Seviyeleri:
  1. Kullanıcı Özellikleri (User Features)
  2. Ürün Özellikleri (Product Features)
  3. Kullanıcı-Ürün Etkileşim Özellikleri (User-Product Interaction)
  4. Hedef Değişken (reordered: 1 veya 0)

Çalıştırma: python 02_feature_engineering.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time

t0 = time.time()
VERI_DIZINI = Path(__file__).resolve().parent / "data"

print("=" * 70)
print("ÖZELLİK MÜHENDİSLİĞİ (FEATURE ENGINEERING) BAŞLIYOR")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1) KULLANICI ÖRNEKLEMİ SEÇİMİ (Hızlı ve Sağlıklı Hesaplama)
# -----------------------------------------------------------------------------
# 206.000 kullanıcının 32 milyon satırlık verisini lokal bilgisayarda RAM'i
# şişirmeden, profesyonelce işlemek için 5.000 aktif train kullanıcısı seçiyoruz.
print("\n>>> 1) Kullanıcıların Seçilmesi...")
orders = pd.read_csv(VERI_DIZINI / "orders.csv")

ORNEK_KULLANICI_SAYISI = 5000
train_users = orders[orders["eval_set"] == "train"]["user_id"].unique()[:ORNEK_KULLANICI_SAYISI]
train_users_set = set(train_users)

print(f"  • Seçilen aktif kullanıcı sayısı: {len(train_users_set):,d}")

# Bu kullanıcıların siparişlerini filtreleyelim
orders_sample = orders[orders["user_id"].isin(train_users_set)].copy()

# 'prior' (geçmiş) ve 'train' (hedef) siparişlerini ayıralım
prior_orders = orders_sample[orders_sample["eval_set"] == "prior"]
train_target_orders = orders_sample[orders_sample["eval_set"] == "train"]

prior_order_ids = set(prior_orders["order_id"])
print(f"  • Geçmiş (prior) sipariş sayısı : {len(prior_order_ids):,d}")
print(f"  • Hedef (train) sipariş sayısı  : {len(train_target_orders):,d}")

# -----------------------------------------------------------------------------
# 2) GEÇMİŞ SEPETLERİN (PRIOR) OKUNMASI (Data Leakage Kilidi)
# -----------------------------------------------------------------------------
print("\n>>> 2) Geçmiş Sepet Verilerinin (Prior) Belleğe Alınması...")
# 32 milyon satırlık dosyayı parça parça (chunk) okuyup sadece seçtiğimiz
# kullanıcıların geçmiş sepetlerini çekiyoruz.
chunks = []
for chunk in pd.read_csv(VERI_DIZINI / "order_products__prior.csv", chunksize=2_000_000):
    match = chunk[chunk["order_id"].isin(prior_order_ids)]
    if len(match) > 0:
        chunks.append(match)

prior_df = pd.concat(chunks, ignore_index=True)
print(f"  • Geçmiş sepet satır sayısı: {len(prior_df):,d}")

# orders bilgisiyle (user_id, order_number vb.) prior_df'i birleştirelim
prior_df = prior_df.merge(
    prior_orders[["order_id", "user_id", "order_number", "days_since_prior_order"]],
    on="order_id",
    how="left"
)

# -----------------------------------------------------------------------------
# 3) KULLANICI ÖZELLİKLERİ (USER FEATURES)
# -----------------------------------------------------------------------------
print("\n>>> 3) Müşteri Seviyesi Özellikler Hesaplanıyor...")
user_features = prior_df.groupby("user_id").agg(
    user_total_orders=("order_number", "max"),
    user_total_products=("product_id", "count"),
    user_unique_products=("product_id", "nunique"),
    user_reorder_ratio=("reordered", "mean")
).reset_index()

# Ortalama sipariş aralığını orders tablosundan ekleyelim
user_days = prior_orders.groupby("user_id")["days_since_prior_order"].mean().reset_index()
user_days.columns = ["user_id", "user_avg_days_between_orders"]
user_features = user_features.merge(user_days, on="user_id", how="left")

# Ortalama sepet büyüklüğü = toplam alınan ürün / toplam sipariş sayısı
user_features["user_avg_basket_size"] = (
    user_features["user_total_products"] / user_features["user_total_orders"]
)
print("  • Müşteri özellikleri hazır: (user_total_orders, user_avg_days, user_reorder_ratio, ...)")

# -----------------------------------------------------------------------------
# 4) ÜRÜN ÖZELLİKLERİ (PRODUCT FEATURES)
# -----------------------------------------------------------------------------
print("\n>>> 4) Ürün Seviyesi Özellikler Hesaplanıyor...")
prod_features = prior_df.groupby("product_id").agg(
    prod_total_purchases=("order_id", "count"),
    prod_reorder_ratio=("reordered", "mean"),
    prod_avg_add_to_cart_order=("add_to_cart_order", "mean")
).reset_index()

# Ürün kataloğundan reyon (aisle) ve kategori (department) bilgilerini ekleyelim
products = pd.read_csv(VERI_DIZINI / "products.csv")
prod_features = prod_features.merge(
    products[["product_id", "aisle_id", "department_id"]],
    on="product_id",
    how="left"
)
print("  • Ürün özellikleri hazır: (prod_total_purchases, prod_reorder_ratio, aisle_id, ...)")

# -----------------------------------------------------------------------------
# 5) KULLANICI - ÜRÜN ETKİLEŞİMİ (USER-PRODUCT FEATURES - EN GÜÇLÜ SİNYAL!)
# -----------------------------------------------------------------------------
print("\n>>> 5) Müşteri + Ürün Etkileşim Özellikleri Hesaplanıyor...")
user_prod_features = prior_df.groupby(["user_id", "product_id"]).agg(
    up_total_orders=("order_id", "count"),
    up_first_order_number=("order_number", "min"),
    up_last_order_number=("order_number", "max"),
    up_avg_add_to_cart=("add_to_cart_order", "mean")
).reset_index()

# Kullanıcı özellikleri ile birleştirip oranları hesaplayalım:
user_prod_features = user_prod_features.merge(
    user_features[["user_id", "user_total_orders"]], on="user_id", how="left"
)

# up_order_rate: Müşteri kendi siparişlerinin yüzde kaçında bu ürünü aldı?
user_prod_features["up_order_rate"] = (
    user_prod_features["up_total_orders"] / user_prod_features["user_total_orders"]
)

# up_orders_since_last: Müşteri son kaç sipariştir bu ürünü almıyor? (Çok kritik!)
user_prod_features["up_orders_since_last"] = (
    user_prod_features["user_total_orders"] - user_prod_features["up_last_order_number"]
)
print("  • Etkileşim özellikleri hazır: (up_total_orders, up_order_rate, up_orders_since_last, ...)")

# -----------------------------------------------------------------------------
# 6) HEDEF DEĞİŞKENİN (reordered: 1 veya 0) EKLENMESİ VE BİRLEŞTİRME
# -----------------------------------------------------------------------------
print("\n>>> 6) Hedef Değişken (reordered) ile Nihai Tablonun Oluşturulması...")

# Hedef sepetleri okuyalım (order_products__train.csv)
train_products = pd.read_csv(VERI_DIZINI / "order_products__train.csv")
target_order_ids = set(train_target_orders["order_id"])
train_target_products = train_products[train_products["order_id"].isin(target_order_ids)]

# Hedef siparişlerdeki (user_id, product_id) çiftleri:
train_target_products = train_target_products.merge(
    train_target_orders[["order_id", "user_id"]], on="order_id", how="left"
)

# Hedefte olan ürünlere reordered = 1 bayrağı
target_pairs = set(zip(train_target_products["user_id"], train_target_products["product_id"]))

# user_prod_features tablomuzdaki her çift hedef sepette var mı?
user_prod_features["reordered"] = [
    1 if (u, p) in target_pairs else 0
    for u, p in zip(user_prod_features["user_id"], user_prod_features["product_id"])
]

# Şimdi tüm tabloları tek bir çatı altında topluyoruz:
master_df = user_prod_features.merge(user_features, on="user_id", how="left")
master_df = master_df.merge(prod_features, on="product_id", how="left")

# İsim temizliği (çift sütunları ayıkla)
if "user_total_orders_x" in master_df.columns:
    master_df["user_total_orders"] = master_df["user_total_orders_x"]
    master_df.drop(columns=["user_total_orders_x", "user_total_orders_y"], inplace=True)

# Çıktı dosyasını kaydedelim
cikti_yolu = VERI_DIZINI / "train_master_features.parquet"
master_df.to_parquet(cikti_yolu, index=False)

print(f"\n" + "=" * 70)
print(f"TEK NİHAİ TABLO BAŞARIYLA ÜRETİLDİ!")
print(f"  • Toplam Satır Sayısı  : {len(master_df):,d} (Kullanıcı-Ürün çifti)")
print(f"  • Toplam Sütun Sayısı  : {len(master_df.columns):d}")
print(f"  • Pozitif Hedef (1)    : {(master_df['reordered'] == 1).sum():,d} (Sepete tekrar girenler)")
print(f"  • Negatif Hedef (0)    : {(master_df['reordered'] == 0).sum():,d} (Bu sefer alınmayanlar)")
print(f"  • Kaydedilen Dosya     : {cikti_yolu.name}")
print(f"  • Toplam Geçen Süre    : {time.time() - t0:.2f} saniye")
print("=" * 70)

print("\nNİHAİ EĞİTİM TABLOSUNDAN İLK 3 SATIR ÖRNEĞİ:")
print(master_df[["user_id", "product_id", "up_total_orders", "up_order_rate", "up_orders_since_last", "user_total_orders", "prod_reorder_ratio", "reordered"]].head(3))
