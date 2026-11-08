"""
ADIM 2: KEŞİFÇİ VERİ ANALİZİ (EDA) - INSTACART VERİ SETİ
=========================================================
Bu betik, veri bilimi yaşam döngüsünün 2. adımını temsil eder.
4 standart EDA protokolünü satır satır inceler:
  2.1 Boyut ve Tipler (Shape & Info)
  2.2 Veri Hijyeni (Eksik Değerler / NaN ve Aykırı Değerler / Outliers)
  2.3 Hedef Değişken Analizi (reordered: 1 vs 0)
  2.4 Değişkenler Arası İlişkiler (Zaman ve Ürün Davranışları)

Çalıştırma: python 01_eda_kesif.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Veri klasörümüzün yolu
VERI_DIZINI = Path(__file__).resolve().parent / "data"

print("=" * 70)
print("INSTACART VERİ SETİ KEŞİFÇİ VERİ ANALİZİ (EDA) BAŞLIYOR")
print("=" * 70)

# -----------------------------------------------------------------------------
# 2.1) BOYUT VE TİPLER (SHAPE & DATA TYPES)
# -----------------------------------------------------------------------------
print("\n>>> 2.1) BOYUT VE VERİ TİPLERİNİN İNCELENMESİ")

# Tabloları belleğe alıyoruz
orders = pd.read_csv(VERI_DIZINI / "orders.csv")
products = pd.read_csv(VERI_DIZINI / "products.csv")
departments = pd.read_csv(VERI_DIZINI / "departments.csv")
aisles = pd.read_csv(VERI_DIZINI / "aisles.csv")
order_train = pd.read_csv(VERI_DIZINI / "order_products__train.csv")

tablolar = {
    "orders.csv (Siparişler)": orders,
    "products.csv (Ürün Kataloğu)": products,
    "departments.csv (Departmanlar)": departments,
    "aisles.csv (Reyonlar)": aisles,
    "order_products__train.csv (Eğitim Sepetleri)": order_train
}

for isim, df in tablolar.items():
    print(f"  • {isim:<45}: {df.shape[0]:>10,d} satır | {df.shape[1]:>2d} sütun")

toplam_kullanici = orders["user_id"].nunique()
print(f"\n  [ÖZET] Toplam Tekil Kullanıcı Sayısı: {toplam_kullanici:,d}")
print(f"  [ÖZET] Toplam Tekil Ürün Sayısı      : {products['product_id'].nunique():,d}")

# -----------------------------------------------------------------------------
# 2.2) VERİ HİJYENİ: EKSİK DEĞERLER (NaN) VE AYKIRI DEĞERLER (OUTLIERS)
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> 2.2) VERİ HİJYENİ: EKSİK DEĞER (NaN) VE AYKIRI DEĞER ANALİZİ")

# Hangi tabloda kaç eksik veri var?
print("\n[1] orders.csv tablosundaki eksik değerler:")
eksikler = orders.isnull().sum()
for sutun, adet in eksikler.items():
    oran = (adet / len(orders)) * 100
    print(f"  • {sutun:<25}: {adet:>10,d} eksik (%{oran:.2f})")

# days_since_prior_order sütunundaki NaN gizemini çözüyoruz:
ilk_siparisler = orders[orders["order_number"] == 1]
nan_sayisi = orders["days_since_prior_order"].isnull().sum()

print("\n[2] days_since_prior_order Sütunundaki NaN Analizi:")
print(f"  • Toplam NaN sayısı                     : {nan_sayisi:,d}")
print(f"  • order_number == 1 olan sipariş sayısı : {len(ilk_siparisler):,d}")
print(f"  • Toplam tekil müşteri sayısı          : {toplam_kullanici:,d}")

if nan_sayisi == len(ilk_siparisler) == toplam_kullanici:
    print("  --> SONUÇ: NaN değerler BİR HATA DEĞİLDİR! Her müşterinin ilk siparişinde")
    print("      daha önce bir sipariş olmadığı için bu değer mantıksal olarak boştur.")
    print("      ASLA SİLİNMEYECEK, 0 ile doldurulacak veya 'is_first_order' bayrağı açılacaktır.")

# Aykırı Değerler (Outliers)
print("\n[3] days_since_prior_order Aykırı Değer (Outlier) Dağılımı (Top 5):")
top_gunler = orders["days_since_prior_order"].value_counts().head(5)
for gun, adet in top_gunler.items():
    print(f"  • {int(gun):>2d} gün sonra tekrar sipariş : {adet:>10,d} sipariş")

print("  --> YORUM: 30. gün en yüksek zirvedir çünkü sistem 30 günden fazlasını 30'a kırpmıştır.")
print("  --> YORUM: 7. gün ikinci zirvedir; müşteriler haftalık market döngüsüne sahiptir.")

# -----------------------------------------------------------------------------
# 2.3) HEDEF DEĞİŞKENİN DAĞILIMI (TARGET: reordered)
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> 2.3) HEDEF DEĞİŞKEN (TARGET: reordered) ANALİZİ")

reorder_sayilari = order_train["reordered"].value_counts()
reorder_orani = order_train["reordered"].mean() * 100

print(f"  • reordered = 1 (Tekrar Sipariş): {reorder_sayilari[1]:>10,d} (%{reorder_orani:.2f})")
print(f"  • reordered = 0 (İlk Kez Alınan): {reorder_sayilari[0]:>10,d} (%{100 - reorder_orani:.2f})")
print("  --> YORUM: Sınıflar oldukça dengelidir (~%60 tekrar, %40 yeni).")
print("      Model tek bir sınıfa ezber yapmadan sağlıklı şekilde öğrenebilir.")

# -----------------------------------------------------------------------------
# 2.4) DEĞİŞKENLER ARASI İLİŞKİLER VE DAVRANIŞSAL ALIŞKANLIKLAR
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> 2.4) TÜKETİCİ DAVRANIŞLARI VE ALIŞVERİŞ ALIŞKANLIKLARI")

# 1. En yoğun alışveriş günleri (0 = Pazar, 1 = Pazartesi)
gun_adlari = {0: "Pazar (0)", 1: "Pazartesi (1)", 2: "Salı (2)", 3: "Çarşamba (3)", 
              4: "Perşembe (4)", 5: "Cuma (5)", 6: "Cumartesi (6)"}
gun_dagilimi = orders["order_dow"].value_counts().sort_index()

print("\n[1] Haftanın Günlerine Göre Sipariş Sayıları:")
for gun_no, adet in gun_dagilimi.items():
    print(f"  • {gun_adlari[gun_no]:<15}: {adet:>10,d} sipariş")

# 2. En popüler 5 ürün
populer_urunler = (
    order_train["product_id"].value_counts().head(5).reset_index()
)
populer_urunler.columns = ["product_id", "siparis_sayisi"]
populer_urunler = populer_urunler.merge(products, on="product_id")

print("\n[2] En Çok Sipariş Edilen İlk 5 Ürün (Muz Çılgınlığı):")
for _, satir in populer_urunler.iterrows():
    print(f"  • {satir['product_name']:<30}: {satir['siparis_sayisi']:>8,d} kez sepete girdi")

print("\n" + "=" * 70)
print("EDA TAMAMLANDI! Tüm bulgular sonraki Özellik Mühendisliği adımına hazırdır.")
print("=" * 70)
