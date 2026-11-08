# GROUP41 - CENG313 Introduction to Data Science

## Proje
Kaggle yarışması: [CSIRO - Image2Biomass Prediction](https://www.kaggle.com/competitions/csiro-biomass)

Amaç: Mera fotoğraflarından 5 biomass değerini (gram) tahmin etmek.
- Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g
- Değerlendirme: global ağırlıklı R². Tüm (görüntü, hedef) satırları tek havuzda, gram cinsinden hesaplanıyor. Ağırlıklar: Dry_Total 0.5, GDM 0.2, diğerleri 0.1

## Takım
| İsim | Öğrenci No | GitHub |
|------|-----------|--------|
| Bilgehan Akbaş | 23118080055 | Bilgehankbs |
| Ömer Nuri Zaimoğlu | 23118080015 | OmerNuriZ |

## Veri
- 357 eğitim görüntüsü, hepsi 2000x1000 piksel
- Metadata (State, Species, tarih, NDVI, Height) sadece eğitim setinde var, test setinde yok
- Test seti gizli, submission Kaggle notebook üzerinden yapılıyor

## EDA Bulguları
- Veri temiz: eksik, tekrar eden ya da negatif değer yok
- Species etiketi baskın türü gösteriyor, kesin içeriği değil
- Hedefler sağa çarpık, modeller log1p ölçeğinde eğitilip tahminler grama geri çevrilecek
- Total = Green + Dead + Clover ve GDM = Green + Clover
- Konum ve mevsim birbirine karışmış, bu yüzden GroupKFold kullanılacak
- Yeşil örtü oranı GDM ile ilişkili ama yoğun çayırlarda doyuyor
- Görüntünün sol ve sağ yarısı çok benzer
- En zor hedefler Clover ve Dead

## Yaklaşım
1. EDA: veri kalitesi, hedef dağılımları, metadata, görüntü, mevsim, örtü oranı ve sol/sağ analizi
2. Baseline modeller: ortalama, renk özellikleri + Ridge, pretrained CNN özellikleri + Ridge
3. Değerlendirme: 5-fold GroupKFold CV, ağırlıklı R²
4. Assignment 2: kendi çözümümüz

## Nasıl Çalıştırılır
1. Kaggle'da yarışma kurallarını kabul et ve Settings → API bölümünden token oluştur
2. `assignment1_eda_baseline.ipynb` dosyasını Google Colab'da aç
3. Colab Secrets'a `KAGGLE_API_TOKEN` adıyla token'ı ekle
4. Çalışma zamanı → Tümünü çalıştır (veri yaklaşık 1 GB, otomatik indiriliyor)

## Dosya Yapısı
- `assignment1_eda_baseline.ipynb`: EDA ve baseline modeller
- `src/data.py`: veri okuma
- `src/models.py`: modeller
- `src/utils.py`: yardımcı fonksiyonlar

## İlerleme
- 08.11.2026: Repo kuruldu, veri okuma ve pivot fonksiyonu eklendi
- 08.11.2026: EDA tamamlandı: hedef dağılımları, metadata, görüntü, mevsim, örtü oranı, sol/sağ analizi