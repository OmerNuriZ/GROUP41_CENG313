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
- Hedefler sağa çarpık. Başta log1p ölçeğinde eğitmeyi planladık, ama deneylerde gram ölçeği daha iyi çıktı (bkz. Sonuçlar)
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

## Sonuçlar
5-fold GroupKFold (grup = State + Sampling_Date), out-of-fold tahminler üzerinde global ağırlıklı R².
Ayrıntılar notebook'un 10-14. bölümlerinde.

| Model | Özellikler | Hedef ölçeği | alpha | CV weighted R² | Fold std |
|-------|-----------|--------------|-------|----------------|----------|
| B0 | Her hedef için eğitim ortalaması | - | - | 0.230 | 0.055 |
| B2 | Renk/örtü özellikleri (15) + Ridge | log1p | 100 | 0.336 | 0.108 |
| B2 | Renk/örtü özellikleri (15) + Ridge | gram | 100 | 0.381 | 0.050 |
| B3 | ResNet50 özellikleri (2048) + Ridge | log1p | 1000 | 0.519 | 0.041 |
| **B3** | **ResNet50 özellikleri (2048) + Ridge** | **gram** | **1000** | **0.531** | **0.084** |

B3'ün hedef bazında R²'si: Green 0.37, Dead 0.24, Clover 0.33, GDM 0.36, Total 0.40.

### Deneyler
| Deney | Sonuç |
|-------|-------|
| Random KFold vs GroupKFold (B2) | Random KFold 0.382 ± 0.007 (5 seed), GroupKFold 0.336. Random KFold skoru +0.046 şişiriyor. B0'da fark yok (0.230 / 0.231), yani fark modelin aynı yer/tarihi tanımasından geliyor. |
| 3 hedef tahmin et, GDM ve Total'i topla | log1p ölçeğinde çok daha kötü (B2: 0.196 / 0.336, B3: 0.365 / 0.519). `expm1` ile geri dönen tahminler ortalamanın altında kalıyor ve toplamda birikiyor. Gram ölçeğinde direkt modelle eşit (B2: 0.380 / 0.381, B3: 0.528 / 0.531). |
| log1p vs gram ölçeği | Gram ölçeği iki modelde de daha iyi. Metrik gram cinsinden kare hata ölçtüğü için yüksek biomass'lı örnekleri iyi tahmin etmek önemli. |

### Hata analizi (B3)
- En kötü 6 tahminin hepsi NSW'den ve hepsi büyük düşük tahmin (gerçek Total 111-186 g, tahmin 17-79 g). Bu 6 görüntü hatanın %26'sı.
- Model tahminleri ortalamaya sıkıştırıyor. Total'i en yüksek %25 olan görüntüler hatanın %63'ü.
- Üstten çekilen fotoğrafta hacim görünmüyor: yoğun çayırlar aynı görünüyor, uzun ama seyrek bitkiler (Phalaris, Lucerne) az tahmin ediliyor.

## Nasıl Çalıştırılır
1. Kaggle'da yarışma kurallarını kabul et ve Settings → API bölümünden token oluştur
2. `assignment1_eda_baseline.ipynb` dosyasını Google Colab'da aç
3. Colab Secrets'a `KAGGLE_API_TOKEN` adıyla token'ı ekle
4. Çalışma zamanı → Çalışma zamanı türünü değiştir → GPU (B3 için)
5. Çalışma zamanı → Tümünü çalıştır (veri yaklaşık 1 GB, otomatik indiriliyor)

## Dosya Yapısı
- `assignment1_eda_baseline.ipynb`: EDA (bölüm 1-9), baseline modeller, deneyler ve hata analizi (bölüm 10-14)
- `kaggle_submission.ipynb`: Kaggle'da çalışan submission notebook'u (B3)
- `folds.csv`: 5-fold GroupKFold atamaları (`image_path`, `fold`)
- `src/data.py`: veri okuma, GroupKFold ve Random KFold fold'ları
- `src/models.py`: B0, B2, B3 modelleri ve çapraz doğrulama
- `src/utils.py`: global ağırlıklı R² metriği

## İlerleme
- 08.11.2026: Repo kuruldu, veri okuma ve pivot fonksiyonu eklendi
- 08.11.2026: EDA tamamlandı: hedef dağılımları, metadata, görüntü, mevsim, örtü oranı, sol/sağ analizi