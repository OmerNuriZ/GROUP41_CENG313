# GROUP41 - CENG313 Introduction to Data Science

**Grup ID:** 41

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
1. EDA: veri kalitesi, aykırı değerler, hedef dağılımları, metadata, görüntü, mevsim, örtü oranı ve sol/sağ analizi
2. Ön işleme: veriyi görüntü başına tek satıra çevirme, görüntü boyutlandırma, StandardScaler
3. Baseline modeller: ortalama, renk özellikleri + Ridge, pretrained CNN özellikleri + Ridge
4. Değerlendirme: 5-fold GroupKFold CV, ağırlıklı R²
5. Assignment 2: kendi çözümümüz

## Sonuçlar
5-fold GroupKFold (grup = State + Sampling_Date), out-of-fold tahminler üzerinde global ağırlıklı R².
Ayrıntılar notebook'un 10-15. bölümlerinde, özet Bölüm 16'da.

| Model | Özellikler | Hedef ölçeği | alpha | CV weighted R² | Fold std |
|-------|-----------|--------------|-------|----------------|----------|
| B0 | Her hedef için eğitim ortalaması | - | - | 0.230 | 0.055 |
| B2 | Renk/örtü özellikleri (15) + Ridge | log1p | 100 | 0.336 | 0.108 |
| B2 | Renk/örtü özellikleri (15) + Ridge | gram | 100 | 0.381 | 0.050 |
| B3 | ResNet50 özellikleri (2048) + Ridge | log1p | 1000 | 0.519 | 0.041 |
| **B3** | **ResNet50 özellikleri (2048) + Ridge** | **gram** | **1000** | **0.531** | **0.084** |

B3'ün hedef bazında R²'si: Green 0.366, Dead 0.235, Clover 0.325, GDM 0.355, Total 0.402.

Ağırlıklı R² hedef bazındaki R²'lerden yüksek, çünkü metrik 5 hedefi tek havuzda, tek bir genel ortalamaya göre
değerlendiriyor. Hedef ortalamaları çok farklı olduğu için (Total ~45 g, Clover ~7 g) hiçbir şey öğrenmeyen B0 bile
0.230 alıyor; hedef bazında R²'leri ise ~0. Modellerin gerçek referans noktası 0 değil, bu 0.230.

### Deneyler
| Deney | Sonuç |
|-------|-------|
| Random KFold vs GroupKFold (B2, log1p) | Random KFold 0.382 ± 0.007 (5 seed), GroupKFold 0.336. Random KFold skoru +0.046 şişiriyor. B0'da fark yok (0.230 / 0.231), yani fark modelin aynı yer/tarihi tanımasından geliyor. |
| 3 hedef tahmin et, GDM ve Total'i topla | log1p ölçeğinde çok daha kötü (B2: 0.196 / 0.336, B3: 0.365 / 0.519). `expm1` ile geri dönen tahminler ortalamanın altında kalıyor ve toplamda birikiyor. Gram ölçeğinde direkt modelle eşit (B2: 0.380 / 0.381, B3: 0.528 / 0.531). |
| log1p vs gram ölçeği | Gram ölçeği iki modelde de daha iyi. Metrik gram cinsinden kare hata ölçtüğü için yüksek biomass'lı örnekleri iyi tahmin etmek önemli. |

### Hata analizi (B3)
- En kötü 6 tahminin hepsi NSW'den ve hepsi büyük düşük tahmin (gerçek Total 111-186 g, tahmin 17-79 g). Bu 6 görüntü hatanın %26'sı.
- Model tahminleri ortalamaya sıkıştırıyor. Total'i en yüksek %25 olan görüntüler hatanın %63'ü.
- Üstten çekilen fotoğrafta hacim görünmüyor: yoğun çayırlar aynı görünüyor, uzun ama seyrek bitkiler (Phalaris, Lucerne) az tahmin ediliyor.

## Submission Geçmişi
Yarışma kapalı olduğu için late submission yapıldı. Skor ekran görüntüsü `submissions/` klasöründe.

| # | Tarih | Notebook | Model | CV weighted R² | Public LB | Private LB |
|---|-------|----------|-------|----------------|-----------|------------|
| 1 | 11.11.2026 | `kaggle_submission.ipynb` (Kaggle V2) | B3: ResNet50 + Ridge, gram, alpha=1000 | 0.531 | 0.43504 | 0.35464 |

**CV ile LB farkı:** CV 0.531, Public LB 0.435, Private LB 0.355. Private skor gizli testin büyük kısmında ölçüldüğü için daha güvenilir. Farkı anlamak için B3'ü farklı CV düzenleriyle tekrar değerlendirdik:

| CV düzeni | B3 | B0 |
|-----------|-----|-----|
| State + tarih GroupKFold (kullandığımız) | 0.531 | 0.230 |
| State bazlı (her fold'da 1 State hiç görülmemiş) | 0.373 | 0.089 |
| Ay bazlı GroupKFold | 0.536 | 0.212 |

Model hiç görmediği bir State'te test edilince skor Private LB'ye çok yakın çıkıyor (0.373 / 0.355). Ay bazlı bölmede düşüş yok.
Yani fark zamandan değil konumdan geliyor: gizli test büyük ihtimalle eğitimdekinden farklı yerlerden ve bizim CV'miz aynı
State'in başka tarihlerini eğitimde bıraktığı için iyimser. Assignment 2'de model seçimini State bazlı CV ile yapacağız.
Kod: notebook Bölüm 15.

## Veri Nasıl İndirilir
Veri repo'ya eklenmiyor. Kaggle kuralları ve dosya boyutu nedeniyle `data/` klasörü `.gitignore`'da.
1. Kaggle'da yarışma sayfasına gir ve kuralları kabul et
2. Kaggle → Settings → API bölümünden token oluştur
3. Notebook'un ilk hücreleri veriyi Kaggle API ile `data/` klasörüne otomatik indiriyor:
   `kaggle competitions download -c csiro-biomass -p data`

## Nasıl Çalıştırılır
1. Kaggle'da yarışma kurallarını kabul et ve Settings → API bölümünden token oluştur
2. `assignment1_eda_baseline.ipynb` dosyasını Google Colab'da aç
3. Colab Secrets'a `KAGGLE_API_TOKEN` adıyla token'ı ekle
4. Çalışma zamanı → Çalışma zamanı türünü değiştir → GPU (B3 için)
5. Çalışma zamanı → Tümünü çalıştır (veri yaklaşık 1 GB, otomatik indiriliyor)

## Dosya Yapısı
- `assignment1_eda_baseline.ipynb`: problem tanımı (giriş), veri genel bakışı (bölüm 1), EDA (bölüm 2-9), ön işleme (bölüm 9.1), baseline modeller, deneyler ve hata analizi (bölüm 10-14), Kaggle submission ve CV ile LB karşılaştırması (bölüm 15), sonuç (bölüm 16)
- `kaggle_submission.ipynb`: Kaggle'da çalışan submission notebook'u (B3). Kaggle'da internet kapalı olduğu için repo klonlanamıyor; bu yüzden `src/models.py`'deki özellik fonksiyonlarının birebir kopyasını içeriyor
- `folds.csv`: 5-fold GroupKFold atamaları (`image_path`, `fold`)
- `src/data.py`: veri okuma, GroupKFold (State + tarih), Random KFold, State bazlı ve ay bazlı fold'lar
- `src/models.py`: B0, B2, B3 modelleri ve çapraz doğrulama
- `src/utils.py`: global ağırlıklı R² metriği
- `src/__init__.py`: boş dosya, `src` klasörünü paket yapar
- `requirements.txt`: kullanılan Python paketleri
- `submissions/`: Kaggle skor ekran görüntüleri

## İlerleme
- 08.11.2026: Repo kuruldu, veri okuma ve pivot fonksiyonu eklendi
- 08.11.2026: EDA tamamlandı: hedef dağılımları, metadata, görüntü, mevsim, örtü oranı, sol/sağ analizi
- 10.11.2026: Ağırlıklı R² metriği ve 5-fold GroupKFold (State + tarih) eklendi, `folds.csv` oluşturuldu
- 10.11.2026: Baseline modeller: B0 ortalama (0.230), B2 renk özellikleri + Ridge (0.336)
- 10.11.2026: Deneyler: Random KFold skoru +0.046 şişiriyor; 3 hedef + toplam log1p'de çöküyor, gram ölçeği daha iyi
- 10.11.2026: B3 pretrained ResNet50 özellikleri + Ridge (0.531, en iyi model)
- 11.11.2026: Hata analizi: model yüksek biomass'ı az tahmin ediyor, en kötü tahminler NSW'de
- 11.11.2026: Kaggle submission notebook'u ve ilk submit (B3, Public LB 0.43504, Private LB 0.35464)
- 11.11.2026: README'ye sonuçlar ve submission geçmişi eklendi; CV-LB farkının konumdan geldiği bulundu
- 13.11.2026: Notebook Bölüm 15: State bazlı ve ay bazlı CV ile LB karşılaştırması (State bazlı B3 0.373, Private LB 0.355)
- 13.11.2026: Notebook'a Sonuç bölümü (16) eklendi; log1p kararının neden revize edildiği açıklandı
- 13.11.2026: Notebook bölüm yapısı ve README düzenlendi