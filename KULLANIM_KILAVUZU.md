# Model Simulator Kullanım Kılavuzu

## Ana Menü

### 1. Yeni Tekil Model Simülasyonu
- Tek bir model için eğitim simülasyonu yapar
- Parametreleri manuel olarak girersiniz
- Anlık grafik gösterir
- Sonucu tabloya ekler

### 2. Toplu Model Simülasyonu (Manuel/CSV)
- Birden fazla modeli tek seferde simüle eder  
- **İki seçenek**: Manuel string girişi VEYA CSV dosyasından okuma
- **7 parametre gerekir** (PrecisionRecallDengesi dahil)
- Format: `ModelAdı,Optimizer,ÖğrenmeHızı,OverfitEpochu,OverfitEğimi,FinalDoğruluk,PrecisionRecallDengesi`
- **ÖNEMLİ**: Artık hedef doğruluğa gerçekçi şekilde ulaşır
- **YENİ**: Precision-Recall dengesi kontrolü
- Grafik göstermez (performans için)
- Toplu işlem raporu verir

### 3. Sonuç Tablosunu Görüntüle
- Tüm simülasyon sonuçlarını tablo halinde gösterir
- 20 farklı metrik içerir
- Formatlanmış ve okunaklı çıktı
- Toplam kayıt sayısını gösterir

### 4. Sonuçları CSV'e Kaydet
- Sonuç tablosunu CSV dosyasına kaydeder
- Dosya adını kendiniz belirlersiniz
- Tüm metrikleri içerir
- Excel'de açılabilir format

### 5. Grafikleri Görselleştir
- Temel performans karşılaştırma grafikleri
- Detaylı epoch-epoch analizi seçeneği
- Model seçimi yapabilirsiniz (tekli/çoklu/aralık)
- 4 farklı grafik türü

### 6. Grafikleri Kaydet
- Seçenek 5'i çalıştırır ve grafikleri PNG olarak kaydeder
- Yüksek çözünürlük desteği
- Otomatik dosya adlandırma
- Ayrı dizine kaydeder

### 7. Karşılaştırmalı Matris & Isı Haritaları
- **YENİ**: Gelişmiş matris görselleştirmeleri
- Performans karşılaştırma matrisi
- Korelasyon ısı haritaları
- Confusion matrix simülasyonları
- Metrikler arası ilişki matrisleri
- Model performans dendrogramı
- Tüm matrisleri toplu gösterme seçeneği

### 8. Ayarları Yönet
- Simülasyon parametrelerini değiştir
- Grafik ayarlarını özelleştir
- Config.json dosyasını düzenle
- Varsayılana sıfırlama seçeneği

### 9. Çıkış
- Programdan güvenli çıkış
- Değişiklikleri kaydeder
- Terminal'i temizler

## Hızlı Başlangıç

### 1. Tek Model Simülasyonu
- Menüden `1` seçin
- Parametreleri girin (Enter = varsayılan)
- Grafikleri inceleyin

### 2. Toplu Simülasyon
- Menüden `2` seçin
- **Alt-menü**: 1=Manuel giriş, 2=CSV'den oku
- Format: `ModelAdı,Optimizer,ÖğrenmeHızı,OverfitEpochu,OverfitEğimi,FinalDoğruluk,PrecisionRecallDengesi`
- Örnek: `VGG19,Adam,0.01,60,0.001,0.92,0.2; ResNet50,SGD,0.05,70,0.002,0.89,-0.1`

### 3. Sonuçları İnceleme
- `3` → Tablo görüntüle
- `5` → Grafikler ve detaylı analiz
- `7` → **YENİ**: Karşılaştırmalı matris & ısı haritaları
- `4` → CSV olarak kaydet

## Temel Parametreler

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| Model Adı | VGG19 | Model ismi |
| Optimizer | Adam | Optimizasyon algoritması |
| Epoch | 100 | Eğitim döngüsü sayısı |
| Öğrenme Hızı | 0.01 | Convergence hızı |
| Overfit Epochu | 60 | Overfitting başlangıcı |
| **Final Doğruluk** | **0.90** | **Hedef validation accuracy** |
| **P-R Dengesi** | **0.0** | **Precision-Recall dengesi** |

### Precision-Recall Dengesi Parametresi 🎯

Bu parametre modelinizin precision ve recall arasındaki dengeyi kontrol eder:

| Değer | Açıklama | Model Davranışı |
|-------|----------|-----------------|
| **-1.0** | **Recall Odaklı** | Daha kapsayıcı, az kaçırır, fazla pozitif |
| **-0.5** | Recall Ağırlıklı | Recall'a öncelik verir |
| **0.0** | **Dengeli** | Precision ve Recall eşit önemde |
| **+0.5** | Precision Ağırlıklı | Precision'a öncelik verir |  
| **+1.0** | **Precision Odaklı** | Daha seçici, az yanlış pozitif |

**Örnek Kullanım:**
- **Tıbbi teşhis**: `-0.8` (hasta kaçırmamak için recall odaklı)
- **Spam filtresi**: `+0.6` (normal maili spam yapmamak için precision odaklı)
- **Genel sınıflandırma**: `0.0` (dengeli yaklaşım)

## Simülasyon İyileştirmeleri ✨

### SON GÜNCELLEME - Final Doğruluk Problemi Çözüldü! 🎯
- **SORUN**: Hedef final doğruluk (örn: 0.84) ile gerçek sonuç (örn: 0.59) arasında %25 fark
- **ÇÖZÜM**: Güçlü final ayarlama algoritması eklendi
- **SONUÇ**: Artık %98-102 hassasiyetle hedef doğruluğa ulaşır

### Yeni Özellikler:
- **Gerçekçi Training Accuracy**: Hedeften 8% yüksek olabilir (maksimum 0.98)
- **Güçlü Final Ayarlama**: Son 20 epoch'ta hedef doğruluğa zorla ulaştırır
- **Hafif Overfitting**: Overfit etkisi daha yumuşak (0.3x faktör)
- **Debug Çıktıları**: Hedef ve gerçekleşen doğruluk konsola yazdırılır
- **Yumuşak Eğriler**: Daha doğal görünümlü grafik eğrileri

### Test Edildi:
✅ VGG-16: Hedef 0.758 → Gerçek ~0.758 (±%2)
✅ EfficientNet: Hedef 0.789 → Gerçek ~0.789 (±%2)  
✅ ViT: Hedef 0.842 → Gerçek ~0.842 (±%2)

## Çıktılar

### Metrikler
- Training/Validation Loss & Accuracy
- F1, Precision, Recall
- ROC-AUC
- Overfit analizi

### Grafikler
- Loss/Accuracy eğrileri
- Model karşılaştırma grafikleri
- Detaylı epoch-epoch analizi

## İpuçları

- **Tek model test** → Seçenek 1
- **Çoklu karşılaştırma** → Seçenek 2 (alt-menüden manuel/CSV seç)
- **Grafik kaydetme** → Seçenek 6
- **Ayar değişikliği** → Seçenek 7

## Örnek Kullanım

```python
# 1. Programı başlat
python model_simulator.py

# 2. Tek model simülasyonu (Seçenek 1)
Model Adı: ResNet50
Optimizer: Adam
Epoch: 150
# ... diğer parametreler (Enter = varsayılan)

# 3. Sonuçları görüntüle (Seçenek 3)
# 4. Grafikleri incele (Seçenek 5)
# 5. CSV'e kaydet (Seçenek 4)
```

## Sorun Giderme

- **Import hatası**: `pip install -r requirements.txt`
- **Grafik görünmüyor**: Matplotlib backend kontrol et
- **CSV hatası**: Dosya yolu ve formatını kontrol et

## Karşılaştırmalı Matris & Isı Haritaları (Seçenek 7) 🔥

### Yeni Görselleştirme Türleri:

#### 1. Performans Karşılaştırma Matrisi
- **Amaç**: Modellerin ana performans metriklerini karşılaştır
- **İçerik**: Validation Accuracy, F1, Precision, Recall, Test Accuracy, ROC-AUC
- **Görsel**: Yeşil-Sarı-Kırmızı ısı haritası
- **Çıktı**: Performans istatistikleri tablosu

#### 2. Korelasyon Isı Haritası
- **Amaç**: Metriklerin birbirleri ile ilişkisini göster
- **İçerik**: Tüm sayısal metriklerin korelasyon matrisi
- **Görsel**: Mavi-Beyaz-Kırmızı ısı haritası (üçgen maske)
- **Çıktı**: Yüksek korelasyonlar listesi (|r| > 0.7)

#### 3. Confusion Matrix Simülasyonu
- **Amaç**: Her model için tahmin edilen confusion matrix
- **Hesaplama**: Precision, Recall, Accuracy değerlerinden simülasyon
- **Görsel**: Her model için ayrı mavi tonlarda ısı haritası
- **Limit**: Maksimum 6 model (performans için)

#### 4. Metrikler Arası İlişki Matrisi
- **Amaç**: Scatter plot matrisi ile ilişkileri göster
- **İçerik**: Ana 6 metrik arası tüm ikili ilişkiler
- **Görsel**: Diagonal'da histogram, diğerlerinde scatter plot
- **Çıktı**: Her scatter plot'ta korelasyon katsayısı

#### 5. Model Performans Dendrogramı
- **Amaç**: Hiyerarşik kümeleme ile model benzerliklerini göster
- **Algoritma**: Ward linkage method
- **Çıktı**: Ağaç diagram + benzerlik açıklamaları
- **Gereksinim**: Scipy kütüphanesi (`pip install scipy`)

#### 6. Tüm Matrisleri Göster
- Yukarıdaki 5 görselleştirmeyi sırayla gösterir
- Her görsel arasında kullanıcı onayı bekler
- Sonunda toplu kaydetme seçeneği sunar

### Kullanım Örnekleri:

```bash
# Ana menüden
7 → Karşılaştırmalı Matris & Isı Haritaları

# Alt-menü seçenekleri:
1 → Performans karşılaştırma
2 → Korelasyon analizi  
3 → Confusion matrix'ler
4 → Metrik ilişkileri
5 → Model benzerlik ağacı
6 → Hepsini göster
```

### Özellikler:
- **Otomatik kaydetme**: Matrix_visualizations/ dizinine
- **Hata yönetimi**: Eksik kütüphane uyarıları
- **Performans optimizasyonu**: Büyük veri setleri için limit
- **İnteraktif**: Her görsel için detaylı açıklamalar

### Teknik Detaylar:
- **Renk paletleri**: Seaborn ile profesyonel görünüm
- **Çözünürlük**: Config dosyasından ayarlanabilir DPI
- **Format**: PNG formatında kaydetme
- **Boyut**: Otomatik figure boyutlandırma
