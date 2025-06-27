# Model Simülasyon Terminali

Bu uygulama, makine öğrenmesi eğitim süreçlerini simüle etmek için geli### Sınıf Yapısı

```python
class ModelSimulator:
    def __init__(self)                           # DataFrame başlatma
    def run_new_simulation(self)                 # Tekil simülasyon döngüsü
    def run_batch_simulation(self)               # Toplu simülasyon döngüsü
    def load_from_csv(self)                      # CSV'den hazır veri yükleme
    def _get_user_config(self)                   # Kullanıcı input'u alma
    def _read_batch_from_csv(self, filepath)     # CSV dosyasından veri okuma
    def _simulate_curves(self, config)           # Eğri simülasyonu
    def _calculate_metrics(self, curves)         # Metrik hesaplama
    def _plot_results(self, curves)              # Görselleştirme
    def display_results_table(self)              # Tablo görüntüleme
    def save_to_csv(self)                        # CSV kaydetme
```raktif bir Python uygulamasıdır.

## Özellikler

- **Sınıf Tabanlı Yapı**: `ModelSimulator` sınıfı ile temiz ve organize kod yapısı
- **İnteraktif Terminal Arayüzü**: Kullanıcı dostu menü sistemi
- **Tekil ve Toplu Simülasyon**: Hem tek model hem de çoklu model simülasyonu desteği
- **CSV İçe/Dışa Aktarma**: Hazır verileri yükleme ve sonuçları kaydetme
- **Gerçekçi Simülasyonlar**: Exponential decay ve overfitting etkilerini içeren matematiksel modeller
- **Gelişmiş Görselleştirme**: 
  - Temel performans karşılaştırma grafikleri
  - Detaylı epoch-epoch analiz grafikleri
  - Overfitting analizi ve metrik görselleştirme
  - Etkileşimli model seçimi (tekli, çoklu, aralık)
- **Yapılandırılabilir Ayarlar**: JSON tabanlı config sistemi ile tüm parametreler ayarlanabilir
- **Veri Yönetimi**: Pandas DataFrame ile sonuç tablosu ve CSV işlemleri
- **Analiz Metrikleri**: Detaylı performans analizi ve istatistiksel ölçümler
- **Hata Toleransı**: Geçersiz girişlerde bile devam edebilir sistem
- **Mükerrer Veri Yönetimi**: Otomatik tekrar eden kayıt temizleme
- **Grafik Kaydetme**: Yüksek çözünürlüklü PNG formatında grafik kaydetme

## Kurulum

### Gereksinimler
```bash
pip install -r requirements.txt
```

Gerekli paketler:
- pandas >= 1.5.0
- numpy >= 1.21.0  
- matplotlib >= 3.5.0

## Kullanım

Uygulamayı çalıştırmak için:
```bash
python model_simulator.py
```

### Ana Menü Seçenekleri

1. **Yeni Tekil Model Simülasyonu Yap**: Tek bir model eğitimi simüle eder
2. **Toplu Model Simülasyonu Yap**: Birden fazla modeli tek seferde simüle eder (Manuel giriş veya CSV dosyasından)
3. **Hazır Tablodan Veri Yükle (CSV)**: Harici CSV dosyasından hazır model verilerini içe aktarır
4. **Sonuç Tablosunu Görüntüle**: Tüm simülasyon sonuçlarını tablo halinde gösterir
5. **Sonuçları CSV Olarak Kaydet**: Sonuçları CSV dosyasına aktarır
6. **Model Simülasyonunu Görselleştir**: Detaylı epoch-epoch görselleştirme ile grafikleri gösterir
7. **Grafikleri Kaydet**: Oluşturulan grafikleri PNG formatında kaydeder
8. **Ayarları Yönet**: Simülasyon parametrelerini ve sistem ayarlarını düzenler
9. **Çıkış**: Uygulamadan çıkar

### Simülasyon Parametreleri

Yeni simülasyon oluştururken aşağıdaki parametreleri ayarlayabilirsiniz:

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| Model Adı | VGG19 | Simüle edilecek model adı |
| Optimizasyon Algoritması | Adam | Kullanılan optimizer |
| Toplam Epoch | 100 | Eğitim epoch sayısı |
| Öğrenme Hızı Etkisi | 0.01 | Öğrenme hızının convergence üzerindeki etkisi |
| Minimum Kayıp | 0.1 | Ulaşılabilir minimum loss değeri |
| Maksimum Başarı | 0.95 | Ulaşılabilir maksimum accuracy |
| Gürültü Seviyesi | 0.02 | Eğri üzerindeki rastgele varyasyon |
| Overfit Başlangıç Epochu | 60 | Overfitting'in başladığı epoch |
| Overfit Eğimi | 0.001 | Overfitting'in şiddeti |

### Hesaplanan Metrikler

Simülasyon sonrasında aşağıdaki metrikler otomatik olarak hesaplanır:

- **En Düşük Kayıp Epochu**: Validation loss'un minimum olduğu epoch
- **Overfit Başlangıç Epochu**: Overfitting'in başladığı epoch
- **Eğrilik**: Loss eğrisinin ikinci türev ortalaması
- **Sapma**: Training ve validation loss arasındaki ortalama fark
- **Overfit Eğilimi**: Son epochlarda validation loss artış eğilimi

## Çıktılar

### Grafikler
Her simülasyon sonrasında iki grafik gösterilir:
1. **Loss Grafiği**: Training ve validation loss eğrileri
2. **Accuracy Grafiği**: Training ve validation accuracy eğrileri

### CSV Kayıtları
Sonuç tablosu CSV formatında kaydedilebilir ve şu sütunları içerir:
- Model bilgileri (ad, optimizer, öğrenme oranı, epoch sayısı)
- Final performans metrikleri (accuracy, loss)
- Analiz sonuçları (overfit başlangıcı, eğrilik, sapma)

## Teknik Detaylar

### Simülasyon Algoritması

1. **Training Loss**: 
   ```
   loss = 2.0 * exp(-learning_rate * epoch) + min_loss + noise
   ```

2. **Validation Loss**: 
   - İlk epochlarda training loss ile benzer
   - Overfit başlangıcından sonra doğrusal artış

3. **Training Accuracy**:
   ```
   accuracy = max_acc * (1 - exp(-learning_rate * epoch * 1.5))
   ```

4. **Validation Accuracy**:
   - İlk epochlarda training accuracy ile benzer  
   - Overfit başlangıcından sonra kademeli azalış

### Sınıf Yapısı

```python
class ModelSimulator:
    def __init__(self)                     # DataFrame başlatma
    def run_new_simulation(self)           # Tekil simülasyon döngüsü
    def run_batch_simulation(self)         # Toplu simülasyon döngüsü
    def _get_user_config(self)             # Kullanıcı input'u alma
    def _simulate_curves(self, config)     # Eğri simülasyonu
    def _calculate_metrics(self, curves)   # Metrik hesaplama
    def _plot_results(self, curves)        # Görselleştirme
    def display_results_table(self)        # Tablo görüntüleme
    def save_to_csv(self)                  # CSV kaydetme
```

## Örnek Kullanım

### Tekil Simülasyon
1. Uygulamayı çalıştırın
2. Menüden "1" seçin
3. Parametreleri girin (Enter = varsayılan)
4. Grafikler otomatik açılacak
5. Sonuçları "3" ile görüntüleyin

### Toplu Simülasyon  
1. Uygulamayı çalıştırın
2. Menüden "2" seçin
3. Toplu veri formatında model bilgilerini girin
4. İşlem raporunu inceleyin
5. Sonuçları "3" ile görüntüleyin
6. İsteğe bağlı "4" ile CSV olarak kaydedin

## Sorun Giderme

- **Import hatası**: `pip install -r requirements.txt` çalıştırın
- **Grafik görünmüyor**: Matplotlib backend ayarlarını kontrol edin
- **CSV kaydetme hatası**: Dosya yolunu ve yazma izinlerini kontrol edin

## Katkıda Bulunma

Bu proje eğitim amaçlıdır. İyileştirme önerileri için issue açabilirsiniz.

### Hazır Veri Yükleme (CSV İçe Aktarma)

Önceden hazırlanmış model sonuçlarını doğrudan tabloya aktarmak için "3. Hazır Tablodan Veri Yükle (CSV)" seçeneğini kullanın.

#### Özellikler
- **Doğrudan İçe Aktarma**: Simülasyon yapmadan hazır verileri tabloya ekler
- **Sütun Uyumluluk Kontrolü**: CSV dosyasının format uyumluluğunu otomatik kontrol eder
- **Mükerrer Kayıt Yönetimi**: Aynı model adına sahip kayıtların üzerine yazar
- **Detaylı Raporlama**: Yüklenen, güncellenen ve temizlenen kayıt sayılarını gösterir

#### Gerekli CSV Formatı

CSV dosyası tam olarak aşağıdaki sütunlara sahip olmalıdır:

```csv
Model Adı,Optimizasyon Algoritması,Öğrenme Oranı,Epoch Sayısı,Nihai Eğitim Başarımı,Nihai Doğrulama Başarımı,Nihai Eğitim Kaybı,Nihai Doğrulama Kaybı,En Düşük Kayıp Epochu,Overfit Başlangıç Epochu,Eğrilik,Sapma,Overfit Eğilimi
ResNet152,Adam,0.001,200,0.9845,0.9234,0.0234,0.1456,78,85,0.0012,0.0456,0.0234
EfficientNet-B7,AdamW,0.002,150,0.9876,0.9345,0.0156,0.1234,65,72,0.0015,0.0378,0.0189
```

#### Hata Yönetimi
- **Dosya Bulunamadı**: Yanlış dosya yolu durumunda açık hata mesajı
- **Format Uyumsuzluğu**: Sütun isimleri uyuşmazsa işlem durdurulur
- **Bozuk CSV**: Geçersiz format durumunda kullanıcı bilgilendirilir
- **Mükerrer Veriler**: Aynı model adına sahip kayıtlar otomatik olarak güncellenir

#### Örnek Kullanım
1. Menüden "3" seçin
2. Dosya yolunu girin (örn: `ready_models_table.csv`)
3. Sistem otomatik olarak:
   - Dosyayı okur
   - Format uyumluluğunu kontrol eder
   - Verileri birleştirir
   - Mükerrer kayıtları temizler
   - Sonuç raporunu gösterir

Birden fazla modeli tek seferde simüle etmek için "2. Toplu Model Simülasyonu Yap" seçeneğini kullanın.

#### Veri Girişi Yöntemleri

**1. Manuel String Girişi**
```
ModelAdı,OptimizasyonAlgoritması,ÖğrenmeHızıEtkisi,OverfitBaşlangıçEpochu,OverfitEğimi; Model2,...
```

**2. CSV Dosyasından Okuma**
- CSV dosya yolunu belirtin
- Otomatik sütun tanıma ve düzeltme
- Başlıklı veya başlıksız dosya desteği

#### CSV Dosya Formatları

**✅ Doğru Başlık Formatı:**
```csv
model_name,optimizer,learning_rate_effect,overfit_start_epoch,overfit_slope
VGG19,Adam,0.01,60,0.001
ResNet50,SGD,0.05,70,0.002
```

**⚠️ Yanlış Başlık Formatı (Otomatik Düzeltilir):**
```csv
ModelAdı,OptimizasyonAlgoritması,ÖğrenmeHızı,OverfitBaşlangıç,OverfitEğim
VGG19,Adam,0.01,60,0.001
ResNet50,SGD,0.05,70,0.002
```

**📝 Başlıksız Format (Otomatik Tanınır):**
```csv
VGG19,Adam,0.01,60,0.001
ResNet50,SGD,0.05,70,0.002
```

#### Örnek Manuel Kullanım
```
Model_A,Adam,0.05,60,0.005; Model_B,SGD,0.08,80,0.002; Model_C,RMSprop,0.03,70,0.003
```

#### Özellikler
- **Otomatik Ayrıştırma**: Girilen string otomatik olarak model parametrelerine ayrıştırılır
- **CSV Okuma**: Farklı sütun isimleri olan CSV dosyalarını otomatik düzeltir
- **Hata Toleransı**: Geçersiz parametreli modeller atlanır, diğerleri işlenir
- **Varsayılan Değerler**: Belirtilmeyen parametreler için makul varsayılanlar kullanılır
- **Toplu İşlem**: Grafik çizdirme devre dışı bırakılır (performans için)
- **Detaylı Rapor**: İşlem sonunda başarılı/başarısız model sayıları gösterilir
- **Akıllı Hata Mesajları**: CSV okuma hatalarında özel uyarılar

#### Varsayılan Parametreler
- Toplam Epoch: 100
- Minimum Kayıp: 0.1  
- Maksimum Başarı: 0.95
- Gürültü Seviyesi: 0.02

## Gelişmiş Görselleştirme

### Model Simülasyonu Görselleştirme (Seçenek 6)

Bu özellik iki aşamalı görselleştirme sunar:

#### 1. Temel Performans Grafikleri
- **Model Doğrulama Başarımı**: Tüm modellerin validation accuracy karşılaştırması
- **F1, Precision, Recall**: Metrik karşılaştırma çubuk grafikleri  
- **Overfit Eğilimi Analizi**: Modellerin overfitting durumu (renk kodlu)
- **Test Başarımı vs ROC-AUC**: Scatter plot ile korelasyon analizi

#### 2. Detaylı Epoch-Epoch Analizi

Temel grafikler gösterildikten sonra kullanıcıdan hangi modellerin detaylarını görmek istediği sorulur:

**Model Seçim Formatları:**
- **Tekil**: `3` (3. modeli seç)
- **Çoklu**: `1,3,5` (1., 3. ve 5. modelleri seç)
- **Aralık**: `[1-4]` veya `[2-5]` (belirli aralıktaki modelleri seç)
- **İptal**: Boş bırak veya Ctrl+C

**Detaylı Grafik İçeriği:**
- **Training & Validation Loss**: Epoch bazında kayıp eğrileri
- **Training & Validation Accuracy**: Epoch bazında başarım eğrileri
- **Overfitting Analizi**: Loss farkı ve overfitting bölgesi gösterimi
- **Performans Metrikleri**: Tüm hesaplanan metriklerin özet tablosu

#### Etkileşimli Kaydetme
- Her detaylı grafik için ayrı kaydetme seçeneği
- Toplu kaydetme için indis seçimi desteği
- Otomatik dosya adlandırma: `detailed_[model_no]_[model_name].png`

### Grafik Kaydetme (Seçenek 7)

Seçenek 7, doğrudan görselleştirme menüsünü açar ve tüm grafik kaydetme seçeneklerini sunar.

## Yapılandırılabilir Ayarlar (Seçenek 8)

Yeni config.json tabanlı ayar sistemi ile tüm simülasyon parametreleri özelleştirilebilir.

### Ayar Kategorileri

#### 1. Simülasyon Ayarları
```json
"simulation_settings": {
    "random_seed": 42,           // Rastgelelik tohumu (tutarlılık için)
    "noise_level": 0.02,         // Eğri gürültü seviyesi
    "min_loss": 0.1,             // Minimum ulaşılabilir kayıp
    "max_accuracy": 0.95         // Maksimum ulaşılabilir doğruluk
}
```

#### 2. Varsayılan Model Parametreleri
```json
"default_model_parameters": {
    "total_epochs": 100,         // Varsayılan epoch sayısı
    "learning_rate_effect": 0.01, // Öğrenme hızı etkisi
    "overfit_start_epoch": 60,   // Overfit başlangıç epochu
    "overfit_slope": 0.001,      // Overfit eğimi
    "target_final_accuracy": 0.90 // Hedef final doğruluk
}
```

#### 3. Grafik Ayarları
```json
"graphics_settings": {
    "output_directory": "./outgraph", // Çıktı dizini
    "dpi": 300,                      // Grafik çözünürlüğü
    "figure_size": [12, 8],          // Temel grafik boyutu
    "detailed_figure_size": [15, 10] // Detaylı grafik boyutu
}
```

#### 4. Metrik Hesaplama Ayarları
```json
"metrics_settings": {
    "model_bias_range": [-0.6, 0.6],        // Model bias aralığı
    "precision_adjustment_factor": [0.94, 1.06], // Precision ayarlama
    "external_val_range": [0.85, 0.98],     // External validation aralığı
    "test_acc_range": [0.88, 0.99],         // Test accuracy aralığı
    "roc_auc_range": [1.02, 1.15]           // ROC-AUC aralığı
}
```

#### 5. Görüntüleme Ayarları
```json
"display_settings": {
    "decimal_places": 4,           // Ondalık basamak sayısı
    "max_model_name_length": 15,   // Max model adı uzunluğu
    "table_width": 80              // Tablo genişliği
}
```

### Ayar Yönetimi Menüsü

1. **Simülasyon Ayarları**: Rastgelelik ve gürültü parametreleri
2. **Varsayılan Model Parametreleri**: Yeni simülasyonlarda kullanılan defaults
3. **Grafik Ayarları**: Çıktı dizini, DPI, boyutlar
4. **Metrik Hesaplama Ayarları**: Performans metriklerinin hesaplanma şekli
5. **Görüntüleme Ayarları**: Tablo formatı ve sayısal hassasiyet
6. **Tüm Ayarları Görüntüle**: Mevcut ayarların detaylı listesi
7. **Ayarları Varsayılana Sıfırla**: Factory reset
8. **Geri Dön**: Ana menüye dön

### Config Dosyası

Ayarlar `config.json` dosyasında saklanır ve şu özellikler sunar:
- **Otomatik Yükleme**: Program başlangıcında ayarlar otomatik yüklenir
- **Hata Toleransı**: Dosya yoksa veya bozuksa varsayılan ayarlar kullanılır
- **Otomatik Kaydetme**: Ayar değişiklikleri otomatik kaydedilir
- **UTF-8 Desteği**: Türkçe karakter desteği

## Dosya Yapısı

```
model_simulator.py          # Ana uygulama
config.json                 # Ayar dosyası
outgraph/                   # Grafik çıktıları (otomatik oluşur)
├── model_analysis.png      # Temel karşılaştırma grafikleri
├── detailed_1_VGG19.png    # Model detay grafikleri
└── detailed_2_ResNet50.png
README.md                   # Bu doküman
requirements.txt            # Python bağımlılıkları
```

## Gelişmiş Kullanım Örnekleri

### Toplu Analiz İş Akışı
1. `python model_simulator.py` ile uygulamayı başlat
2. Seçenek 2 ile toplu simülasyon yap (CSV'den oku)
3. Seçenek 4 ile sonuç tablosunu incele
4. Seçenek 6 ile görselleştirme yap:
   - Genel grafikleri incele
   - İlginç modelleri detaylı analiz için seç (örn: `[1-3]`)
   - Grafikleri kaydet
5. Seçenek 5 ile sonuçları CSV olarak kaydet

### Ayar Optimizasyonu İş Akışı
1. Seçenek 8 ile ayarları aç
2. Simülasyon ayarlarını güncelle (random_seed, noise_level)
3. Varsayılan parametreleri optimize et
4. Grafik ayarlarını ihtiyaca göre düzenle
5. Test simülasyonu çalıştır
6. Sonuçları değerlendir

### Karşılaştırmalı Analiz
1. Farklı optimizerlar için toplu simülasyon
2. Görselleştirme ile overfitting eğilimlerini karşılaştır
3. F1, Precision, Recall metriklerini analiz et
4. En iyi performanslı modellerin detaylarını incele
5. Sonuçları raporla

## Performans İpuçları

- **Toplu Simülasyon**: Büyük batch'ler için grafik rendering devre dışı
- **Detaylı Görselleştirme**: Sadece ilginç modeller için kullan
- **Config Ayarları**: Random seed sabit tutarak tekrarlanabilir sonuçlar al
- **Grafik Kaydetme**: Yüksek DPI sadece sunum grafikleri için kullan

## Sorun Giderme

### Config Dosyası Sorunları
- **Dosya Bulunamadı**: Uygulama varsayılan ayarları kullanır ve yeni dosya oluşturur
- **JSON Hatası**: Hatalı JSON formatında dosyayı düzelt veya sil
- **İzin Hatası**: Dosya yazma izinlerini kontrol et

### Grafik Sorunları  
- **Grafik Açılmıyor**: Matplotlib backend ayarlarını kontrol et
- **Kaydetme Hatası**: Çıktı dizini izinlerini ve disk alanını kontrol et
- **Çözünürlük Sorunları**: DPI ayarını düşür veya figure_size küçült

### Performans Sorunları
- **Yavaş Detaylı Grafik**: Daha az model seç veya figure_size küçült
- **Bellek Sorunu**: Toplu simülasyonda model sayısını azalt
- **CSV Okuma Hatası**: Dosya formatını ve encoding'i kontrol et
