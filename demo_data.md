# Demo: Toplu Model Simülasyonu

Bu dosya, toplu model simülasyonu özelliğini test etmek için örnek veriler içerir.

## Test Verileri

### Manuel String Girişi Örnekleri

#### Örnek 1: Temel Modeller
```
VGG19,Adam,0.01,60,0.001; ResNet50,SGD,0.05,70,0.002; DenseNet121,RMSprop,0.03,65,0.0015
```

#### Örnek 2: Çeşitli Optimizers  
```
Model_A,Adam,0.02,50,0.003; Model_B,SGD,0.08,80,0.005; Model_C,AdamW,0.015,55,0.002; Model_D,RMSprop,0.04,75,0.004
```

#### Örnek 3: Farklı Overfit Senaryoları
```
EarlyOverfit,Adam,0.1,30,0.01; LateOverfit,SGD,0.01,90,0.001; ModerateOverfit,RMSprop,0.05,60,0.005
```

#### Örnek 4: Hata Testi (bazı geçersiz veriler içerir)
```
ValidModel,Adam,0.02,60,0.003; InvalidModel,SGD; AnotherValid,RMSprop,0.03,70,0.002
```

### CSV Dosya Örnekleri

Çalışma dizininde aşağıdaki test dosyaları bulunmaktadır:

#### test_models_correct.csv (Doğru başlıklar)
```csv
model_name,optimizer,learning_rate_effect,overfit_start_epoch,overfit_slope
VGG19,Adam,0.01,60,0.001
ResNet50,SGD,0.05,70,0.002
DenseNet121,RMSprop,0.03,65,0.0015
EfficientNet,AdamW,0.02,55,0.0025
```

#### test_models_wrong_headers.csv (Yanlış başlıklar - otomatik düzeltilir)
```csv
ModelAdı,OptimizasyonAlgoritması,ÖğrenmeHızı,OverfitBaşlangıç,OverfitEğim
VGG19,Adam,0.01,60,0.001
ResNet50,SGD,0.05,70,0.002
DenseNet121,RMSprop,0.03,65,0.0015
```

### CSV Veri Yükleme Örnekleri

Çalışma dizininde aşağıdaki hazır veri dosyaları bulunmaktadır:

#### ready_models_table.csv (Doğru format - yüklenebilir)
```csv
Model Adı,Optimizasyon Algoritması,Öğrenme Oranı,Epoch Sayısı,Nihai Eğitim Başarımı,Nihai Doğrulama Başarımı,Nihai Eğitim Kaybı,Nihai Doğrulama Kaybı,En Düşük Kayıp Epochu,Overfit Başlangıç Epochu,Eğrilik,Sapma,Overfit Eğilimi
ResNet152,Adam,0.001,200,0.9845,0.9234,0.0234,0.1456,78,85,0.0012,0.0456,0.0234
EfficientNet-B7,AdamW,0.002,150,0.9876,0.9345,0.0156,0.1234,65,72,0.0015,0.0378,0.0189
DenseNet201,SGD,0.01,180,0.9798,0.9187,0.0289,0.1567,89,95,0.0018,0.0489,0.0267
MobileNetV3,RMSprop,0.005,120,0.9723,0.9098,0.0345,0.1678,56,63,0.0021,0.0523,0.0298
Vision Transformer,AdamW,0.0001,100,0.9912,0.9456,0.0123,0.1123,45,52,0.0009,0.0312,0.0156
```

#### wrong_format_table.csv (Yanlış format - hata verir)
```csv
Model Name,Optimizer,Learning Rate,Epochs,Train Acc,Val Acc,Train Loss,Val Loss
ResNet50,Adam,0.001,100,0.95,0.92,0.05,0.08
VGG16,SGD,0.01,80,0.93,0.89,0.07,0.11
```

## Kullanım Talimatları

### Manuel String Girişi
1. model_simulator.py'yi çalıştırın
2. Menüden "2" seçin  
3. Veri girişi yöntemi olarak "1" seçin
4. Yukarıdaki string örneklerden birini kopyalayıp yapıştırın
5. Sonuçları inceleyin

### CSV Dosyasından Okuma
1. model_simulator.py'yi çalıştırın
2. Menüden "2" seçin
3. Veri girişi yöntemi olarak "2" seçin
4. Test dosyası yolunu girin (örn: test_models_correct.csv)
5. Sonuçları inceleyin

### CSV Veri Yükleme
1. model_simulator.py'yi çalıştırın
2. Menüden "3" seçin
3. Dosya yolunu girin (örn: ready_models_table.csv)
4. Sonuçları "4" ile görüntüleyin

### Test Senaryoları
- **Doğru Format**: `ready_models_table.csv` → Başarılı yükleme
- **Yanlış Format**: `wrong_format_table.csv` → Sütun uyumsuzluğu hatası
- **Olmayan Dosya**: `nonexistent.csv` → Dosya bulunamadı hatası
- **Mükerrer Veriler**: Aynı dosyayı iki kez yükleme → Otomatik güncelleme

## Beklenen Sonuçlar

- Geçerli modeller başarıyla simüle edilecek
- Geçersiz veriler atlanacak ve uyarı verilecek  
- İşlem sonu raporu gösterilecek
- Tüm sonuçlar ana tabloya eklenecek
