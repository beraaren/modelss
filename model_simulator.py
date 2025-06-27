import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import os
import json

class ModelSimulator:
    """
    Makine öğrenmesi eğitim süreçlerini simüle etmek için sınıf.
    """
    
    def __init__(self):
        """
        Başlatıcı metot. Simülasyon sonuçlarını depolamak için DataFrame oluşturur.
        """
        self.results_df = pd.DataFrame(columns=[
            'Model Adı', 'Optimizasyon Algoritması', 'Öğrenme Oranı', 'Epoch Sayısı',
            'Nihai Eğitim Başarımı', 'Nihai Doğrulama Başarımı', 
            'Nihai Eğitim Kaybı', 'Nihai Doğrulama Kaybı',
            'F1 Skoru', 'Precision', 'Recall', 'Harici Validasyon Başarımı',
            'Test Başarımı', 'ROC-AUC', 'Confusion Matrix Diagonal Ortalaması',
            'En Düşük Kayıp Epochu', 'Overfit Başlangıç Epochu', 
            'Eğrilik', 'Sapma', 'Overfit Eğilimi'
        ])
        
        # Config dosyasını yükle
        self.config = self._load_config()
        
        # Rastgelelik seed'ini ayarla
        np.random.seed(self.config['simulation_settings']['random_seed'])
        
    def run_new_simulation(self):
        """
        Yeni bir simülasyonu baştan sona yürüten ana metot.
        """
        print("\n=== Yeni Model Simülasyonu Başlatılıyor ===")
        
        # 1. Kullanıcıdan parametreleri al
        config = self._get_user_config()
        
        # 2. Eğrileri simüle et
        curves = self._simulate_curves(config)
        
        # 3. Metrikleri hesapla
        metrics = self._calculate_metrics(curves, config)
        
        # 4. Yeni satır oluştur ve DataFrame'e ekle
        new_row = {
            'Model Adı': config['model_name'],
            'Optimizasyon Algoritması': config['optimizer'],
            'Öğrenme Oranı': config['learning_rate_effect'],
            'Epoch Sayısı': config['total_epochs'],
            'Nihai Eğitim Başarımı': curves['training_accuracy'][-1],
            'Nihai Doğrulama Başarımı': curves['validation_accuracy'][-1],
            'Nihai Eğitim Kaybı': curves['training_loss'][-1],
            'Nihai Doğrulama Kaybı': curves['validation_loss'][-1],
            'F1 Skoru': metrics['f1_score'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'Harici Validasyon Başarımı': metrics['external_val_acc'],
            'Test Başarımı': metrics['test_acc'],
            'ROC-AUC': metrics['roc_auc'],
            'Confusion Matrix Diagonal Ortalaması': metrics['cm_diagonal_avg'],
            'En Düşük Kayıp Epochu': metrics['min_loss_epoch'],
            'Overfit Başlangıç Epochu': metrics['overfit_start_epoch'],
            'Eğrilik': metrics['curvature'],
            'Sapma': metrics['divergence'],
            'Overfit Eğilimi': metrics['overfit_tendency']
        }
        
        # DataFrame'e yeni satırı ekle
        new_index = len(self.results_df)
        for key, value in new_row.items():
            self.results_df.loc[new_index, key] = value
        
        # 5. Sonuçları görselleştir
        self._plot_results(curves, metrics)
        
        print("\n✓ Simülasyon başarıyla tamamlandı ve sonuçlar tabloya eklendi.")
        
    def _get_user_config(self) -> Dict[str, Any]:
        """
        Kullanıcıdan interaktif olarak simülasyon parametrelerini alan özel metot.
        """
        print("\nSimülasyon parametrelerini girin (Enter = varsayılan değer):")
        
        config = {}
        defaults = self.config['default_model_parameters']
        
        # Model adı
        model_name = input("Model Adı (VGG19): ").strip()
        config['model_name'] = model_name if model_name else "VGG19"
        
        # Optimizasyon algoritması
        optimizer = input("Optimizasyon Algoritması (Adam): ").strip()
        config['optimizer'] = optimizer if optimizer else "Adam"
        
        # Toplam epoch
        epochs_input = input(f"Toplam Epoch ({defaults['total_epochs']}): ").strip()
        config['total_epochs'] = int(epochs_input) if epochs_input else defaults['total_epochs']
        
        # Öğrenme hızı etkisi
        lr_input = input(f"Öğrenme Hızı Etkisi ({defaults['learning_rate_effect']}): ").strip()
        config['learning_rate_effect'] = float(lr_input) if lr_input else defaults['learning_rate_effect']
        
        # Minimum kayıp
        min_loss_input = input(f"Minimum Kayıp ({self.config['simulation_settings']['min_loss']}): ").strip()
        config['min_loss'] = float(min_loss_input) if min_loss_input else self.config['simulation_settings']['min_loss']
        
        # Maksimum başarı
        max_acc_input = input(f"Maksimum Başarı ({self.config['simulation_settings']['max_accuracy']}): ").strip()
        config['max_accuracy'] = float(max_acc_input) if max_acc_input else self.config['simulation_settings']['max_accuracy']
        
        # Gürültü seviyesi
        noise_input = input(f"Gürültü Seviyesi ({self.config['simulation_settings']['noise_level']}): ").strip()
        config['noise_level'] = float(noise_input) if noise_input else self.config['simulation_settings']['noise_level']
        
        # Overfit başlangıç epochu
        overfit_start_input = input(f"Overfit Başlangıç Epochu ({defaults['overfit_start_epoch']}): ").strip()
        config['overfit_start_epoch'] = int(overfit_start_input) if overfit_start_input else defaults['overfit_start_epoch']
        
        # Overfit eğimi
        overfit_slope_input = input(f"Overfit Eğimi ({defaults['overfit_slope']}): ").strip()
        config['overfit_slope'] = float(overfit_slope_input) if overfit_slope_input else defaults['overfit_slope']
        
        # Final doğruluk hedefi
        final_acc_input = input(f"Final Doğruluk Hedefi ({defaults['target_final_accuracy']}): ").strip()
        config['target_final_accuracy'] = float(final_acc_input) if final_acc_input else defaults['target_final_accuracy']
        
        # Precision-Recall dengesi
        pr_balance_input = input(f"Precision-Recall Dengesi (-1=Recall odaklı, 0=Dengeli, +1=Precision odaklı) ({defaults.get('precision_recall_balance', 0)}): ").strip()
        config['precision_recall_balance'] = float(pr_balance_input) if pr_balance_input else defaults.get('precision_recall_balance', 0)
        
        return config
    
    def _simulate_curves(self, config: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Gerçekçi ve doğal görünen bir eğitim/validasyon eğrisi üretir.
        - Final doğruluk eğitim doğruluğu olur ve bu değer, finalin yarısı ±%15 rastgele sapma ile seçilir.
        - Overfit etkisi polinom eğriyle, tepe noktası overfit epochunda olacak şekilde uygulanır.
        - Eğri 5 parçaya bölünür, her parçaya ve her elemana azalan rastgelelik eklenir.
        """
        epochs = np.arange(1, config['total_epochs'] + 1)
        n = len(epochs)
        settings = self.config['simulation_settings']
        min_loss = config.get('min_loss', settings['min_loss'])
        max_acc = config.get('max_accuracy', settings['max_accuracy'])
        noise_min = settings.get('min_noise', 0.005)
        noise_max = settings.get('max_noise', 0.03)

        # 1. Final doğrulukları belirle
        target_val_acc = config.get('target_final_accuracy', max_acc)
        val_final = target_val_acc
        # Eğitim doğruluğu validation'dan her zaman yüksek olmalı (gerçekçi)
        train_final = val_final + np.random.uniform(0.02, 0.08)  # %2-8 arası fark
        train_final = np.clip(train_final, val_final + 0.01, 0.99)

        # 2. Başlangıç doğrulukları
        train_start = np.random.uniform(0.15, 0.25)
        val_start = train_start * np.random.uniform(0.85, 0.95)  # Val her zaman train'den düşük başlar

        # 3. Overfit epochu ve polinom eğrileri
        overfit_epoch = config.get('overfit_start_epoch', 60)
        overfit_slope = config.get('overfit_slope', 0.001)
        overfit_x = overfit_epoch / n
        x = np.linspace(0, 1, n)
        
        # Sürekli ve yumuşak eğriler oluştur - overfit epochunda sıçrama olmasın
        train_curve = np.zeros(n)
        val_curve = np.zeros(n)
        
        # Önce tüm eğriyi sürekli olarak oluştur
        # Training: Baştan sona sürekli artış (overfit'ten sonra yavaşlar)
        for i in range(n):
            progress = x[i]
            # Training eğrisi: başta hızlı, sonra yavaş (sürekli)
            if progress <= overfit_x:
                train_curve[i] = train_start + (train_final * 0.85 - train_start) * (progress / overfit_x) ** 0.6
            else:
                # Overfit noktasındaki değerden devam et (süreklilik için)
                overfit_train_val = train_start + (train_final * 0.85 - train_start) * 1.0 ** 0.6
                post_progress = (progress - overfit_x) / (1 - overfit_x)
                train_curve[i] = overfit_train_val + (train_final - overfit_train_val) * (post_progress ** 1.3)
        
        # Validation: Başta training'i takip eder, overfit'ten sonra yavaşça düşer
        val_peak_progress = min(overfit_x * 1.1, 0.9)  # Peak biraz overfit'ten sonra
        for i in range(n):
            progress = x[i]
            if progress <= val_peak_progress:
                # Peak'e kadar yükseliş
                val_curve[i] = val_start + (val_final * 1.02 - val_start) * (progress / val_peak_progress) ** 0.7
            else:
                # Peak'ten sonra yavaş düşüş
                peak_val = val_start + (val_final * 1.02 - val_start) * 1.0 ** 0.7
                post_peak_progress = (progress - val_peak_progress) / (1 - val_peak_progress)
                drop_amount = overfit_slope * 30 * post_peak_progress ** 0.6
                val_curve[i] = peak_val - drop_amount
        
        # Son değerleri yumuşak ayarlama - sert geçişler yok
        # Son 5 epoch'u yumuşak bir geçişle hedefe doğru çek
        final_smooth_epochs = min(5, n // 10)  # En fazla 5 epoch veya toplam epochun %10'u
        if final_smooth_epochs > 0:
            # Mevcut son değerlerden hedefe yumuşak geçiş
            current_val_end = val_curve[-final_smooth_epochs:]
            current_train_end = train_curve[-final_smooth_epochs:]
            
            # Yumuşak interpolasyon ile hedefe ulaş
            smooth_weights = np.linspace(0, 1, final_smooth_epochs)
            for i in range(final_smooth_epochs):
                idx = -final_smooth_epochs + i
                weight = smooth_weights[i]
                val_curve[idx] = current_val_end[i] * (1 - weight) + val_final * weight
                train_curve[idx] = current_train_end[i] * (1 - weight) + train_final * weight
        else:
            # Sadece son değeri ayarla
            val_curve[-1] = val_final
            train_curve[-1] = train_final
        
        val_curve = np.clip(val_curve, 0.1, 0.99)
        train_curve = np.clip(train_curve, 0.1, 0.99)

        # 4. Eğrileri 5 parçaya böl, her parçaya ayrı rastgelelik uygula
        part_size = n // 5
        for i in range(5):
            start = i * part_size
            end = (i + 1) * part_size if i < 4 else n
            # Son parçada (son 20%) çok az gürültü uygula
            if i == 4:  # Son parça
                part_noise = noise_min * 0.3  # Çok düşük gürültü
            else:
                part_noise = np.random.uniform(noise_min, noise_max)
            val_curve[start:end] += np.random.normal(0, part_noise, end - start)
            train_curve[start:end] += np.random.normal(0, part_noise * 0.7, end - start)

        # 5. Her elemana sona yaklaştıkça çok yumuşak azalan rastgelelik uygula
        # Son 20 epoch'ta neredeyse hiç gürültü olmasın
        decay = np.linspace(1, 0.05, n)  # 0.2'den 0.05'e düşürdük
        # Son 20 epoch'ta ekstra yumuşatma
        last_20_mask = np.zeros(n)
        last_20_start = max(0, n - 20)
        last_20_mask[last_20_start:] = np.linspace(0.8, 0.02, n - last_20_start)  # Son 20'de çok az
        final_decay = np.minimum(decay, 1 - last_20_mask)
        
        val_curve += np.random.normal(0, noise_max * final_decay, n)
        train_curve += np.random.normal(0, noise_max * 0.7 * final_decay, n)
        val_curve = np.clip(val_curve, 0.1, 0.99)
        train_curve = np.clip(train_curve, 0.1, 0.99)

        # 6. Loss eğrileri: doğruluk eğrisinin tersi, min_loss'a yaklaşır
        train_loss = 1.5 * (1 - train_curve) + min_loss
        val_loss = 1.5 * (1 - val_curve) + min_loss
        train_loss += np.random.normal(0, noise_min, n)
        val_loss += np.random.normal(0, noise_min, n)
        train_loss = np.maximum(train_loss, 0.01)
        val_loss = np.maximum(val_loss, 0.01)

        print(f"🎯 Simülasyon: Final eğitim doğruluğu={train_final:.3f}, Final validasyon doğruluğu={val_final:.3f}")
        print(f"   Overfit epochu: {overfit_epoch}, Overfit eğimi: {overfit_slope}")
        print(f"   P-R Dengesi: {config.get('precision_recall_balance', 0):.2f} ({'Precision' if config.get('precision_recall_balance', 0) > 0 else 'Recall' if config.get('precision_recall_balance', 0) < 0 else 'Dengeli'} odaklı)")

        return {
            'training_loss': train_loss,
            'validation_loss': val_loss,
            'training_accuracy': train_curve,
            'validation_accuracy': val_curve,
            'epochs': epochs
        }
    
    def _calculate_metrics(self, curves: Dict[str, np.ndarray], config: Dict[str, Any]) -> Dict[str, float]:
        """
        Üretilen eğrileri ve konfigürasyonu alır, analiz metriklerini hesaplar.
        """
        val_loss = curves['validation_loss']
        train_loss = curves['training_loss']
        val_acc = curves['validation_accuracy']
        train_acc = curves['training_accuracy']
        
        # En düşük kayıp epochu
        min_loss_epoch = np.argmin(val_loss) + 1
        
        # Overfit başlangıç epochu (validation loss artmaya başladığı nokta)
        overfit_start_epoch = config['overfit_start_epoch']
        
        # Eğrilik (loss curve'ün ikinci türevi)
        loss_diff = np.diff(val_loss)
        curvature = np.mean(np.abs(np.diff(loss_diff)))
        
        # Sapma (train ve validation loss arasındaki ortalama fark)
        divergence = np.mean(np.abs(val_loss - train_loss))
        
        # Overfit eğilimi (son epochlarda validation loss artışı)
        if len(val_loss) > 10:
            recent_trend = np.mean(val_loss[-10:]) - np.mean(val_loss[-20:-10])
            overfit_tendency = max(0, recent_trend)
        else:
            overfit_tendency = 0
        
        # Gelişmiş metrikler hesaplama
        final_val_acc = val_acc[-1]
        metrics_config = self.config['metrics_settings']
        
        # Precision-Recall trade-off simülasyonu
        # Model bias değerini config'ten al (CSV'den veya kullanıcıdan gelen parametre)
        model_bias = config.get('precision_recall_balance', 0.0)  # -1 ile +1 arası
        model_bias = np.clip(model_bias, -1.0, 1.0)  # Güvenlik için sınırla
        
        # Base precision ve recall değerleri accuracy etrafında (daha az rastgelelik)
        precision_adj = metrics_config['precision_adjustment_factor']
        recall_adj = metrics_config['recall_adjustment_factor']
        base_precision = final_val_acc * np.random.uniform(precision_adj[0], precision_adj[1])
        base_recall = final_val_acc * np.random.uniform(recall_adj[0], recall_adj[1])
        
        # Trade-off etkisini model_bias'a göre uygula (rastgelelik azaltıldı)
        precision_trade = metrics_config['precision_trade_off']
        recall_trade = metrics_config['recall_trade_off']
        
        if model_bias > 0:  # Precision odaklı model (daha seçici)
            precision = base_precision + abs(model_bias) * precision_trade[1]  # Precision artır
            recall = base_recall - abs(model_bias) * recall_trade[0]           # Recall azalt
        else:  # Recall odaklı model (daha kapsayıcı)
            precision = base_precision - abs(model_bias) * precision_trade[0]  # Precision azalt
            recall = base_recall + abs(model_bias) * recall_trade[1]           # Recall artır
        
        # Değerleri 0-1 aralığında tut
        precision = np.clip(precision, 0, 1)
        recall = np.clip(recall, 0, 1)
        
        # F1 Skoru - Precision ve Recall'dan matematiksel olarak hesaplanır
        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0
        f1_score = np.clip(f1_score, 0, 1)
        
        # Harici validasyon başarımı (validation accuracy'den biraz düşük)
        ext_val_range = metrics_config['external_val_range']
        external_val_acc = final_val_acc * np.random.uniform(ext_val_range[0], ext_val_range[1])
        external_val_acc = np.clip(external_val_acc, 0, 1)
        
        # Test başarımı (genellikle validation'dan biraz düşük)
        test_range = metrics_config['test_acc_range']
        test_acc = final_val_acc * np.random.uniform(test_range[0], test_range[1])
        test_acc = np.clip(test_acc, 0, 1)
        
        # ROC-AUC skoru (genellikle accuracy'den biraz yüksek)
        roc_range = metrics_config['roc_auc_range']
        roc_auc = final_val_acc * np.random.uniform(roc_range[0], roc_range[1])
        roc_auc = np.clip(roc_auc, 0, 1)
        
        # Confusion Matrix Diagonal Ortalaması
        cm_range = metrics_config['cm_diagonal_range']
        cm_diagonal_avg = final_val_acc * np.random.uniform(cm_range[0], cm_range[1])
        cm_diagonal_avg = np.clip(cm_diagonal_avg, 0, 1)
        
        # Ondalık basamak sayısını config'ten al
        decimal_places = self.config['display_settings']['decimal_places']
        
        return {
            'min_loss_epoch': min_loss_epoch,
            'overfit_start_epoch': overfit_start_epoch,
            'curvature': round(curvature, decimal_places),
            'divergence': round(divergence, decimal_places),
            'overfit_tendency': round(overfit_tendency, decimal_places),
            'f1_score': round(f1_score, decimal_places),
            'precision': round(precision, decimal_places),
            'recall': round(recall, decimal_places),
            'external_val_acc': round(external_val_acc, decimal_places),
            'test_acc': round(test_acc, decimal_places),
            'roc_auc': round(roc_auc, decimal_places),
            'cm_diagonal_avg': round(cm_diagonal_avg, decimal_places)
        }
    
    def _plot_results(self, curves: Dict[str, np.ndarray], metrics: Dict[str, float]):
        """
        matplotlib kullanarak kayıp ve başarı grafiklerini çizer.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        epochs = curves['epochs']
        
        # Loss grafiği
        ax1.plot(epochs, curves['training_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, curves['validation_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.axvline(x=metrics['min_loss_epoch'], color='g', linestyle='--', 
                   label=f'Min Loss Epoch: {metrics["min_loss_epoch"]}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy grafiği
        ax2.plot(epochs, curves['training_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, curves['validation_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.axvline(x=metrics['overfit_start_epoch'], color='orange', linestyle='--', 
                   label=f'Overfit Start: {metrics["overfit_start_epoch"]}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def display_results_table(self):
        """
        Sınıfta tutulan DataFrame'ini formatlı ve okunaklı bir şekilde terminale yazdırır.
        """
        if self.results_df.empty:
            print("\n⚠️  Henüz gösterilecek bir sonuç yok.")
            return
        
        print("\n" + "="*80)
        print("                    SIMÜLASYON SONUÇLARI TABLOSU")
        print("="*80)
        
        # DataFrame'i daha okunaklı formatta göster
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 15)
        
        # Sayısal değerleri yuvarla
        display_df = self.results_df.copy()
        numeric_columns = ['Öğrenme Oranı', 'Nihai Eğitim Başarımı', 'Nihai Doğrulama Başarımı', 
                          'Nihai Eğitim Kaybı', 'Nihai Doğrulama Kaybı', 'F1 Skoru', 'Precision', 
                          'Recall', 'Harici Validasyon Başarımı', 'Test Başarımı', 'ROC-AUC',
                          'Confusion Matrix Diagonal Ortalaması', 'Eğrilik', 'Sapma', 'Overfit Eğilimi']
        
        for col in numeric_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(4)
        
        print(display_df.to_string(index=False))
        print("="*80)
        print(f"Toplam {len(self.results_df)} simülasyon sonucu görüntüleniyor.\n")
    
    def save_to_csv(self):
        """
        Kullanıcıdan dosya adı ister ve DataFrame'i CSV olarak kaydeder.
        """
        if self.results_df.empty:
            print("\n⚠️  Kaydedilecek veri yok.")
            return
        
        filename = input("\nCSV dosya adını girin ").strip()
        
        # .csv uzantısı yoksa ekle
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        try:
            # Mevcut çalışma dizinine kaydet
            filepath = os.path.join(os.getcwd(), filename)
            self.results_df.to_csv(filepath, index=False, encoding='utf-8')
            print(f"✓ Tablo başarıyla '{filename}' olarak kaydedildi.")
            print(f"Dosya konumu: {filepath}")
        except Exception as e:
            print(f"❌ Kaydetme hatası: {e}")
    
    def run_batch_simulation(self):
        """
        Toplu simülasyon işlevselliğini yöneten metot.
        Manuel giriş veya CSV dosyasından model parametrelerini okur ve simülasyon çalıştırır.
        """
        print("\n=== Toplu Model Simülasyonu Başlatılıyor ===")
        
        print("\nVeri girişi yöntemi seçin:")
        print("1. Manuel veri girişi (string format)")
        print("2. CSV dosyasından oku")
        
        choice = input("\nSeçiminiz (1-2): ").strip()
        
        batch_input = ""
        
        if choice == "1":
            print("\nLütfen model verilerini aşağıdaki formatta girin (her modeli noktalı virgülle ayırın):")
            print("ModelAdı,OptimizasyonAlgoritması,ÖğrenmeHızıEtkisi,OverfitBaşlangıçEpochu,OverfitEğimi,FinalDoğruluk,PrecisionRecallDengesi; ModelAdı2,Algoritma2, ...")
            print("\nÖRNEK:")
            print("Model_A,Adam,0.05,60,0.005,0.92,0.2; Model_B,SGD,0.08,80,0.002,0.89,-0.3")
            print("\nNOT: PrecisionRecallDengesi: -1=Recall odaklı, 0=Dengeli, +1=Precision odaklı")
            print("     Diğer parametreler varsayılan değerlerle atanacak.")
            
            # Kullanıcıdan toplu veri al
            batch_input = input("\nToplu model verilerini girin: ").strip()
            
        elif choice == "2":
            csv_path = input("\nCSV dosya yolunu girin: ").strip()
            if not csv_path:
                print("❌ Boş dosya yolu. İşlem iptal edildi.")
                return
                
            try:
                batch_input = self._read_batch_from_csv(csv_path)
                print(f"✓ CSV dosyasından {len(batch_input.split(';'))} model verisi okundu.")
            except Exception as e:
                print(f"❌ CSV okuma hatası: {e}")
                print("📋 Bu hata büyük olasılıkla etiket/sütun isimlerinin uyumsuzluğundan kaynaklanıyor.")
                print("💡 Beklenen sütun isimleri: model_name, optimizer, learning_rate_effect, overfit_start_epoch, overfit_slope, target_final_accuracy, precision_recall_balance")
                print("💡 Alternatif olarak, sütunları bu sırada düzenleyip başlık satırını kaldırabilirsiniz.")
                return
        else:
            print("❌ Geçersiz seçim. İşlem iptal edildi.")
            return
        
        if not batch_input:
            print("❌ Boş veri girişi. İşlem iptal edildi.")
            return
        
        # Model verilerini ayrıştır
        models_data = []
        model_strings = batch_input.split(';')
        
        print(f"\n📊 {len(model_strings)} model verisi işlenmeye başlıyor...")
        
        for i, model_str in enumerate(model_strings, 1):
            model_str = model_str.strip()
            if not model_str:
                continue
                
            try:
                # Parametreleri virgülle ayır
                params = [param.strip() for param in model_str.split(',')]
                
                # En az 7 parametre olmalı (precision_recall_balance eklendi)
                if len(params) < 7:
                    print(f"⚠️  UYARI: Model {i} için girilen parametreler eksik (7 parametre gerekli, {len(params)} verildi). Bu model atlanıyor.")
                    continue
                
                # Parametreleri dict'e dönüştür
                model_config = {
                    'model_name': params[0],
                    'optimizer': params[1],
                    'learning_rate_effect': float(params[2]),
                    'overfit_start_epoch': int(params[3]),
                    'overfit_slope': float(params[4]),
                    'target_final_accuracy': float(params[5]),
                    'precision_recall_balance': float(params[6]),  # Yeni parametre
                    # Varsayılan değerler
                    'total_epochs': 100,
                    'min_loss': 0.1,
                    'max_accuracy': 0.95,
                    'noise_level': 0.02
                }
                
                models_data.append(model_config)
                print(f"✓ Model {i}: '{params[0]}' başarıyla ayrıştırıldı.")
                
            except (ValueError, IndexError) as e:
                print(f"⚠️  UYARI: Model {i} için girilen parametreler geçersiz ({str(e)}). Bu model atlanıyor.")
                continue
        
        if not models_data:
            print("❌ Hiçbir geçerli model verisi bulunamadı. İşlem iptal edildi.")
            return
        
        # Toplu simülasyon başlat
        successful_count = 0
        skipped_count = len(model_strings) - len(models_data)
        
        print(f"\n🚀 {len(models_data)} model için simülasyon başlatılıyor...")
        
        for i, config in enumerate(models_data, 1):
            try:
                print(f"\n📈 Model {i}/{len(models_data)}: '{config['model_name']}' simüle ediliyor...")
                
                # Eğrileri simüle et
                curves = self._simulate_curves(config)
                
                # Metrikleri hesapla
                metrics = self._calculate_metrics(curves, config)
                
                # Yeni satır oluştur (run_batch için)
                new_row = {
                    'Model Adı': config['model_name'],
                    'Optimizasyon Algoritması': config['optimizer'],
                    'Öğrenme Oranı': config['learning_rate_effect'],
                    'Epoch Sayısı': config['total_epochs'],
                    'Nihai Eğitim Başarımı': curves['training_accuracy'][-1],
                    'Nihai Doğrulama Başarımı': curves['validation_accuracy'][-1],
                    'Nihai Eğitim Kaybı': curves['training_loss'][-1],
                    'Nihai Doğrulama Kaybı': curves['validation_loss'][-1],
                    'F1 Skoru': metrics['f1_score'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'Harici Validasyon Başarımı': metrics['external_val_acc'],
                    'Test Başarımı': metrics['test_acc'],
                    'ROC-AUC': metrics['roc_auc'],
                    'Confusion Matrix Diagonal Ortalaması': metrics['cm_diagonal_avg'],
                    'En Düşük Kayıp Epochu': metrics['min_loss_epoch'],
                    'Overfit Başlangıç Epochu': metrics['overfit_start_epoch'],
                    'Eğrilik': metrics['curvature'],
                    'Sapma': metrics['divergence'],
                    'Overfit Eğilimi': metrics['overfit_tendency']
                }
                
                # DataFrame'e yeni satırı ekle
                new_index = len(self.results_df)
                for key, value in new_row.items():
                    self.results_df.loc[new_index, key] = value
                
                successful_count += 1
                print(f"✓ '{config['model_name']}' başarıyla simüle edildi.")
                
            except Exception as e:
                print(f"❌ '{config['model_name']}' simülasyonu başarısız: {str(e)}")
                skipped_count += 1
                continue
        
        # İşlem özeti
        print(f"\n📋 Toplu işlem tamamlandı!")
        print(f"✅ {successful_count} model başarıyla simüle edildi")
        if skipped_count > 0:
            print(f"⚠️  {skipped_count} model geçersiz veri nedeniyle atlandı")
        print(f"📊 Toplam {len(self.results_df)} simülasyon sonucu tabloda mevcut.")
    
    def _read_batch_from_csv(self, filepath: str) -> str:
        """
        CSV dosyasından toplu simülasyon verilerini okur ve string formatına çevirir.
        """
        try:
            # Önce başlıklı olarak okumayı dene
            df = pd.read_csv(filepath)
            
            # Beklenen sütun isimleri (varsayılan etiket sırası)
            expected_columns = ['model_name', 'optimizer', 'learning_rate_effect', 'overfit_start_epoch', 'overfit_slope', 'target_final_accuracy', 'precision_recall_balance']
            
            # Eğer sütun sayısı doğruysa ama isimler farklıysa, varsayılan sırayla eşleştir
            if len(df.columns) == len(expected_columns):
                if not all(col in df.columns for col in expected_columns):
                    print(f"\n⚠️  CSV sütun isimleri beklenen format ile uyuşmuyor.")
                    print(f"Beklenen: {expected_columns}")
                    print(f"Bulunan: {list(df.columns)}")
                    print("Varsayılan etiket sırası kullanılarak deneniyor...")
                    
                    # Sütunları varsayılan sırayla yeniden adlandır
                    df.columns = expected_columns
            
            # Eğer başlık satırı sayısal veri içeriyorsa, başlıksız olarak tekrar oku
            elif len(df.columns) == len(expected_columns):
                try:
                    # İlk satırın sayısal olup olmadığını kontrol et
                    first_row = df.iloc[0]
                    if all(isinstance(val, (int, float)) or (isinstance(val, str) and val.replace('.', '').replace('-', '').isdigit()) for val in first_row[2:]):
                        print("\n💡 Başlıksız CSV dosyası tespit edildi. Varsayılan sütun isimleri atanıyor...")
                        # Başlıksız olarak tekrar oku
                        df = pd.read_csv(filepath, header=None, names=expected_columns)
                except:
                    pass
                    
            # DataFrame'i string formatına çevir
            batch_data = []
            for _, row in df.iterrows():
                try:
                    model_data = f"{row['model_name']},{row['optimizer']},{row['learning_rate_effect']},{row['overfit_start_epoch']},{row['overfit_slope']},{row['target_final_accuracy']},{row['precision_recall_balance']}"
                    batch_data.append(model_data)
                except KeyError as e:
                    print(f"❌ Satır atlandı - eksik sütun: {e}")
                    continue
                except Exception as e:
                    print(f"❌ Satır işlenirken hata: {e}")
                    continue
            
            if not batch_data:
                raise ValueError("CSV dosyasından geçerli veri okunamadı")
            
            return "; ".join(batch_data)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV dosyası bulunamadı: {filepath}")
        except pd.errors.EmptyDataError:
            raise ValueError("CSV dosyası boş")
        except pd.errors.ParserError as e:
            raise ValueError(f"CSV ayrıştırma hatası: {e}")
        except Exception as e:
            raise Exception(f"CSV okuma hatası: {e}")
    
    def _create_output_directory(self):
        """Grafik çıktıları için dizin oluşturur."""
        output_dir = self.config['graphics_settings']['output_directory']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✓ '{output_dir}' dizini oluşturuldu.")
        return output_dir
    
    def visualize_models(self):
        """Model simülasyonlarını görselleştiren ana metot."""
        if self.results_df.empty:
            print("\n⚠️  Görselleştirilecek simülasyon verisi yok.")
            return
        
        print("\n=== Model Simülasyonu Görselleştirme ===")
        print(f"Toplam {len(self.results_df)} model simülasyonu mevcut:")
        
        # Mevcut modelleri listele
        for i, model_name in enumerate(self.results_df['Model Adı'], 1):
            print(f"{i}. {model_name}")
        
        # Basit görselleştirme oluştur
        figure_size = self.config['graphics_settings']['figure_size']
        plt.figure(figsize=tuple(figure_size))
        
        # Performance comparison
        models = self.results_df['Model Adı'].tolist()
        val_acc = self.results_df['Nihai Doğrulama Başarımı'].tolist()
        
        plt.subplot(2, 2, 1)
        bars = plt.bar(range(len(models)), val_acc, color='skyblue', alpha=0.7)
        plt.title('Model Doğrulama Başarımı')
        plt.ylabel('Başarım')
        plt.xticks(range(len(models)), [m[:10] + '...' if len(m) > 10 else m for m in models], rotation=45)
        
        # F1, Precision, Recall comparison
        plt.subplot(2, 2, 2)
        x = range(len(models))
        width = 0.25
        
        f1_scores = self.results_df['F1 Skoru'].tolist()
        precisions = self.results_df['Precision'].tolist()
        recalls = self.results_df['Recall'].tolist()
        
        plt.bar([i - width for i in x], f1_scores, width, label='F1', alpha=0.8)
        plt.bar(x, precisions, width, label='Precision', alpha=0.8)
        plt.bar([i + width for i in x], recalls, width, label='Recall', alpha=0.8)
        plt.title('F1, Precision, Recall')
        plt.legend()
        plt.xticks(x, [m[:8] + '...' if len(m) > 8 else m for m in models], rotation=45)
        
        # Overfit analysis
        plt.subplot(2, 2, 3)
        overfit = self.results_df['Overfit Eğilimi'].tolist()
        colors = ['red' if x > 0.05 else 'orange' if x > 0.02 else 'green' for x in overfit]
        plt.bar(range(len(models)), overfit, color=colors, alpha=0.7)
        plt.title('Overfit Eğilimi')
        plt.ylabel('Eğilim')
        plt.xticks(range(len(models)), [m[:8] + '...' if len(m) > 8 else m for m in models], rotation=45)
        
        # Scatter plot
        plt.subplot(2, 2, 4)
        test_acc = self.results_df['Test Başarımı'].tolist()
        roc_auc = self.results_df['ROC-AUC'].tolist()
        plt.scatter(test_acc, roc_auc, c='purple', alpha=0.6)
        plt.title('Test Başarımı vs ROC-AUC')
        plt.xlabel('Test Başarımı')
        plt.ylabel('ROC-AUC')
        
        plt.tight_layout()
        plt.show()
        
        # Detaylı görselleştirme seçeneği
        print("\n" + "="*50)
        print("DETAYLI EPOCH-EPOCH GÖRSELLEŞTİRME")
        print("="*50)
        print("Hangi modellerin epoch-epoch detaylı grafiklerini görmek istiyorsunuz?")
        print("• Tek model için: Model numarasını girin (örn: 3)")
        print("• Birden fazla model için: Virgülle ayırın (örn: 1,3,5)")
        print("• Aralık seçimi için: Köşeli parantez kullanın (örn: [1-4] veya [2-5])")
        print("• Boş bırakırsanız veya Ctrl+C ile çıkarsanız detaylı grafik gösterilmez")
        
        try:
            detailed_input = input("\nSeçiminiz: ").strip()
            
            if detailed_input:
                selected_indices = self._parse_model_selection(detailed_input, len(models))
                if selected_indices:
                    self._show_detailed_plots(selected_indices)
                    
                    # Kaydetme seçeneği - detaylı grafiklerle birlikte
                    save_choice = input("\nGrafikleri (detaylı grafikler dahil) kaydetmek istiyor musunuz? (e/h): ").strip().lower()
                    if save_choice == 'e':
                        save_indices_input = input("Hangi indislerdeki grafikleri kaydetmek istiyorsunuz? (boş=hepsi, örn: 1,3 veya [1-4]): ").strip()
                        save_indices = self._parse_model_selection(save_indices_input, len(models)) if save_indices_input else list(range(len(models)))
                        self._save_plots_with_details(save_indices, selected_indices)
                else:
                    print("❌ Geçersiz seçim.")
        except KeyboardInterrupt:
            print("\n⚠️  Detaylı görselleştirme iptal edildi.")
        except Exception as e:
            print(f"❌ Girdi hatası: {e}")
        
        # Genel kaydetme seçeneği (detaylı grafik yoksa)
        if not hasattr(self, '_detailed_shown'):
            save_choice = input("\nTemel grafikleri kaydetmek istiyor musunuz? (e/h): ").strip().lower()
            if save_choice == 'e':
                self._save_plots()
    
    def _save_plots(self):
        """Grafikleri kaydet."""
        output_dir = self._create_output_directory()
        filename = f"model_analysis.png"
        filepath = os.path.join(output_dir, filename)
        dpi = self.config['graphics_settings']['dpi']
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"✓ Grafik kaydedildi: {filepath}")
    
    def save_graphs(self):
        """Grafik kaydetme metodu."""
        if self.results_df.empty:
            print("\n⚠️  Kaydedilecek veri yok.")
            return
        
        self.visualize_models()
    
    def _parse_model_selection(self, input_str: str, total_models: int) -> List[int]:
        """
        Kullanıcının model seçimi girdisini ayrıştırır.
        Örn: "1,3,5" -> [0,2,4], "[1-4]" -> [0,1,2,3]
        """
        try:
            indices = []
            
            # Köşeli parantez ile aralık seçimi
            if '[' in input_str and ']' in input_str:
                # [1-4] formatı
                range_part = input_str.strip('[]')
                if '-' in range_part:
                    start, end = range_part.split('-')
                    start_idx = int(start.strip()) - 1  # 0-indexli yap
                    end_idx = int(end.strip()) - 1
                    indices = list(range(max(0, start_idx), min(total_models, end_idx + 1)))
                else:
                    # [3] formatı
                    idx = int(range_part.strip()) - 1
                    if 0 <= idx < total_models:
                        indices = [idx]
            else:
                # Virgülle ayrılmış liste: "1,3,5"
                parts = input_str.split(',')
                for part in parts:
                    idx = int(part.strip()) - 1  # 0-indexli yap
                    if 0 <= idx < total_models:
                        indices.append(idx)
            
            # Tekrar eden değerleri kaldır ve sırala
            indices = sorted(list(set(indices)))
            return indices
            
        except (ValueError, IndexError) as e:
            print(f"⚠️  Girdi ayrıştırma hatası: {e}")
            return []
    
    def _show_detailed_plots(self, selected_indices: List[int]):
        """
        Seçilen modeller için detaylı epoch-epoch grafikleri gösterir.
        """
        self._detailed_shown = True
        
        for idx in selected_indices:
            # DataFrame satırından değerleri al - pandas veri erişimi
            row_data = self.results_df.iloc[idx]
            model_name = str(row_data.iloc[0])  # Model Adı
            optimizer = str(row_data.iloc[1])   # Optimizasyon Algoritması  
            lr = float(row_data.iloc[2])        # Öğrenme Oranı
            epochs = int(row_data.iloc[3])      # Epoch Sayısı
            overfit_start = int(row_data.iloc[16])  # Overfit Başlangıç Epochu
            val_acc = float(row_data.iloc[5])   # Nihai Doğrulama Başarımı
            
            print(f"\n📊 Model {idx+1}: {model_name} detaylı grafikleri oluşturuluyor...")
            
            # Model parametrelerinden config oluştur
            config = {
                'model_name': model_name,
                'optimizer': optimizer,
                'learning_rate_effect': lr,
                'total_epochs': epochs,
                'min_loss': 0.1,  # Varsayılan değerler
                'max_accuracy': 0.95,
                'noise_level': 0.02,
                'overfit_start_epoch': overfit_start,
                'overfit_slope': 0.001,
                'target_final_accuracy': val_acc
            }
            
            # Eğrileri yeniden simüle et (aynı parametrelerle)
            np.random.seed(42 + idx)  # Tutarlı sonuçlar için seed
            curves = self._simulate_curves(config)
            metrics = self._calculate_metrics(curves, config)
            
            # Detaylı grafik oluştur
            self._create_detailed_plot(curves, metrics, model_name, idx)
    
    def _create_detailed_plot(self, curves: Dict[str, np.ndarray], metrics: Dict[str, float], 
                             model_name: str, model_idx: int):
        """
        Tek bir model için detaylı epoch-epoch grafiği oluşturur.
        """
        detailed_size = self.config['graphics_settings']['detailed_figure_size']
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=tuple(detailed_size))
        fig.suptitle(f'Detaylı Analiz: {model_name}', fontsize=16, fontweight='bold')
        
        epochs = curves['epochs']
        
        # 1. Loss Curves with Details
        ax1.plot(epochs, curves['training_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, curves['validation_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.axvline(x=metrics['min_loss_epoch'], color='green', linestyle='--', 
                   label=f'Min Loss: Epoch {metrics["min_loss_epoch"]}', alpha=0.7)
        ax1.axvline(x=metrics['overfit_start_epoch'], color='orange', linestyle='--', 
                   label=f'Overfit Start: Epoch {metrics["overfit_start_epoch"]}', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy Curves with Details
        ax2.plot(epochs, curves['training_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, curves['validation_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.axhline(y=metrics['external_val_acc'], color='purple', linestyle=':', 
                   label=f'External Val: {metrics["external_val_acc"]:.3f}', alpha=0.7)
        ax2.axhline(y=metrics['test_acc'], color='brown', linestyle=':', 
                   label=f'Test Acc: {metrics["test_acc"]:.3f}', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training & Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Loss difference hesapla (overfitting analizi için)
        loss_diff = curves['validation_loss'] - curves['training_loss']
        
        # 3. Loss Difference (Overfitting Indicator)
        ax3.plot(epochs, loss_diff, 'orange', linewidth=2, label='Val Loss - Train Loss')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.axhline(y=metrics['divergence'], color='red', linestyle='--', 
                   label=f'Avg Divergence: {metrics["divergence"]:.4f}', alpha=0.7)
        ax3.fill_between(epochs, 0, loss_diff, where=(loss_diff > 0), 
                        color='red', alpha=0.3, label='Overfitting Zone')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss Difference')
        ax3.set_title('Overfitting Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Metrics Summary
        ax4.axis('off')
        metrics_text = f"""
        PERFORMANS METRİKLERİ
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Final Validation Accuracy: {curves['validation_accuracy'][-1]:.4f}
        Final Training Accuracy: {curves['training_accuracy'][-1]:.4f}
        
        F1 Score: {metrics['f1_score']:.4f}
        Precision: {metrics['precision']:.4f}
        Recall: {metrics['recall']:.4f}
        
        Test Accuracy: {metrics['test_acc']:.4f}
        ROC-AUC: {metrics['roc_auc']:.4f}
        External Validation: {metrics['external_val_acc']:.4f}
        
        Min Loss Epoch: {metrics['min_loss_epoch']}
        Overfit Start Epoch: {metrics['overfit_start_epoch']}
        Curvature: {metrics['curvature']:.4f}
        Divergence: {metrics['divergence']:.4f}
        Overfit Tendency: {metrics['overfit_tendency']:.4f}
        """
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Bu grafiği kaydetme seçeneği
        save_individual = input(f"Bu model ({model_name}) grafiğini kaydetmek istiyor musunuz? (e/h): ").strip().lower()
        if save_individual == 'e':
            output_dir = self._create_output_directory()
            filename = f"detailed_{model_idx+1}_{model_name.replace(' ', '_')}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()  # Belleği temizle
            print(f"✓ Detaylı grafik kaydedildi: {filepath}")
    
    def _save_plots_with_details(self, save_indices: List[int], detailed_indices: List[int]):
        """
        Hem temel grafikleri hem de seçilen detaylı grafikleri kaydet.
        """
        output_dir = self._create_output_directory()
        
        # Temel grafikleri kaydet
        if save_indices:
            filename = f"model_analysis_overview.png"
            filepath = os.path.join(output_dir, filename)
            # Temel grafik figürünü tekrar oluştur (eğer kaybolmuşsa)
            figure_size = self.config['graphics_settings']['figure_size']
            plt.figure(figsize=tuple(figure_size))
            # ... (temel grafik kodunu buraya ekleyebiliriz ama şimdilik sadece son figürü kaydet)
            dpi = self.config['graphics_settings']['dpi']
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"✓ Genel grafik kaydedildi: {filepath}")
        
        # Detaylı grafikleri kaydet
        for idx in detailed_indices:
            if idx in save_indices:
                row_data = self.results_df.iloc[idx]
                model_name = str(row_data.iloc[0])    # Model Adı
                optimizer = str(row_data.iloc[1])     # Optimizasyon Algoritması
                lr = float(row_data.iloc[2])          # Öğrenme Oranı
                epochs = int(row_data.iloc[3])        # Epoch Sayısı
                overfit_start = int(row_data.iloc[16]) # Overfit Başlangıç Epochu
                val_acc = float(row_data.iloc[5])     # Nihai Doğrulama Başarımı
                
                # Model config'i yeniden oluştur
                config = {
                    'model_name': model_name,
                    'optimizer': optimizer,
                    'learning_rate_effect': lr,
                    'total_epochs': epochs,
                    'min_loss': 0.1,
                    'max_accuracy': 0.95,
                    'noise_level': 0.02,
                    'overfit_start_epoch': overfit_start,
                    'overfit_slope': 0.001,
                    'target_final_accuracy': val_acc
                }
                
                # Eğrileri yeniden simüle et
                np.random.seed(42 + idx)
                curves = self._simulate_curves(config)
                metrics = self._calculate_metrics(curves, config)
                
                # Detaylı grafiği oluştur ve kaydet
                self._create_and_save_detailed_plot(curves, metrics, model_name, idx, output_dir)
    
    def _create_and_save_detailed_plot(self, curves: Dict[str, np.ndarray], metrics: Dict[str, float], 
                                      model_name: str, model_idx: int, output_dir: str):
        """
        Detaylı grafik oluştur ve doğrudan kaydet (göstermeden).
        """
        detailed_size = self.config['graphics_settings']['detailed_figure_size']
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=tuple(detailed_size))
        fig.suptitle(f'Detaylı Analiz: {model_name}', fontsize=16, fontweight='bold')
        
        epochs = curves['epochs']
        
        # Loss Curves
        ax1.plot(epochs, curves['training_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, curves['validation_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.axvline(x=metrics['min_loss_epoch'], color='green', linestyle='--', 
                   label=f'Min Loss: Epoch {metrics["min_loss_epoch"]}', alpha=0.7)
        ax1.axvline(x=metrics['overfit_start_epoch'], color='orange', linestyle='--', 
                   label=f'Overfit Start: Epoch {metrics["overfit_start_epoch"]}', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy Curves
        ax2.plot(epochs, curves['training_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, curves['validation_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.axhline(y=metrics['external_val_acc'], color='purple', linestyle=':', 
                   label=f'External Val: {metrics["external_val_acc"]:.3f}', alpha=0.7)
        ax2.axhline(y=metrics['test_acc'], color='brown', linestyle=':', 
                   label=f'Test Acc: {metrics["test_acc"]:.3f}', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training & Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Overfitting Analysis
        loss_diff = curves['validation_loss'] - curves['training_loss']
        ax3.plot(epochs, loss_diff, 'orange', linewidth=2, label='Val Loss - Train Loss')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.axhline(y=metrics['divergence'], color='red', linestyle='--', 
                   label=f'Avg Divergence: {metrics["divergence"]:.4f}', alpha=0.7)
        ax3.fill_between(epochs, 0, loss_diff, where=(loss_diff > 0), 
                        color='red', alpha=0.3, label='Overfitting Zone')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss Difference')
        ax3.set_title('Overfitting Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Performance Metrics
        ax4.axis('off')
        metrics_text = f"""
        PERFORMANS METRİKLERİ
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Final Validation Accuracy: {curves['validation_accuracy'][-1]:.4f}
        Final Training Accuracy: {curves['training_accuracy'][-1]:.4f}
        
        F1 Score: {metrics['f1_score']:.4f}
        Precision: {metrics['precision']:.4f}
        Recall: {metrics['recall']:.4f}
        
        Test Accuracy: {metrics['test_acc']:.4f}
        ROC-AUC: {metrics['roc_auc']:.4f}
        External Validation: {metrics['external_val_acc']:.4f}
        
        Min Loss Epoch: {metrics['min_loss_epoch']}
        Overfit Start Epoch: {metrics['overfit_start_epoch']}
        Curvature: {metrics['curvature']:.4f}
        Divergence: {metrics['divergence']:.4f}
        Overfit Tendency: {metrics['overfit_tendency']:.4f}
        """
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Kaydet
        filename = f"detailed_{model_idx+1}_{model_name.replace(' ', '_').replace('/', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()  # Belleği temizle
        print(f"✓ Detaylı grafik kaydedildi: {filepath}")
    
    def visualize_matrix_comparisons(self):
        """
        Karşılaştırmalı matris görselleştirmeleri ve ısı haritaları oluşturur.
        """
        if self.results_df.empty:
            print("\n⚠️  Görselleştirilecek veri yok. Önce simülasyon yapmalısınız.")
            return
        
        print("\n" + "="*60)
        print("        KARŞILAŞTIRMALI MATRİS & ISI HARİTALARI")
        print("="*60)
        print("Hangi tür görselleştirme yapmak istiyorsunuz?")
        print("1. Performans Karşılaştırma Matrisi")
        print("2. Korelasyon Isı Haritası")
        print("3. Confusion Matrix Simülasyonu")
        print("4. Metrikler Arası İlişki Matrisi")
        print("5. Model Performans Dendrogramı")
        print("6. Tüm Matrisleri Göster")
        print("7. Ana Menüye Dön")
        print("-"*60)
        
        choice = input("Seçiminiz (1-7): ").strip()
        
        if choice == "1":
            self._show_performance_comparison_matrix()
        elif choice == "2":
            self._show_correlation_heatmap()
        elif choice == "3":
            self._show_confusion_matrix_simulation()
        elif choice == "4":
            self._show_metrics_relationship_matrix()
        elif choice == "5":
            self._show_performance_dendrogram()
        elif choice == "6":
            self._show_all_matrix_visualizations()
        elif choice == "7":
            return
        else:
            print("\n❌ Geçersiz seçim, lütfen tekrar deneyin.")
    
    def _show_performance_comparison_matrix(self):
        """
        Modellerin performans metriklerini karşılaştırmalı matris olarak gösterir.
        """
        print("\n📊 Performans Karşılaştırma Matrisi oluşturuluyor...")
        
        # Ana performans metriklerini seç
        performance_metrics = [
            'Nihai Doğrulama Başarımı', 'F1 Skoru', 'Precision', 'Recall', 
            'Test Başarımı', 'ROC-AUC'
        ]
        
        # Veri hazırlama
        data_matrix = self.results_df[['Model Adı'] + performance_metrics].copy()
        data_matrix = data_matrix.set_index('Model Adı')
        
        # Isı haritası oluştur
        plt.figure(figsize=(12, 8))
        
        # Renk haritası seçimi
        sns.heatmap(data_matrix, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='RdYlGn',
                   center=0.7,
                   cbar_kws={'label': 'Performans Skoru'},
                   linewidths=0.5,
                   square=True)
        
        plt.title('Model Performans Karşılaştırma Matrisi', fontsize=16, fontweight='bold')
        plt.ylabel('Modeller', fontsize=12)
        plt.xlabel('Performans Metrikleri', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        # İstatistiksel özet
        print("\n📈 Performans İstatistikleri:")
        print(data_matrix.describe().round(3))
    
    def _show_correlation_heatmap(self):
        """
        Tüm metriklerin birbirleri ile korelasyon ısı haritasını gösterir.
        """
        print("\n🔥 Korelasyon Isı Haritası oluşturuluyor...")
        
        # Sayısal kolonları seç
        numeric_columns = self.results_df.select_dtypes(include=[np.number]).columns
        correlation_data = self.results_df[numeric_columns]
        
        # Korelasyon matrisini hesapla
        correlation_matrix = correlation_data.corr()
        
        # Isı haritası oluştur
        plt.figure(figsize=(14, 10))
        
        # Maske oluştur (üst üçgen için)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   fmt='.2f', 
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   cbar_kws={'label': 'Korelasyon Katsayısı'},
                   linewidths=0.5)
        
        plt.title('Metrikler Arası Korelasyon Isı Haritası', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        # Yüksek korelasyonları bul
        print("\n🔍 Yüksek Korelasyonlar (|r| > 0.7):")
        # Korelasyon matrisinin mutlak değerini al
        abs_corr = correlation_matrix.abs()
        high_corr_mask = (abs_corr > 0.7) & (abs_corr < 1.0)  # 1.0 olan kendi korelasyonları hariç tut
        
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if high_corr_mask.iloc[i, j]:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i], 
                        correlation_matrix.columns[j], 
                        correlation_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            for col1, col2, corr_val in high_corr_pairs:
                print(f"  • {col1[:25]:25} ↔ {col2[:25]:25} : {corr_val:.3f}")
        else:
            print("  Yüksek korelasyon bulunamadı.")
    
    def _show_confusion_matrix_simulation(self):
        """
        Modeller için confusion matrix simülasyonu yapar ve görselleştirir.
        """
        print("\n🎯 Confusion Matrix Simülasyonu oluşturuluyor...")
        
        # Kaç model olduğunu kontrol et
        n_models = len(self.results_df)
        if n_models > 6:
            print(f"⚠️  {n_models} model var. İlk 6 model için confusion matrix gösterilecek.")
            models_to_show = self.results_df.head(6)
        else:
            models_to_show = self.results_df
        
        # Subplot düzeni hesapla
        n_cols = min(3, len(models_to_show))
        n_rows = (len(models_to_show) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.ravel()
        else:
            axes = axes.ravel()
        
        for idx, (_, model_row) in enumerate(models_to_show.iterrows()):
            # Confusion matrix simülasyonu
            precision = model_row['Precision']
            recall = model_row['Recall']
            val_acc = model_row['Nihai Doğrulama Başarımı']
            
            # Simüle edilmiş confusion matrix değerleri
            # Bu değerler precision, recall ve accuracy'ye uygun olacak şekilde hesaplanır
            total_samples = 1000  # Örnek toplam veri sayısı
            
            # Pozitif sınıf sayısı (yaklaşık olarak)
            positive_samples = int(total_samples * 0.5)  # %50 pozitif varsayımı
            negative_samples = total_samples - positive_samples
            
            # True Positive
            tp = int(positive_samples * recall)
            # False Negative
            fn = positive_samples - tp
            # False Positive (precision'dan hesapla)
            if precision > 0:
                fp = int(tp * (1 - precision) / precision)
            else:
                fp = 0
            # True Negative
            tn = negative_samples - fp
            
            # Değerleri düzelt (negatif olmaması için)
            fp = max(0, fp)
            tn = max(0, tn)
            
            # Confusion matrix
            cm = np.array([[tn, fp], [fn, tp]])
            
            # Normalizasyon
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Görselleştirme
            ax = axes[idx] if len(models_to_show) > 1 else axes[0]
            
            sns.heatmap(cm_normalized, 
                       annot=True, 
                       fmt='.2f', 
                       cmap='Blues',
                       ax=ax,
                       cbar=False,
                       square=True,
                       xticklabels=['Negatif', 'Pozitif'],
                       yticklabels=['Negatif', 'Pozitif'])
            
            model_name = model_row['Model Adı']
            ax.set_title(f'{model_name}\nAcc: {val_acc:.3f}, P: {precision:.3f}, R: {recall:.3f}')
            ax.set_xlabel('Tahmin Edilen')
            ax.set_ylabel('Gerçek')
        
        # Boş subplotları gizle
        for idx in range(len(models_to_show), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Model Confusion Matrix Simülasyonları', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _show_metrics_relationship_matrix(self):
        """
        Metriklerin birbirleri ile ilişkisini scatter plot matrisi olarak gösterir.
        """
        print("\n📊 Metrikler Arası İlişki Matrisi oluşturuluyor...")
        
        # Ana metrikleri seç
        key_metrics = [
            'Nihai Doğrulama Başarımı', 'F1 Skoru', 'Precision', 'Recall',
            'ROC-AUC', 'Overfit Eğilimi'
        ]
        
        # Veri hazırlama
        metrics_data = self.results_df[key_metrics].copy()
        
        # Scatter plot matrisi oluştur
        fig, axes = plt.subplots(len(key_metrics), len(key_metrics), 
                                figsize=(15, 12))
        
        for i, metric1 in enumerate(key_metrics):
            for j, metric2 in enumerate(key_metrics):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal'da histogram göster
                    ax.hist(metrics_data[metric1], bins=10, alpha=0.7, color='skyblue')
                    ax.set_ylabel('Frekans', fontsize=8)
                else:
                    # Scatter plot
                    ax.scatter(metrics_data[metric2], metrics_data[metric1], 
                             alpha=0.7, s=50, c='coral')
                    
                    # Korelasyon katsayısını hesapla ve göster
                    corr_coef = metrics_data[metric1].corr(metrics_data[metric2])
                    ax.text(0.05, 0.95, f'r={corr_coef:.2f}', 
                           transform=ax.transAxes, fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                
                # Eksen etiketleri
                if i == len(key_metrics) - 1:
                    ax.set_xlabel(metric2, fontsize=8)
                if j == 0:
                    ax.set_ylabel(metric1, fontsize=8)
                
                # Tick'leri küçült
                ax.tick_params(axis='both', which='major', labelsize=7)
        
        plt.suptitle('Metrikler Arası İlişki Matrisi', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _show_performance_dendrogram(self):
        """
        Model performanslarına göre hiyerarşik kümeleme dendrogramı gösterir.
        """
        print("\n🌳 Model Performans Dendrogramı oluşturuluyor...")
        
        try:
            from scipy.cluster.hierarchy import dendrogram, linkage
            from scipy.spatial.distance import pdist
        except ImportError:
            print("❌ Scipy kütüphanesi bulunamadı. 'pip install scipy' komutunu çalıştırın.")
            return
        
        # Performans metriklerini seç
        performance_cols = [
            'Nihai Doğrulama Başarımı', 'F1 Skoru', 'Precision', 'Recall', 
            'Test Başarımı', 'ROC-AUC'
        ]
        
        # Veri hazırlama
        data_for_clustering = self.results_df[performance_cols].values
        model_names = self.results_df['Model Adı'].tolist()
        
        # Hiyerarşik kümeleme
        linkage_matrix = linkage(data_for_clustering, method='ward')
        
        # Dendrogram oluştur
        plt.figure(figsize=(12, 8))
        
        dendrogram(linkage_matrix,
                  labels=model_names,
                  leaf_rotation=90,
                  leaf_font_size=10)
        
        plt.title('Model Performans Dendrogramı\n(Hiyerarşik Kümeleme)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Modeller', fontsize=12)
        plt.ylabel('Mesafe', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        print("\n📋 Dendrogram Açıklaması:")
        print("• Yakın modeller benzer performans gösterir")
        print("• Yüksek dallar daha farklı model gruplarını gösterir")
        print("• Düşük mesafe = Yüksek benzerlik")
    
    def _show_all_matrix_visualizations(self):
        """
        Tüm matris görselleştirmelerini sırayla gösterir.
        """
        print("\n🎨 Tüm matris görselleştirmeleri sırayla gösteriliyor...")
        
        visualizations = [
            ("Performans Karşılaştırma Matrisi", self._show_performance_comparison_matrix),
            ("Korelasyon Isı Haritası", self._show_correlation_heatmap),
            ("Confusion Matrix Simülasyonu", self._show_confusion_matrix_simulation),
            ("Metrikler Arası İlişki Matrisi", self._show_metrics_relationship_matrix),
            ("Model Performans Dendrogramı", self._show_performance_dendrogram)
        ]
        
        for i, (name, func) in enumerate(visualizations, 1):
            print(f"\n{i}/{len(visualizations)} - {name}")
            try:
                func()
                if i < len(visualizations):
                    input("\nDevam etmek için Enter'a basın...")
            except Exception as e:
                print(f"❌ {name} görselleştirme hatası: {e}")
                continue
        
        print("\n✅ Tüm matris görselleştirmeleri tamamlandı!")
        
        # Kaydetme seçeneği
        save_choice = input("\nGrafikleri kaydetmek ister misiniz? (e/h): ").lower().strip()
        if save_choice == 'e':
            self._save_matrix_visualizations()
    
    def _save_matrix_visualizations(self):
        """
        Matris görselleştirmelerini dosyaya kaydet.
        """
        output_dir = self._create_output_directory()
        matrix_dir = os.path.join(output_dir, "matrix_visualizations")
        
        if not os.path.exists(matrix_dir):
            os.makedirs(matrix_dir)
        
        print(f"\n💾 Matris görselleştirmeleri kaydediliyor: {matrix_dir}")
        
        # Her görselleştirmeyi ayrı ayrı kaydet
        visualizations = [
            ("performance_comparison_matrix", self._show_performance_comparison_matrix),
            ("correlation_heatmap", self._show_correlation_heatmap),
            ("confusion_matrix_simulation", self._show_confusion_matrix_simulation),
            ("metrics_relationship_matrix", self._show_metrics_relationship_matrix),
            ("performance_dendrogram", self._show_performance_dendrogram)
        ]
        
        for filename, func in visualizations:
            try:
                # Görselleştirmeyi oluştur (show=False)
                func()
                
                # Kaydet
                filepath = os.path.join(matrix_dir, f"{filename}.png")
                dpi = self.config['graphics_settings']['dpi']
                plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
                plt.close()  # Belleği temizle
                
                print(f"  ✅ {filename}.png kaydedildi")
                
            except Exception as e:
                print(f"  ❌ {filename} kaydedilemedi: {e}")
        
        print(f"\n📁 Tüm matris görselleştirmeleri şu dizine kaydedildi:\n{matrix_dir}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Config dosyasından ayarları yükler.
        """
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✓ Ayarlar dosyası yüklendi: {config_path}")
            return config
        except FileNotFoundError:
            print(f"⚠️  Ayarlar dosyası bulunamadı: {config_path}")
            print("Varsayılan ayarlar kullanılıyor...")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"❌ Ayarlar dosyası JSON hatası: {e}")
            print("Varsayılan ayarlar kullanılıyor...")
            return self._get_default_config()
        except Exception as e:
            print(f"❌ Ayarlar dosyası yükleme hatası: {e}")
            print("Varsayılan ayarlar kullanılıyor...")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Varsayılan ayarları döndürür.
        """
        return {
            "simulation_settings": {
                "random_seed": 42,
                "noise_level": 0.02,
                "min_loss": 0.1,
                "max_accuracy": 0.95
            },
            "default_model_parameters": {
                "total_epochs": 100,
                "learning_rate_effect": 0.01,
                "overfit_start_epoch": 60,
                "overfit_slope": 0.001,
                "target_final_accuracy": 0.90,
                "precision_recall_balance": 0.0
            },
            "graphics_settings": {
                "output_directory": "./outgraph",
                "dpi": 300,
                "figure_size": [12, 8],
                "detailed_figure_size": [15, 10]
            },
            "metrics_settings": {
                "model_bias_range": [-0.6, 0.6],
                "precision_adjustment_factor": [0.94, 1.06],
                "recall_adjustment_factor": [0.94, 1.06],
                "precision_trade_off": [0.12, 0.15],
                "recall_trade_off": [0.12, 0.15],
                "external_val_range": [0.85, 0.98],
                "test_acc_range": [0.88, 0.99],
                "roc_auc_range": [1.02, 1.15],
                "cm_diagonal_range": [0.95, 1.05]
            },
            "display_settings": {
                "decimal_places": 4,
                "max_model_name_length": 15,
                "table_width": 80
            }
        }
    
    def save_config(self):
        """
        Mevcut ayarları config dosyasına kaydeder.
        """
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            print(f"✓ Ayarlar dosyası kaydedildi: {config_path}")
        except Exception as e:
            print(f"❌ Ayarlar dosyası kaydetme hatası: {e}")
    
    def update_config_settings(self):
        """
        Kullanıcının ayarları güncellemesine izin verir.
        """
        print("\n=== AYARLAR YÖNETİMİ ===")
        print("Hangi ayar kategorisini güncellemek istiyorsunuz?")
        print("1. Simülasyon Ayarları (rastgelelik, gürültü vs.)")
        print("2. Varsayılan Model Parametreleri")
        print("3. Grafik Ayarları")
        print("4. Metrik Hesaplama Ayarları")
        print("5. Görüntüleme Ayarları")
        print("6. Tüm Ayarları Görüntüle")
        print("7. Ayarları Varsayılana Sıfırla")
        print("8. Geri Dön")
        
        choice = input("\nSeçiminiz (1-8): ").strip()
        
        if choice == "1":
            self._update_simulation_settings()
        elif choice == "2":
            self._update_default_model_parameters()
        elif choice == "3":
            self._update_graphics_settings()
        elif choice == "4":
            self._update_metrics_settings()
        elif choice == "5":
            self._update_display_settings()
        elif choice == "6":
            self._display_all_settings()
        elif choice == "7":
            self._reset_to_defaults()
        elif choice == "8":
            return
        else:
            print("❌ Geçersiz seçim.")
    
    def _update_simulation_settings(self):
        """Simülasyon ayarlarını günceller."""
        print("\n--- Simülasyon Ayarları ---")
        sim_settings = self.config['simulation_settings']
        
        print(f"Mevcut Random Seed: {sim_settings['random_seed']}")
        new_seed = input("Yeni Random Seed (boş=değiştirme): ").strip()
        if new_seed:
            sim_settings['random_seed'] = int(new_seed)
        
        print(f"Mevcut Gürültü Seviyesi: {sim_settings['noise_level']}")
        new_noise = input("Yeni Gürültü Seviyesi (boş=değiştirme): ").strip()
        if new_noise:
            sim_settings['noise_level'] = float(new_noise)
        
        print(f"Mevcut Minimum Kayıp: {sim_settings['min_loss']}")
        new_min_loss = input("Yeni Minimum Kayıp (boş=değiştirme): ").strip()
        if new_min_loss:
            sim_settings['min_loss'] = float(new_min_loss)
        
        print(f"Mevcut Maksimum Doğruluk: {sim_settings['max_accuracy']}")
        new_max_acc = input("Yeni Maksimum Doğruluk (boş=değiştirme): ").strip()
        if new_max_acc:
            sim_settings['max_accuracy'] = float(new_max_acc)
        
        self.save_config()
        print("✓ Simülasyon ayarları güncellendi.")
    
    def _update_default_model_parameters(self):
        """Varsayılan model parametrelerini günceller."""
        print("\n--- Varsayılan Model Parametreleri ---")
        model_params = self.config['default_model_parameters']
        
        print(f"Mevcut Varsayılan Epoch: {model_params['total_epochs']}")
        new_epochs = input("Yeni Varsayılan Epoch (boş=değiştirme): ").strip()
        if new_epochs:
            model_params['total_epochs'] = int(new_epochs)
        
        print(f"Mevcut Öğrenme Hızı Etkisi: {model_params['learning_rate_effect']}")
        new_lr = input("Yeni Öğrenme Hızı Etkisi (boş=değiştirme): ").strip()
        if new_lr:
            model_params['learning_rate_effect'] = float(new_lr)
        
        print(f"Mevcut Overfit Başlangıç Epochu: {model_params['overfit_start_epoch']}")
        new_overfit = input("Yeni Overfit Başlangıç Epochu (boş=değiştirme): ").strip()
        if new_overfit:
            model_params['overfit_start_epoch'] = int(new_overfit)
        
        print(f"Mevcut Overfit Eğimi: {model_params['overfit_slope']}")
        new_slope = input("Yeni Overfit Eğimi (boş=değiştirme): ").strip()
        if new_slope:
            model_params['overfit_slope'] = float(new_slope)
        
        print(f"Mevcut Hedef Final Doğruluk: {model_params['target_final_accuracy']}")
        new_target = input("Yeni Hedef Final Doğruluk (boş=değiştirme): ").strip()
        if new_target:
            model_params['target_final_accuracy'] = float(new_target)
        
        self.save_config()
        print("✓ Varsayılan model parametreleri güncellendi.")
    
    def _update_graphics_settings(self):
        """Grafik ayarlarını günceller."""
        print("\n--- Grafik Ayarları ---")
        gfx_settings = self.config['graphics_settings']
        
        print(f"Mevcut Çıktı Dizini: {gfx_settings['output_directory']}")
        new_dir = input("Yeni Çıktı Dizini (boş=değiştirme): ").strip()
        if new_dir:
            gfx_settings['output_directory'] = new_dir
        
        print(f"Mevcut DPI: {gfx_settings['dpi']}")
        new_dpi = input("Yeni DPI (boş=değiştirme): ").strip()
        if new_dpi:
            gfx_settings['dpi'] = int(new_dpi)
        
        print(f"Mevcut Grafik Boyutu: {gfx_settings['figure_size']}")
        new_size = input("Yeni Grafik Boyutu [genişlik,yükseklik] (boş=değiştirme): ").strip()
        if new_size:
            width, height = map(int, new_size.strip('[]').split(','))
            gfx_settings['figure_size'] = [width, height]
        
        self.save_config()
        print("✓ Grafik ayarları güncellendi.")
    
    def _update_metrics_settings(self):
        """Metrik hesaplama ayarlarını günceller."""
        print("\n--- Metrik Hesaplama Ayarları ---")
        print("Bu ayarlar performans metriklerinin nasıl hesaplandığını etkiler.")
        print("(Detaylı ayarlar için config.json dosyasını manuel olarak düzenleyin)")
        
        metrics_settings = self.config['metrics_settings']
        
        print(f"Model Bias Aralığı: {metrics_settings['model_bias_range']}")
        print(f"External Validation Aralığı: {metrics_settings['external_val_range']}")
        print(f"Test Accuracy Aralığı: {metrics_settings['test_acc_range']}")
        print(f"ROC-AUC Aralığı: {metrics_settings['roc_auc_range']}")
        
        print("\n⚠️  Bu ayarlar gelişmiş kullanıcılar içindir.")
        print("Değiştirmek istiyorsanız config.json dosyasını düzenleyin.")
    
    def _update_display_settings(self):
        """Görüntüleme ayarlarını günceller."""
        print("\n--- Görüntüleme Ayarları ---")
        display_settings = self.config['display_settings']
        
        print(f"Mevcut Ondalık Basamaklar: {display_settings['decimal_places']}")
        new_decimal = input("Yeni Ondalık Basamak Sayısı (boş=değiştirme): ").strip()
        if new_decimal:
            display_settings['decimal_places'] = int(new_decimal)
        
        print(f"Mevcut Max Model Adı Uzunluğu: {display_settings['max_model_name_length']}")
        new_length = input("Yeni Max Model Adı Uzunluğu (boş=değiştirme): ").strip()
        if new_length:
            display_settings['max_model_name_length'] = int(new_length)
        
        print(f"Mevcut Tablo Genişliği: {display_settings['table_width']}")
        new_width = input("Yeni Tablo Genişliği (boş=değiştirme): ").strip()
        if new_width:
            display_settings['table_width'] = int(new_width)
        
        self.save_config()
        print("✓ Görüntüleme ayarları güncellendi.")
    
    def _display_all_settings(self):
        """Tüm ayarları görüntüler."""
        print("\n" + "="*60)
        print("                 TÜM AYARLAR")
        print("="*60)
        
        print("\n1. SİMÜLASYON AYARLARI:")
        sim = self.config['simulation_settings']
        print(f"   • Random Seed: {sim['random_seed']}")
        print(f"   • Gürültü Seviyesi: {sim['noise_level']}")
        print(f"   • Minimum Kayıp: {sim['min_loss']}")
        print(f"   • Maksimum Doğruluk: {sim['max_accuracy']}")
        
        print("\n2. VARSAYILAN MODEL PARAMETRELERİ:")
        model = self.config['default_model_parameters']
        print(f"   • Varsayılan Epoch: {model['total_epochs']}")
        print(f"   • Öğrenme Hızı Etkisi: {model['learning_rate_effect']}")
        print(f"   • Overfit Başlangıç Epochu: {model['overfit_start_epoch']}")
        print(f"   • Overfit Eğimi: {model['overfit_slope']}")
        print(f"   • Hedef Final Doğruluk: {model['target_final_accuracy']}")
        
        print("\n3. GRAFİK AYARLARI:")
        gfx = self.config['graphics_settings']
        print(f"   • Çıktı Dizini: {gfx['output_directory']}")
        print(f"   • DPI: {gfx['dpi']}")
        print(f"   • Grafik Boyutu: {gfx['figure_size']}")
        print(f"   • Detaylı Grafik Boyutu: {gfx['detailed_figure_size']}")
        
        print("\n4. METRİK HESAPLAMA AYARLARI:")
        metrics = self.config['metrics_settings']
        print(f"   • Model Bias Aralığı: {metrics['model_bias_range']}")
        print(f"   • External Val Aralığı: {metrics['external_val_range']}")
        print(f"   • Test Acc Aralığı: {metrics['test_acc_range']}")
        print(f"   • ROC-AUC Aralığı: {metrics['roc_auc_range']}")
        
        print("\n5. GÖRÜNTÜLEME AYARLARI:")
        display = self.config['display_settings']
        print(f"   • Ondalık Basamaklar: {display['decimal_places']}")
        print(f"   • Max Model Adı Uzunluğu: {display['max_model_name_length']}")
        print(f"   • Tablo Genişliği: {display['table_width']}")
        
        print("="*60)
    
    def _reset_to_defaults(self):
        """Ayarları varsayılana sıfırlar."""
        confirm = input("\n⚠️  Tüm ayarları varsayılana sıfırlamak istediğinizden emin misiniz? (e/h): ").strip().lower()
        if confirm == 'e':
            self.config = self._get_default_config()
            self.save_config()
            print("✓ Tüm ayarlar varsayılana sıfırlandı.")
        else:
            print("İşlem iptal edildi.")
    
def main():
    """
    Ana uygulama döngüsünü çalıştıran ve kullanıcı etkileşimini yöneten giriş noktası.
    """
    # ModelSimulator nesnesi oluştur
    simulator = ModelSimulator()
    
    print("🤖 Makine Öğrenmesi Model Simülasyon Terminali'ne Hoş Geldiniz!")
    
    # Ana uygulama döngüsü
    while True:
        print("\n" + "="*50)
        print("      MODEL SİMÜLASYON TERMİNALİ")
        print("="*50)
        print("1. Yeni Tekil Model Simülasyonu Yap")
        print("2. Toplu Model Simülasyonu Yap (Manuel/CSV)")
        print("3. Sonuç Tablosunu Görüntüle") 
        print("4. Sonuçları CSV Olarak Kaydet")
        print("5. Model Simülasyonunu Görselleştir")
        print("6. Grafikleri Kaydet")
        print("7. Karşılaştırmalı Matris & Isı Haritaları")
        print("8. Ayarları Yönet")
        print("9. Çıkış")
        print("-"*50)
        
        choice = input("Seçiminiz (1-9): ").strip()
        
        if choice == "1":
            try:
                simulator.run_new_simulation()
            except KeyboardInterrupt:
                print("\n⚠️  Simülasyon iptal edildi.")
            except Exception as e:
                print(f"❌ Simülasyon hatası: {e}")
                
        elif choice == "2":
            try:
                simulator.run_batch_simulation()
            except KeyboardInterrupt:
                print("\n⚠️  Toplu simülasyon iptal edildi.")
            except Exception as e:
                print(f"❌ Toplu simülasyon hatası: {e}")
                
        elif choice == "3":
            simulator.display_results_table()
            
        elif choice == "4":
            simulator.save_to_csv()
            
        elif choice == "5":
            try:
                simulator.visualize_models()
            except KeyboardInterrupt:
                print("\n⚠️  Görselleştirme iptal edildi.")
            except Exception as e:
                print(f"❌ Görselleştirme hatası: {e}")
                
        elif choice == "6":
            try:
                simulator.save_graphs()
            except KeyboardInterrupt:
                print("\n⚠️  Grafik kaydetme iptal edildi.")
            except Exception as e:
                print(f"❌ Grafik kaydetme hatası: {e}")
        
        elif choice == "7":
            try:
                simulator.visualize_matrix_comparisons()
            except KeyboardInterrupt:
                print("\n⚠️  Matris görselleştirme iptal edildi.")
            except Exception as e:
                print(f"❌ Matris görselleştirme hatası: {e}")
        
        elif choice == "8":
            try:
                simulator.update_config_settings()
            except KeyboardInterrupt:
                print("\n⚠️  Ayar güncellemesi iptal edildi.")
            except Exception as e:
                print(f"❌ Ayar güncellemesi hatası: {e}")
            
        elif choice == "9":
            print("\n👋 Çıkış yapılıyor. İyi günler!")
            break
            
        else:
            print("\n❌ Geçersiz seçim, lütfen tekrar deneyin.")
        
        # Devam etmek için kullanıcıdan onay al
        if choice in ["1", "2", "3", "4", "5", "6", "7", "8"]:
            input("\nDevam etmek için Enter'a basın...")


if __name__ == "__main__":
    main()
