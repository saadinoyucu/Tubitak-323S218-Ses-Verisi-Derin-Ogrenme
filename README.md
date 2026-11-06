# Deprem Sonrası Psikolojik Etkilerin Yapay Zekâ ile Değerlendirilmesi

Bu depo, **TÜBİTAK 1001 – 323S218** kodlu *“Deprem Sonrası Psikolojik Etkilerin Yapay Zekâ ile Değerlendirilmesi: Anksiyete, Depresyon ve Travma Sonrası Stres Bozukluğu Tespiti”* başlıklı proje kapsamında geliştirilen ses sınıflandırma kodlarını içerir.

## Depo Kapsamı

Bu depo yalnızca kaynak kod ve model mimarisi çizimlerini içerir. Katılımcı verileri, ses kayıtları, öznitelik CSV dosyaları, eğitilmiş model ağırlıkları ve deney sonuçları gizlilik, etik ve veri yönetimi gereklilikleri nedeniyle paylaşılmamaktadır.

Kodlar, ses verilerinden elde edilen özniteliklerle `kontrol` ve `hasta` sınıflarını ayırt etmeye yönelik eğitim, değerlendirme ve raporlama iş akışını sağlar. Proje bulguları klinik değerlendirme veya tanı yerine geçmez.

## Kullanılan Yaklaşımlar

- Öznitelik temsilleri: `MFCC`, `HuBERT`, `Wav2Vec`, `google-HeAR`
- Derin öğrenme mimarileri: `TCN`, `CNN_LSTM`, `TCN_LSTM`
- Değerlendirme metrikleri: doğruluk, F1 skoru, kesinlik, duyarlılık ve ROC-AUC

## Kod Yapısı

- `main_train.py`: Eğitim, test ve raporlama akışının ana giriş noktasıdır.
- `dataset_utils.py`: CSV tabanlı veri yükleme, ölçekleme ve veri hazırlama işlemlerini yürütür.
- `train_utils.py`: Eğitim döngüsü, kayıp fonksiyonu ve model ağırlıklarının kaydedilmesini yönetir.
- `eval_utils.py`: Test metriklerini hesaplar; karışıklık matrisi ve ROC eğrisi üretir.
- `report_utils.py`, `plot_utils.py`: Deney sonuçlarını özetler ve görselleştirir.
- `model_tcn.py`, `model_cnn.py`, `model_tcn_lstm.py`: Kullanılan model mimarilerini tanımlar.
- `generate_all_graphs.py`, `generate_new_graphs.py`: Deney çıktılarından toplu grafikler üretir.
- `project_paths.py`: Proje içi yol ve adlandırma standartlarını yönetir.
- `Model Mimarileri Grafik/`: Model mimarilerinin `.drawio` kaynakları ve PNG görselleridir.

## Çalıştırma

Kod, proje kapsamında uygun biçimde hazırlanmış ve paylaşılmayan öznitelik CSV dosyalarını bekler. Gerekli veri yapısı sağlandıktan sonra örnek eğitim komutu:

```bash
python main_train.py --feature_type MFCC
```

Desteklenen öznitelik türleri: `MFCC`, `HuBERT`, `Wav2Vec`, `google-HeAR` ve `all`.

## Başlıca Bağımlılıklar

`torch`, `torchaudio`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `librosa`, `soundfile`, `transformers` ve `tqdm`.

## Veri ve Çıktı Politikası

Bu depoya ham/işlenmiş veri, kişisel veri, model ağırlığı veya deney çıktısı eklenmemelidir. Kodun çalışması sırasında oluşan CSV, grafik, rapor ve model dosyaları sürüm kontrolü dışında tutulmalıdır.

## Proje Bilgisi

- Destek programı: TÜBİTAK 1001
- Proje kodu: 323S218
- Proje adı: Deprem Sonrası Psikolojik Etkilerin Yapay Zekâ ile Değerlendirilmesi: Anksiyete, Depresyon ve Travma Sonrası Stres Bozukluğu Tespiti
