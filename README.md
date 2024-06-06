Duygu Tanıma Uygulaması

Bu uygulama duygu tanıma üzerine bir örnek sunar. Verileri kullanarak metin tabanlı duygu tanıma modeli oluşturur ve ardından bu modeli bir FastAPI uygulaması aracılığıyla kullanılabilir hale getirir.

Gereksinimler

- Python 3.x

Kurulum

- git clone https://github.com/Mustimain/AIfinal
- cd <repository>

Kullanım

1-) WordNet Kütüphanesi İndirme

    #import nltk
    #nltk.download('wordnet')

2-) Veri dosyalarını path hazırlama:

 - Eğitim verisi: emotionsdata_train.csv
 - Test verisi: emotionsdata_test.csv

3-) Uygulamayı başlatın:
 
 - python main.py


Tarayıcınızdan http://127.0.0.1:8000/ adresine giderek uygulamayı kullanabilirsiniz.

API Uç Noktaları
 - /predict/logistic_regression: Lojistik Regresyon modeli ile tahmin yapar.
 - /predict/decision_tree: Karar Ağacı modeli ile tahmin yapar.
 - /predict/svm: Destek Vektör Makineleri (SVM) modeli ile tahmin yapar.
 - /predict/random_forest: Rastgele Orman modeli ile tahmin yapar.
