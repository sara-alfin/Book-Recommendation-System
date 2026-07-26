# Book Recommendation System

Book-Crossing veri seti üzerinde, **sıfırdan yazılmış NMF (Non-negative
Matrix Factorization)** algoritması ile çalışan bir kitap/yazar öneri
sistemi. Projede scikit-learn veya scipy'nin hazır NMF fonksiyonu
**kullanılmamıştır** — çarpımsal güncelleme (multiplicative update) kuralı
ve NNDSVD başlatma yöntemi `nmf.py` içinde kendi implementasyonum olarak
yazılmıştır.

## Proje Yapısı

```
.
├── nmf.py              # Kendi yazdığım NMF implementasyonu
├── recommend.py         # Veri hazırlama + öneri üretme (kitap ve yazar modu)
├── veri/
│   ├── Books.csv
│   ├── Ratings.csv
│   └── Users.csv
└── README.md
```



## Kurulum

```bash
pip install numpy pandas
```

## Kullanım

```bash
# Kitap başlığına göre öneri (varsayılan)
python recommend.py --user-id 178667 --mode book

# Yazara göre öneri
python recommend.py --user-id 178667 --mode author --init random

# Farklı bir rank ile
python recommend.py --user-id 178667 --mode book --rank 20
```

## Yöntem

1. `Books.csv`, `Ratings.csv`, `Users.csv` birleştirilir.
2. En az 250 kitaba oy vermiş kullanıcılar ve en az 50 oy almış
   kitaplar/yazarlar filtrelenir.
3. Kullanıcı × kitap (veya yazar) puan matrisi (**V**) oluşturulur.
4. `nmf.py` içindeki çarpımsal güncelleme kuralı ile **V ≈ W · H**
   ayrıştırması hesaplanır (NNDSVD veya rastgele başlatma seçilebilir).
5. Yeniden oluşturulan matristen (**V̂ = W·H**), kullanıcının henüz
   puanlamadığı en yüksek skorlu 10 kitap/yazar önerilir.

## Kaynaklar / References

- Book-Crossing Dataset. Cai-Nicolas Ziegler. Erişim:
  https://www2.informatik.uni-freiburg.de/~cziegler/BX/
- Lee, D. D., & Seung, H. S. (2001). *Algorithms for Non-negative Matrix
  Factorization*. Advances in Neural Information Processing Systems (NIPS).
- Boutsidis, C., & Gallopoulos, E. (2008). *SVD based initialization: A
  head start for nonnegative matrix factorization*. Pattern Recognition,
  41(4), 1350–1362.
- Abdelrahman M. (2024). *Book Recommendation System*. Medium.
  https://medium.com/@abdelrahman.m2922/book-recommendation-system-fa510e2d5a24
- Xaradxarma. (2023). *Book Recommendation System*. Medium.
  https://medium.com/@xaradxarma/book-recommendation-system-8cdb77585b65
- GitHub Repository: [CEPHAL0/Book-Recommendation-System](https://github.com/CEPHAL0/Book-Recommendation-System)

> Not: `nmf.py` içindeki NMF algoritması bu projede **kendim tarafından
> yazılmıştır**; yukarıdaki Lee & Seung (2001) ve Boutsidis & Gallopoulos
> (2008) referansları yalnızca uyguladığım yöntemlerin matematiksel
> temelini gösterir, kod bu kaynaklardan kopyalanmamıştır.
