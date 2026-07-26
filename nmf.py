"""
Non-negative Matrix Factorization (NMF) — kendi implementasyonum.

Bu modül, kitap/yazar öneri sistemi için kullanılan NMF algoritmasının
sıfırdan (scipy/sklearn'ün hazır NMF fonksiyonu kullanılmadan) yazılmış
halidir. Çarpımsal güncelleme (multiplicative update) kuralı ve NNDSVD
başlatma yöntemi uygulanmıştır.

Referans (yalnızca algoritmanın matematiksel temeli için):
  Lee, D. D., & Seung, H. S. (2001). Algorithms for Non-negative Matrix
  Factorization. Advances in Neural Information Processing Systems.

  Boutsidis, C., & Gallopoulos, E. (2008). SVD based initialization:
  A head start for nonnegative matrix factorization. Pattern Recognition.
"""

import random

import numpy as np


def choose_rank(matrix: np.ndarray) -> int:
    """V matrisi için rastgele fakat (m*n)/(m+n) sınırını aşmayan bir rank seçer."""
    m, n = matrix.shape
    upper_bound = (m * n) / (m + n)
    while True:
        r = random.randint(1, min(m, n))
        if r < upper_bound:
            return r


def random_init(V: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """W ve H'yi V'nin pozitif değer aralığında rastgele başlatır."""
    n, m = V.shape
    positive_values = V[V > 0]
    low, high = positive_values.min(), positive_values.max()
    W = np.random.uniform(low, high, (n, r))
    H = np.random.uniform(low, high, (r, m))
    return W, H


def nndsvd_init(V: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """
    NNDSVD (Non-Negative Double Singular Value Decomposition) başlatma.
    SVD bileşenlerinin pozitif/negatif kısımlarına göre W ve H'yi kurar.
    """
    u, s, vt = np.linalg.svd(V, full_matrices=False)
    v = vt.T

    W = np.zeros((V.shape[0], r))
    H = np.zeros((r, V.shape[1]))

    W[:, 0] = np.sqrt(s[0]) * np.abs(u[:, 0])
    H[0, :] = np.sqrt(s[0]) * np.abs(v[:, 0])

    for i in range(1, r):
        ui, vi = u[:, i], v[:, i]

        ui_pos, ui_neg = np.clip(ui, 0, None), np.clip(-ui, 0, None)
        vi_pos, vi_neg = np.clip(vi, 0, None), np.clip(-vi, 0, None)

        ui_pos_norm, ui_neg_norm = np.linalg.norm(ui_pos), np.linalg.norm(ui_neg)
        vi_pos_norm, vi_neg_norm = np.linalg.norm(vi_pos), np.linalg.norm(vi_neg)

        norm_pos = ui_pos_norm * vi_pos_norm
        norm_neg = ui_neg_norm * vi_neg_norm

        if norm_pos >= norm_neg:
            scale = np.sqrt(s[i] * norm_pos)
            W[:, i] = scale / max(ui_pos_norm, 1e-12) * ui_pos
            H[i, :] = scale / max(vi_pos_norm, 1e-12) * vi_pos
        else:
            scale = np.sqrt(s[i] * norm_neg)
            W[:, i] = scale / max(ui_neg_norm, 1e-12) * ui_neg
            H[i, :] = scale / max(vi_neg_norm, 1e-12) * vi_neg

    return W, H


def multiplicative_update(
    df,
    max_iter: int = 2000,
    init_mode: str = "nndsvd",
    rank: int | None = None,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lee & Seung (2001) çarpımsal güncelleme kuralına göre V ≈ W @ H
    ayrıştırmasını hesaplar.

    Parametreler
    ------------
    df : pandas.DataFrame
        Faktörize edilecek (kullanıcı x kitap/yazar) puan matrisi. NaN
        değerler çağıran taraftan 0 ile doldurulmuş olmalıdır.
    max_iter : int
        Azami iterasyon sayısı (yakınsama olmasa dahi burada durur).
    init_mode : {"random", "nndsvd"}
        Başlatma yöntemi.
    rank : int, optional
        Faktörizasyon rankı (r). Verilmezse rastgele seçilir ve
        `rank.txt` dosyasına kaydedilir; sonraki çalıştırmalarda oradan
        okunur.
    tol : float
        H ve W'deki değişim bu değerin altına inince iterasyon durur.

    Döndürür
    --------
    (W, H) : tuple[np.ndarray, np.ndarray]
    """
    V = df.to_numpy(dtype=np.float64)

    if rank is None:
        try:
            with open("rank.txt", "r") as f:
                rank = int(f.readline().strip())
        except (FileNotFoundError, ValueError):
            rank = choose_rank(V)
            with open("rank.txt", "w") as f:
                f.write(str(rank))

    if init_mode == "random":
        W, H = random_init(V, rank)
    elif init_mode == "nndsvd":
        W, H = nndsvd_init(V, rank)
    else:
        raise ValueError(f"Bilinmeyen init_mode: {init_mode!r}")

    epsilon = 1e-9  # sadece sıfıra bölünmeyi önlemek için
    for _ in range(max_iter):
        H_next = H * (W.T @ V) / (W.T @ W @ H + epsilon)
        W_next = W * (V @ H_next.T) / (W @ H_next @ H_next.T + epsilon)

        delta = np.linalg.norm(H - H_next) + np.linalg.norm(W - W_next)
        H, W = H_next, W_next

        if delta < tol:
            break

    return W, H
