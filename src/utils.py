import numpy as np

# Kaggle metriğindeki hedef ağırlıkları
WEIGHTS = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
}


def weighted_r2(y_true, y_pred, targets=None):
    """Global ağırlıklı R² (Kaggle tanımı).

    Tüm (görüntü, hedef) satırları tek havuzda, gram cinsinden değerlendirilir.
    Her satırın ağırlığı hedefinin ağırlığıdır:
        ybar = sum(w * y) / sum(w)
        R2   = 1 - sum(w * (y - p)^2) / sum(w * (y - ybar)^2)

    y_true, y_pred: (n_görüntü, n_hedef) dizi/DataFrame (gram cinsinden).
    targets: sütun sırası. DataFrame verilirse sütun adlarından alınır.
    """
    if targets is None:
        targets = list(y_true.columns) if hasattr(y_true, "columns") else list(WEIGHTS)
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    assert yt.shape == yp.shape and yt.shape[1] == len(targets), "şekiller uyumsuz"

    w = np.tile([WEIGHTS[t] for t in targets], (yt.shape[0], 1))
    ybar = (w * yt).sum() / w.sum()
    ss_res = (w * (yt - yp) ** 2).sum()
    ss_tot = (w * (yt - ybar) ** 2).sum()
    return 1.0 - ss_res / ss_tot
