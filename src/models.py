import numpy as np

from data import TARGETS
from utils import weighted_r2


# --------------------------------------------------------------------------
# B0: her hedef için eğitim ortalaması
# --------------------------------------------------------------------------
class MeanBaseline:
    """Görüntüye bakmadan her hedef için eğitim ortalamasını (gram) tahmin eder."""

    def fit(self, X, Y):
        self.mean_ = np.asarray(Y, dtype=float).mean(axis=0)
        return self

    def predict(self, X):
        return np.tile(self.mean_, (len(X), 1))


# --------------------------------------------------------------------------
# Çapraz doğrulama
# --------------------------------------------------------------------------
def cross_validate(make_model, X, df, fold_col="fold", targets=TARGETS):
    """Hazır fold kolonuyla CV. (oof_tahmin[n,5], genel_skor, fold_skorları) döner.

    make_model: her fold için yeni model üreten fonksiyon (fit/predict).
    Skor: tüm OOF tahminleri üzerinde global ağırlıklı R² (+ fold bazında).
    """
    X = np.asarray(X, dtype=float)
    Y = df[targets].to_numpy(dtype=float)
    oof = np.zeros_like(Y)
    scores = []
    for k in sorted(df[fold_col].unique()):
        va = (df[fold_col] == k).to_numpy()
        model = make_model().fit(X[~va], Y[~va])
        oof[va] = model.predict(X[va])
        scores.append(weighted_r2(Y[va], oof[va], targets))
    return oof, weighted_r2(Y, oof, targets), scores


def per_target_r2(Y, P, targets=TARGETS):
    """Hedef bazında sıradan R² (teşhis amaçlı, Kaggle skoru değil)."""
    Y, P = np.asarray(Y, float), np.asarray(P, float)
    return {t: 1 - ((Y[:, i] - P[:, i]) ** 2).sum() / ((Y[:, i] - Y[:, i].mean()) ** 2).sum()
            for i, t in enumerate(targets)}
