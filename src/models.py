import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
# B2: renk / örtü özellikleri + Ridge
# --------------------------------------------------------------------------
COLOR_FEATURE_NAMES = [
    "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std",
    "yesillik", "exg", "ortu", "h_mean", "s_mean", "v_mean", "h_std", "s_std", "v_std",
]


def color_features(path, size=(200, 100)):
    """Tek görüntüden renk ve örtü özellikleri (EDA'daki tanımlarla aynı)."""
    im = Image.open(path).convert("RGB").resize(size)
    rgb = np.asarray(im, dtype=float) / 255
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    toplam = r.mean() + g.mean() + b.mean() + 1e-9
    yesillik = g.mean() / toplam
    exg = (2 * g - r - b).mean()                 # excess green
    ortu = ((g > r) & (g > b)).mean()            # yeşil piksel oranı
    hsv = np.asarray(im.convert("HSV"), dtype=float).reshape(-1, 3) / 255
    return [r.mean(), g.mean(), b.mean(), r.std(), g.std(), b.std(),
            yesillik, exg, ortu, *hsv.mean(0), *hsv.std(0)]


def extract_color_features(df, root="data"):
    return pd.DataFrame([color_features(f"{root}/{p}") for p in df["image_path"]],
                        columns=COLOR_FEATURE_NAMES, index=df.index)


class LogRidge:
    """log1p ölçeğinde StandardScaler + Ridge.
    Tahmin expm1 ile grama döner, negatifler 0'a kırpılır."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, Y):
        Y = np.asarray(Y, dtype=float)
        self.model_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha)).fit(X, np.log1p(Y))
        return self

    def predict(self, X):
        return np.clip(np.expm1(self.model_.predict(X)), 0, None)


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
