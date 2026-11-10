import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import TARGETS
from utils import weighted_r2

# Bağımsız 3 hedef. GDM = Green + Clover, Total = Green + Dead + Clover
BASE_TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g"]
BASE_IDX = [TARGETS.index(t) for t in BASE_TARGETS]


def derive_from_base(base):
    """(n, 3) [Green, Dead, Clover] -> (n, 5) TARGETS sırasında, GDM ve Total toplamdan."""
    base = np.asarray(base, dtype=float)
    g, d, c = base[:, 0], base[:, 1], base[:, 2]
    out = {"Dry_Green_g": g, "Dry_Dead_g": d, "Dry_Clover_g": c,
           "GDM_g": g + c, "Dry_Total_g": g + d + c}
    return np.column_stack([out[t] for t in TARGETS])


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
    Tahmin expm1 ile grama döner, negatifler 0'a kırpılır.

    derive=True: sadece Green, Dead, Clover öğrenilir; GDM ve Total toplamdan türetilir.
    Bu durumda da fit 5 hedefli Y alır, predict 5 kolon (TARGETS sırasında) döner.
    log=False: log1p dönüşümü yapılmaz, doğrudan gram ölçeğinde öğrenilir.
    """

    def __init__(self, alpha=1.0, derive=False, log=True):
        self.alpha = alpha
        self.derive = derive
        self.log = log

    def fit(self, X, Y):
        Y = np.asarray(Y, dtype=float)
        if self.derive:
            Y = Y[:, BASE_IDX]
        if self.log:
            Y = np.log1p(Y)
        self.model_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha)).fit(X, Y)
        return self

    def predict(self, X):
        p = self.model_.predict(X)
        if self.log:
            p = np.expm1(p)
        p = np.clip(p, 0, None)
        return derive_from_base(p) if self.derive else p


# --------------------------------------------------------------------------
# B3: pretrained CNN özellikleri + Ridge
# --------------------------------------------------------------------------
def extract_cnn_features(df, root="data", batch_size=16, tile=448, device=None):
    """ImageNet ön-eğitimli ResNet50 ile görüntü başına 2048 boyutlu özellik (fine-tune yok).

    2000x1000 görüntü sol/sağ iki 1000x1000 kareye bölünür (EDA: küçültürken detay kaybolmasın),
    her yarı tile x tile'a küçültülür, iki yarının özellikleri ortalanır.
    torch/torchvision sadece bu fonksiyon çağrılınca import edilir.
    """
    import torch
    import torchvision
    from torchvision import transforms as T

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    net = torchvision.models.resnet50(weights=weights)
    net.fc = torch.nn.Identity()
    net = net.eval().to(device)
    prep = T.Compose([T.Resize((tile, tile)), T.ToTensor(),
                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    feats, paths = [], list(df["image_path"])
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            halves = []
            for p in paths[i:i + batch_size]:
                im = Image.open(f"{root}/{p}").convert("RGB")
                w, h = im.size
                halves += [prep(im.crop((0, 0, w // 2, h))), prep(im.crop((w // 2, 0, w, h)))]
            out = net(torch.stack(halves).to(device)).cpu().numpy()
            feats.append(out.reshape(-1, 2, out.shape[-1]).mean(1))
    return pd.DataFrame(np.vstack(feats), index=df.index)


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
