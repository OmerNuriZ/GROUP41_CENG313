import pandas as pd
from sklearn.model_selection import GroupKFold

TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]


def load_train(path="data/train.csv"):
    """train.csv'yi okur, her image tek satır olacak şekilde çevirir."""
    df = pd.read_csv(path)

    # 5 target'ı kolonlara çevir
    targets = df.pivot(index="image_path", columns="target_name", values="target")
    targets = targets[TARGETS].reset_index()

    # metadata her image için aynı, tekrarları at
    meta_cols = ["image_path", "Sampling_Date", "State", "Species",
                 "Pre_GSHH_NDVI", "Height_Ave_cm"]
    meta = df[meta_cols].drop_duplicates("image_path")

    return meta.merge(targets, on="image_path")


def make_groups(df):
    """Grup = State + Sampling_Date. Aynı yer ve günden gelen görüntüler aynı fold'a düşer."""
    return df["State"].astype(str) + "_" + df["Sampling_Date"].astype(str)


def make_folds(df, n_splits=5, save_path="folds.csv"):
    """GroupKFold ile fold atar. df'ye 'fold' kolonu ekler (0..n_splits-1).

    GroupKFold rastgelelik içermez, aynı veri her zaman aynı fold'ları verir.
    save_path verilirse (image_path, fold) dosyaya yazılır.
    """
    df = df.copy()
    groups = make_groups(df)
    df["fold"] = -1
    gkf = GroupKFold(n_splits=n_splits)
    for k, (_, va) in enumerate(gkf.split(df, groups=groups)):
        df.iloc[va, df.columns.get_loc("fold")] = k
    if save_path:
        df[["image_path", "fold"]].to_csv(save_path, index=False)
    return df


def load_folds(df, path="folds.csv"):
    """Kaydedilmiş folds.csv'yi df'ye bağlar."""
    return df.drop(columns=["fold"], errors="ignore").merge(
        pd.read_csv(path), on="image_path", how="left")
