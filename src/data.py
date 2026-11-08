import pandas as pd

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


