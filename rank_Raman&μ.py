from xgboost import XGBRanker
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score
from joblib import dump
from typing import Tuple

TOP_ALPHA = 0.10
N_SPLITS  = 5
RANDOM_STATE = 42

LGBM_PARAMS = dict(
    boosting_type="gbdt",
    objective="lambdarank",
    metric="ndcg",
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=600,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    min_data_in_leaf=20,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
)

XGB_PARAMS = dict(
    objective="rank:ndcg",
    learning_rate=0.05,
    n_estimators=800,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    tree_method="hist",
)

FEATURES = [
    "E_intensity","E_shift","E_fwhm","E_area","E_shift_offset","E_fwhm_offset",
    "A_intensity","A_shift","A_fwhm","A_area","A_shift_offset","A_fwhm_offset",
    "width_ratio","shift_ratio","intensity_ratio","area_ratio",
    "shift_diff","width_diff","intensity_diff","area_diff",
    "width_sum","shift_sum","intensity_sum","area_sum",
    "mix_Ashift_Ewidth","mix_Eshift_Awidth","mix_Aintensity_Earea","mix_Eintensity_Aarea",
    "log_E_fwhm","log_A_fwhm","log_width_sum",
    "log_E_intensity","log_A_intensity","log_intensity_sum",
    "log_E_area","log_A_area","log_area_sum",
]

def load_dataset() -> pd.DataFrame:
    # Data loading logic here
    # Assume dataset is loaded as df
    df = pd.DataFrame()  # Placeholder

    # Add relevant features to the dataframe here
    df.attrs["FEATURES"] = FEATURES  # Assigning FEATURES to df.attrs
    return df

def train_rank_and_calibrate(df: pd.DataFrame):
    FEATURES = df.attrs["FEATURES"]
    X_raw = df[FEATURES].values.astype(float)
    y = df["mobility"].values.astype(float)

    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)
    y_rel = to_rel_grades(y, 10)

    groups = make_spatial_groups(df, (5, 5))
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_lgbm = np.zeros(len(y))
    oof_xgb  = np.zeros(len(y))

    for fold, (tr, va) in enumerate(gkf.split(X, y_rel, groups=groups), start=1):
        Xtr, Xva = X[tr], X[va]
        ytr_rel, yva_rel = y_rel[tr], y_rel[va]

        lgbm = lgb.LGBMRanker(**LGBM_PARAMS)
        lgbm.fit(Xtr, ytr_rel, group=np.array([len(tr)], dtype=int))
        oof_lgbm[va] = lgbm.predict(Xva)

        xgb = XGBRanker(**XGB_PARAMS)
        xgb.fit(Xtr, ytr_rel, qid=groups[tr].astype(np.uint32))
        oof_xgb[va] = xgb.predict(Xva)

    best_w = 0.5
    score_final = best_w * oof_lgbm + (1 - best_w) * oof_xgb
    dump({"scaler": scaler, "lgbm": lgbm, "xgb": xgb, "blend_w": best_w}, "rank_models_and_scaler.joblib")
    return {"score_final": score_final}

def to_rel_grades(y: np.ndarray, grades: int = 10) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    bins = np.linspace(y.min(), y.max(), grades + 1)
    return np.digitize(y, bins[1:-1], right=True).astype(int)

def make_spatial_groups(df: pd.DataFrame, blocks: Tuple[int, int] = (5, 5)) -> np.ndarray:
    rows = df["row"].astype(int).values
    cols = df["col"].astype(int).values
    R = int(rows.max()) + 1
    C = int(cols.max()) + 1
    br, bc = blocks
    br_edges = np.linspace(0, R, br + 1).astype(int)
    bc_edges = np.linspace(0, C, bc + 1).astype(int)
    gr = np.digitize(rows, br_edges[1:-1], right=False)
    gc = np.digitize(cols, bc_edges[1:-1], right=False)
    return (gr * bc + gc).astype(int)

def main():
    np.random.seed(RANDOM_STATE)
    df = load_dataset()
    res = train_rank_and_calibrate(df)

    # Save the results, models, and scores
    pd.DataFrame({"score_final": res["score_final"]}).to_csv("rank_scores.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
