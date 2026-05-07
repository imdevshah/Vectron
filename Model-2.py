"""
Model_Final_v2.py
-----------------
Improvements over v1:

[FEATURE ENGINEERING]
  1. Rolling min/max/range/position (oscillator) on ALL lag_feats, not just 2
  2. Horizon-aligned lag added: shift(horizon) — directly economically meaningful
  3. Multi-lag momentum diffs: feat[t] - feat[t-5], feat[t] - feat[t-10]
  4. EWM spans scaled to horizon: longer spans for h=10,25
  5. Cross-sectional normalization extended to spread features (d_cg_by, d_s_t, d_al_cg)
  6. feature_a interaction: time-to-expiry weighted z-score (options intuition)
  7. Rolling features on ALL lag_feats, not just feature_al/am
  8. min_periods fixed: max(3, w//2) instead of 1 to avoid single-point noise
  9. Rolling median added (robust to outliers vs mean)
  10. Autocorrelation at lag-1 within rolling window (mean-reversion signal)

[MODEL]
  11. More seeds (7 vs 5) — reduces variance on ensemble
  12. Horizon 1 gets stronger regularization (higher L2, fewer leaves)
      since h=1 is highest noise

[VALIDATION]
  13. Dual holdout check: score on 3001-3500 vs 3501+ to detect val instability

Usage:
    python Model_Final_v2.py --train train.parquet --test test.parquet
"""

import argparse
import warnings
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEEDS         = [42, 2024, 12345, 99, 420, 7, 314]   # FIX 11: 7 seeds vs 5
HORIZONS      = [1, 3, 10, 25]
VAL_THRESHOLD = 3500

BASE_PARAMS = {
    "objective"       : "regression",
    "metric"          : "rmse",
    "learning_rate"   : 0.015,
    "n_estimators"    : 4200,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "bagging_freq"    : 5,
    "lambda_l1"       : 0.1,
    "verbosity"       : -1,
    "n_jobs"          : -1,
}

# FIX 12: h=1 gets tighter regularization (most noise, least signal)
HORIZON_PARAMS = {
    1:  {"num_leaves": 60,  "min_child_samples": 300, "lambda_l2": 14.0},  # tighter than v1
    3:  {"num_leaves": 75,  "min_child_samples": 225, "lambda_l2": 11.0},
    10: {"num_leaves": 85,  "min_child_samples": 180, "lambda_l2":  9.0},
    25: {"num_leaves": 90,  "min_child_samples": 150, "lambda_l2":  8.0},
}

def get_params(horizon):
    return {**BASE_PARAMS, **HORIZON_PARAMS[horizon]}

NON_FEATURE_COLS = {
    "id", "code", "sub_code", "sub_category",
    "horizon", "ts_index", "weight", "y_target", "group_id",
}

# ---------------------------------------------------------------------------
# Kaggle metric
# ---------------------------------------------------------------------------

def kaggle_score(y_true, y_pred, weights):
    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    weights = np.array(weights)
    num     = np.sum(weights * (y_true - y_pred) ** 2)
    den     = np.sum(weights * y_true ** 2)
    if den <= 0:
        return 0.0
    return float(np.sqrt(1.0 - np.clip(num / den, 0.0, 1.0)))

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(df, horizon):
    df = df.copy()
    df = df.sort_values(["code", "sub_code", "sub_category",
                         "horizon", "ts_index"]).reset_index(drop=True)

    group_cols    = ["code", "sub_code", "sub_category", "horizon"]

    # ── 1. feature_a derived features ────────────────────────────────────────
    df["feature_a_pct"]     = df["feature_a"] / 250.0
    df["feature_a_exp_max"] = df.groupby(group_cols)["feature_a"].transform(
        lambda x: x.expanding().max()
    )
    df["feature_a_exp_pct"] = df["feature_a"] / (df["feature_a_exp_max"] + 1e-7)
    df["feature_a_diff"]    = df.groupby(group_cols)["feature_a"].transform(
        lambda x: x.diff()
    )

    # ── 2. Expanding means of y_target ───────────────────────────────────────
    for col, grp in [("sub_category", "sub_category"),
                     ("code", "code"),
                     ("sub_code", "sub_code")]:
        df[f"{col}_exp_mean"] = (
            df.groupby(grp)["y_target"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
        df[f"{col}_exp_std"] = (
            df.groupby(grp)["y_target"]
            .transform(lambda x: x.shift(1).expanding().std())
        )

    # ── 3. Spread / interaction features ─────────────────────────────────────
    df["d_al_am"]  = df["feature_al"] - df["feature_am"]
    df["r_al_am"]  = df["feature_al"] / (df["feature_am"].abs() + 1e-7)
    df["d_cg_by"]  = df["feature_cg"] - df["feature_by"]
    df["d_s_t"]    = df["feature_s"]  - df["feature_t"]
    df["d_al_cg"]  = df["feature_al"] - df["feature_cg"]

    # ── 4. Cross-sectional normalisation ─────────────────────────────────────
    # FIX 6: Added d_cg_by, d_s_t, d_al_cg to cs_cols (were missing in v1)
    cs_cols = ["feature_al", "feature_am", "feature_cg",
               "feature_by", "d_al_am", "d_cg_by", "d_s_t", "d_al_cg"]
    for col in cs_cols:
        if col not in df.columns:
            continue
        grp = df.groupby("ts_index")[col]
        df[f"{col}_cs_mean"]  = grp.transform("mean")
        df[f"{col}_cs_std"]   = grp.transform("std")
        df[f"{col}_z"]        = (df[col] - df[f"{col}_cs_mean"]) / (df[f"{col}_cs_std"] + 1e-7)
        df[f"{col}_rank"]     = grp.rank(pct=True)
        df[f"{col}_ts_min"]   = grp.transform("min")
        df[f"{col}_ts_max"]   = grp.transform("max")
        df[f"{col}_dist_min"] = df[col] - df[f"{col}_ts_min"]
        df[f"{col}_dist_max"] = df[f"{col}_ts_max"] - df[col]

    # FIX 7: feature_a time-to-expiry interaction with key z-score
    # Intuition: near expiry (feature_a → 0), cross-sectional rank matters more
    if "feature_al_z" in df.columns:
        df["feature_a_al_z_interact"] = df["feature_a_pct"] * df["feature_al_z"]
    if "feature_cg_z" in df.columns:
        df["feature_a_cg_z_interact"] = df["feature_a_pct"] * df["feature_cg_z"]

    # ── 5. Lags of original features ──────────────────────────────────────────
    lag_feats = ["feature_al", "feature_am", "feature_cg",
                 "feature_by", "feature_s", "feature_t"]

    for feat in lag_feats:
        if feat not in df.columns:
            continue

        # FIX 2: Added horizon-aligned lag — directly meaningful economically
        lag_set = sorted(set([1, 3, 5, 10, horizon]))
        for lag in lag_set:
            df[f"{feat}_lag_{lag}"] = df.groupby(group_cols)[feat].shift(lag)

        # diff_1 (short momentum)
        df[f"{feat}_diff_1"] = df.groupby(group_cols)[feat].diff(1)
        df[f"{feat}_pct_1"]  = df.groupby(group_cols)[feat].pct_change(1)

        # FIX 5: Multi-lag momentum diffs (medium-term)
        df[f"{feat}_diff_5"]  = df.groupby(group_cols)[feat].transform(
            lambda x: x - x.shift(5)
        )
        df[f"{feat}_diff_10"] = df.groupby(group_cols)[feat].transform(
            lambda x: x - x.shift(10)
        )

    # ── 6. Rolling stats — ALL lag_feats, not just 2 ──────────────────────────
    # FIX 1: Expanded to all lag_feats
    # FIX 8: min_periods = max(3, w//2) instead of 1
    # FIX 9: Added rolling median (robust to outliers)
    # FIX 10: Added rolling min/max/range/normalized position (oscillator)
    for feat in lag_feats:
        if feat not in df.columns:
            continue
        for w in [5, 10, 20]:
            mp = max(3, w // 2)   # FIX 8
            grp_series = df.groupby(group_cols)[feat]

            df[f"{feat}_roll_mean_{w}"]   = grp_series.transform(
                lambda x: x.rolling(w, min_periods=mp).mean())
            df[f"{feat}_roll_std_{w}"]    = grp_series.transform(
                lambda x: x.rolling(w, min_periods=mp).std())
            df[f"{feat}_roll_median_{w}"] = grp_series.transform(   # FIX 9
                lambda x: x.rolling(w, min_periods=mp).median())
            rmin = grp_series.transform(
                lambda x: x.rolling(w, min_periods=mp).min())
            rmax = grp_series.transform(
                lambda x: x.rolling(w, min_periods=mp).max())
            df[f"{feat}_roll_min_{w}"]   = rmin                      # FIX 10
            df[f"{feat}_roll_max_{w}"]   = rmax
            df[f"{feat}_roll_range_{w}"] = rmax - rmin
            # Normalized oscillator: where is current value in [min, max]?
            # 0 = at rolling low, 1 = at rolling high — real quant signal
            df[f"{feat}_roll_pos_{w}"]   = (
                (df[feat] - rmin) / (rmax - rmin + 1e-7)
            )

    # ── 7. EWM — spans scaled to horizon ──────────────────────────────────────
    # FIX 4: longer spans for long horizons (slow signal needs slower decay)
    if horizon <= 3:
        spans = [3, 5, 10]
    elif horizon <= 10:
        spans = [5, 10, 20]
    else:  # horizon == 25
        spans = [5, 10, 20, 30]

    for feat in ["feature_al", "feature_am", "feature_cg", "feature_by"]:
        if feat not in df.columns:
            continue
        for span in spans:
            df[f"{feat}_ewm_{span}"] = (
                df.groupby(group_cols)[feat]
                .transform(lambda x: x.ewm(span=span, adjust=False).mean())
            )
            # EWM std — captures local volatility with exponential decay
            df[f"{feat}_ewm_std_{span}"] = (
                df.groupby(group_cols)[feat]
                .transform(lambda x: x.ewm(span=span, adjust=False).std())
            )

    # ── 8. Rolling volatility of cross-sectional z-scores ────────────────────
    for col in ["feature_al_z", "feature_cg_z", "d_al_am_z"]:
        if col in df.columns:
            for w in [10, 20]:
                df[f"{col}_roll_std_{w}"] = (
                    df.groupby(group_cols)[col]
                    .transform(lambda x: x.rolling(w, min_periods=max(3, w//2)).std())
                )

    # ── 9. Time features ──────────────────────────────────────────────────────
    df["ts_log"]     = np.log1p(df["ts_index"])
    df["ts_mod_30"]  = df["ts_index"] % 30
    df["ts_mod_90"]  = df["ts_index"] % 90
    df["ts_sin"]     = np.sin(2 * np.pi * df["ts_index"] / 365)
    df["ts_cos"]     = np.cos(2 * np.pi * df["ts_index"] / 365)
    df["ts_sin_100"] = np.sin(2 * np.pi * df["ts_index"] / 100)
    df["ts_cos_100"] = np.cos(2 * np.pi * df["ts_index"] / 100)
    df["ts_horizon"] = df["ts_index"] * df["horizon"]

    # ── 10. Sub-category dummies ──────────────────────────────────────────────
    sub_cat_dummies = pd.get_dummies(df["sub_category"], prefix="subcat", dtype=int)
    df = pd.concat([df, sub_cat_dummies], axis=1)

    df = df.fillna(0)
    return df

# ---------------------------------------------------------------------------
# Train one horizon
# ---------------------------------------------------------------------------

def train_horizon(train_path, test_path, horizon):
    print(f"\n{'='*60}")
    print(f"  HORIZON {horizon}")
    print(f"{'='*60}")

    train_raw = pd.read_parquet(train_path).query(f"horizon == {horizon}").copy()
    test_raw  = pd.read_parquet(test_path).query(f"horizon == {horizon}").copy()

    for df in [train_raw, test_raw]:
        df["group_id"] = (
            df["code"].astype(str) + "_" +
            df["sub_code"].astype(str) + "_" +
            df["sub_category"].astype(str) + "_" +
            df["horizon"].astype(str)
        )
    for col in ["y_target", "weight"]:
        if col not in test_raw.columns:
            test_raw[col] = 0.0

    group_cols_key = ["code", "sub_code", "sub_category", "horizon"]

    # Expanding mean contamination fix (unchanged — correct logic)
    train_fe = build_features(train_raw.copy(), horizon)

    exp_cols = [c for c in train_fe.columns if "_exp_mean" in c or "_exp_std" in c]
    last_exp_stats = (
        train_fe.sort_values(group_cols_key + ["ts_index"])
        .groupby(group_cols_key)[exp_cols]
        .last()
        .reset_index()
    )

    test_fe = build_features(
        pd.concat([train_raw, test_raw], ignore_index=True)
        .sort_values(group_cols_key + ["ts_index"])
        .reset_index(drop=True),
        horizon
    )
    test_fe = test_fe[test_fe["ts_index"] > train_raw["ts_index"].max()].copy()

    if exp_cols:
        test_fe = test_fe.drop(columns=exp_cols)
        test_fe = test_fe.merge(last_exp_stats, on=group_cols_key, how="left")
        test_fe[exp_cols] = test_fe[exp_cols].fillna(0)

    print(f"  Expanding mean fix: {len(exp_cols)} columns corrected for test set")

    feat_cols = [c for c in train_fe.columns if c not in NON_FEATURE_COLS]
    h_params  = get_params(horizon)
    print(f"  Features  : {len(feat_cols)}")
    print(f"  Config    : leaves={h_params['num_leaves']}  "
          f"min_child={h_params['min_child_samples']}  "
          f"L2={h_params['lambda_l2']}")

    # Time-based split
    tr_mask  = train_fe["ts_index"] <= VAL_THRESHOLD
    val_mask = train_fe["ts_index"] >  VAL_THRESHOLD

    X_tr  = train_fe.loc[tr_mask,  feat_cols]
    y_tr  = train_fe.loc[tr_mask,  "y_target"]
    w_tr  = train_fe.loc[tr_mask,  "weight"]
    X_val = train_fe.loc[val_mask, feat_cols]
    y_val = train_fe.loc[val_mask, "y_target"]
    w_val = train_fe.loc[val_mask, "weight"]

    # FIX 13: Dual holdout — check if val score is stable across two sub-windows
    mid = (train_fe.loc[val_mask, "ts_index"].min() +
           train_fe.loc[val_mask, "ts_index"].max()) // 2
    val_early_mask = train_fe["ts_index"].between(VAL_THRESHOLD + 1, mid)
    val_late_mask  = train_fe["ts_index"] > mid

    X_test = test_fe[feat_cols]
    ids    = test_fe["id"]

    print(f"  Train: {len(X_tr):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"  Weight p50={w_tr.median():.1f}  p99={w_tr.quantile(0.99):.0f}  max={w_tr.max():.0f}")

    val_preds  = np.zeros(len(X_val))
    test_preds = np.zeros(len(X_test))

    for i, seed in enumerate(SEEDS, 1):
        print(f"  Seed {i}/{len(SEEDS)} (seed={seed})...", end=" ", flush=True)
        model = lgb.LGBMRegressor(**{**get_params(horizon), "random_state": seed})
        model.fit(
            X_tr, y_tr,
            sample_weight      = w_tr.values,
            eval_set           = [(X_val, y_val)],
            eval_sample_weight = [w_val.values],
            callbacks          = [
                lgb.early_stopping(200, verbose=False),
                lgb.log_evaluation(period=99999),
            ],
        )
        val_preds  += model.predict(X_val)  / len(SEEDS)
        test_preds += model.predict(X_test) / len(SEEDS)
        print("done")

    h_score = kaggle_score(y_val, val_preds, w_val)

    # FIX 13: Dual holdout diagnostic
    val_fe = train_fe.loc[val_mask].copy()
    val_fe["pred"] = val_preds
    early_idx = val_fe["ts_index"] <= mid
    late_idx  = val_fe["ts_index"] >  mid
    score_early = kaggle_score(
        val_fe.loc[early_idx, "y_target"],
        val_fe.loc[early_idx, "pred"],
        val_fe.loc[early_idx, "weight"]
    )
    score_late = kaggle_score(
        val_fe.loc[late_idx, "y_target"],
        val_fe.loc[late_idx, "pred"],
        val_fe.loc[late_idx, "weight"]
    )
    print(f"\n  Horizon {horizon} overall val score : {h_score:.5f}")
    print(f"  Horizon {horizon} early val score   : {score_early:.5f}  (ts <= {mid})")
    print(f"  Horizon {horizon} late  val score   : {score_late:.5f}  (ts >  {mid})")
    if abs(score_early - score_late) > 0.02:
        print(f"  *** WARNING: early/late gap > 0.02 — val may not be representative ***")

    del train_raw, test_raw, train_fe, test_fe
    del X_tr, X_val
    gc.collect()

    return (
        pd.DataFrame({"id": ids.values, "prediction": test_preds}),
        list(y_val), list(val_preds), list(w_val),
        h_score,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(train_path, test_path, output="predictions.csv"):
    all_test_preds = []
    all_y, all_p, all_w = [], [], []
    horizon_scores = {}

    for h in HORIZONS:
        test_df, y_val, p_val, w_val, h_score = train_horizon(
            train_path, test_path, h
        )
        all_test_preds.append(test_df)
        all_y.extend(y_val)
        all_p.extend(p_val)
        all_w.extend(w_val)
        horizon_scores[h] = h_score

    overall = kaggle_score(all_y, all_p, all_w)

    print(f"\n{'='*60}")
    print(f"  PER-HORIZON LOCAL SCORES (val: ts_index > {VAL_THRESHOLD})")
    for h, s in horizon_scores.items():
        print(f"    Horizon {h:2d}: {s:.5f}")
    print(f"  OVERALL LOCAL SCORE : {overall:.5f}")
    print(f"{'='*60}")

    submission = pd.concat(all_test_preds, axis=0, ignore_index=True)
    submission.to_csv(output, index=False)

    print(f"\nSubmission stats:")
    print(f"  Rows : {len(submission):,}")
    print(f"  mean : {submission['prediction'].mean():.6f}")
    print(f"  std  : {submission['prediction'].std():.6f}")
    print(f"  min  : {submission['prediction'].min():.4f}")
    print(f"  max  : {submission['prediction'].max():.4f}")
    print(f"  NaN  : {submission['prediction'].isna().sum()}")
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.parquet")
    parser.add_argument("--test",  default="test.parquet")
    parser.add_argument("--out",   default="predictions.csv")
    args = parser.parse_args()
    run(args.train, args.test, args.out)