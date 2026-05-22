"""
modeling.py
-----------
PART 1 – STAGE 3: AI Modeling — Training, Inference & Evaluation

Responsibilities
~~~~~~~~~~~~~~~~
* Load ``telecom_churn_100k.csv`` produced by the processing stage.
* Apply the exact same feature engineering steps as the notebook:
    - Drop zero-variance, collinear, sparse, and ID columns.
    - Create binary data-user flags.
    - Compute ratio features with division-by-zero guards.
    - Compute network-quality delta.
    - Log-transform heavy-tailed numerical features.
    - Build an engagement composite score.
    - Binarise the tenure column.
* Define numerical and categorical feature sets.
* Perform a three-way stratified split (Train 70 / Val 15 / Test 15).
* Build a leak-free sklearn / imblearn Pipeline:
    - Numerical branch  : median impute → StandardScaler
    - Categorical branch: constant impute → TargetEncoder (smoothing=10) → StandardScaler
    - SMOTE inside ImbPipeline (fit-only, never applied during predict)
    - LightGBM classifier
* Run RandomizedSearchCV for hyperparameter optimisation.
* Calibrate the best pipeline with CalibratedClassifierCV (isotonic, prefit).
* Tune the decision threshold on the validation set.
* Evaluate on the held-out test set and produce a full metrics report.
* Segment customers into risk tiers and compute business KPIs.
* Save the calibrated model and all metadata as a versioned ``joblib`` payload.

Prediction parity guarantee
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Every step — column ordering, drop list, log-transform application,
engagement-score weights, split seeds, pipeline topology, calibration
method, and threshold selection — is transcribed verbatim from the
notebook so that the DAG generates byte-identical predictions.
"""

from __future__ import annotations

import datetime
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.config import (
    COLLINEAR_COLS,
    COST_PER_CAMPAIGN,
    CLV,
    ENGAGEMENT_WEIGHTS,
    ID_COLS,
    LGB_PARAM_GRID,
    LOG_TRANSFORM_COLS,
    LOYAL_VALUE,
    MODELS_DIR,
    OUTPUTS_DIR,
    PREDICTIONS_FILE,
    RANDOM_STATE,
    RISK_TIERS,
    SEED,
    SPARSE_COLS,
    TARGET_COL,
    TELECOM_CHURN_100K_FILE,
    TEST_SIZE,
    VAL_SIZE,
    ZERO_VAR_COLS,
    LATEST_MODEL_SYMLINK,
    MODEL_FILE_TEMPLATE,
    EVAL_METRICS_FILE,
)
from src.logger import get_logger

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

log = get_logger(__name__)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1.  DATA LOADING                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_training_data(path: str | Path = TELECOM_CHURN_100K_FILE) -> pd.DataFrame:
    """
    Load the processed 100 k-row telecom churn dataset.

    Parameters
    ----------
    path : Path to ``telecom_churn_100k.csv``.

    Returns
    -------
    pd.DataFrame — raw loaded data before feature engineering.
    """
    log.info("[Modeling] Loading training data from %s …", path)
    df = pd.read_csv(str(path))
    log.info(
        "[Modeling] Loaded: %s | Churn rate: %.1f%%",
        df.shape, df[TARGET_COL].mean() * 100
    )
    return df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2.  FEATURE ENGINEERING                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps from the notebook exactly.

    Steps (order preserved for prediction parity)
    -----------------------------------------------
    1. Drop zero-variance, collinear, sparse, and ID columns.
    2. Binary data-user flags (before imputation).
    3. Ratio features (guard against division by zero).
    4. Network quality delta (region vs. arrondissement).
    5. Log-transform heavy-tailed features.
    6. Engagement composite score.
    7. Binarise tenure → ``is_loyal`` then drop original ``tenure``.

    Parameters
    ----------
    df_raw : Raw DataFrame as loaded from CSV.

    Returns
    -------
    pd.DataFrame — engineered feature set ready for train/val/test split.
    """
    log.info("[Modeling] Starting feature engineering …")

    # ── Step 1: Drop uninformative columns ──────────────────────────────────
    drop_candidates = ZERO_VAR_COLS + COLLINEAR_COLS + SPARSE_COLS + ID_COLS
    drop_cols = [c for c in drop_candidates if c in df_raw.columns]
    df = df_raw.drop(columns=drop_cols).copy()
    log.info("[Modeling] Dropped %d columns → shape: %s | Dropped: %s",
             len(drop_cols), df.shape, drop_cols)

    # ── Step 2: Binary data-user flags ──────────────────────────────────────
    df["is_data_user"] = (df["data_volume"].fillna(0) > 0).astype(int)
    df["no_data_flag"] = df["data_volume"].isna().astype(int)

    # ── Step 3: Ratio features ───────────────────────────────────────────────
    df["avg_recharge_amount"] = df["montant"]  / df["frequence_rech"].replace(0, np.nan)
    df["avg_revenue_per_tx"]  = df["revenue"]  / df["frequence"].replace(0, np.nan)

    # ── Step 4: Network quality delta ────────────────────────────────────────
    if "region_network_quality_score" in df and "arr_network_quality_score" in df:
        df["network_quality_delta"] = (
            df["region_network_quality_score"] - df["arr_network_quality_score"]
        )

    # ── Step 5: Log-transform heavy-tailed features ──────────────────────────
    for col in LOG_TRANSFORM_COLS:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].fillna(0).clip(lower=0))

    # ── Step 6: Engagement composite score ───────────────────────────────────
    score   = pd.Series(0.0, index=df.index)
    total_w = 0.0
    for col, w in ENGAGEMENT_WEIGHTS.items():
        if col in df.columns:
            col_range = df[col].max() - df[col].min() + 1e-9
            norm      = (df[col] - df[col].min()) / col_range
            score    += w * norm.fillna(0)
            total_w  += w
    df["engagement_score"] = score / total_w

    # ── Step 7: Tenure binarisation ──────────────────────────────────────────
    tenure_col = "tenure"
    if tenure_col in df.columns:
        observed = df[tenure_col].value_counts()
        log.info("[Modeling] Tenure value counts:\n%s", observed.to_string())
        if LOYAL_VALUE not in observed.index:
            raise ValueError(
                f"[Modeling] LOYAL_VALUE '{LOYAL_VALUE}' not found in '{tenure_col}'. "
                f"Observed values: {observed.index.tolist()}"
            )
        df["is_loyal"] = (df[tenure_col] == LOYAL_VALUE).astype(int)
        df.drop(columns=[tenure_col], inplace=True)

    log.info("[Modeling] Feature engineering complete → shape: %s", df.shape)
    return df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3.  FEATURE SET DEFINITION & DATA SPLITTING                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def define_feature_sets(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Identify numerical and categorical feature columns from the engineered
    DataFrame.  Excludes the target column and any residual category columns.

    Returns
    -------
    num_cols, cat_cols
    """
    cat_cols = [c for c in ["region", "top_pack"] if c in df.columns]
    num_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c != TARGET_COL and c not in cat_cols
    ]
    log.info(
        "[Modeling] Numerical features: %d | Categorical features: %s",
        len(num_cols), cat_cols
    )
    return num_cols, cat_cols


def split_data(
    df:       pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> tuple:
    """
    Three-way stratified split: Train 70% / Val 15% / Test 15%.

    The validation set is used **only** for calibration and threshold
    tuning; the test set remains unseen until final evaluation.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df[num_cols + cat_cols]
    y = df[TARGET_COL]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, stratify=y_temp, random_state=SEED
    )

    log.info(
        "[Modeling] Split — Train: %s (%.1f%% churn) | Val: %s | Test: %s",
        X_train.shape, y_train.mean() * 100, X_val.shape, X_test.shape
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4.  PIPELINE BUILDER                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_pipeline(
    model_name: str,
    num_cols:   list[str],
    cat_cols:   list[str],
    use_smote:  bool = True,
) -> ImbPipeline:
    """
    Construct a fully leak-free preprocessing + model ImbPipeline.

    Numerical branch  : median impute → StandardScaler
    Categorical branch: constant impute ('Unknown') → TargetEncoder → StandardScaler
    Resampling        : SMOTE (fit-only via ImbPipeline, never applied at predict)
    Classifier        : one of logistic_regression | random_forest | lightgbm | xgboost

    Parameters
    ----------
    model_name : Classifier key.
    num_cols   : Numerical feature column names.
    cat_cols   : Categorical feature column names.
    use_smote  : Whether to include SMOTE resampling.

    Returns
    -------
    ImbPipeline
    """
    num_transformer = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])

    # TargetEncoder with smoothing — fitted only on training fold inside Pipeline
    cat_transformer = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", TargetEncoder(smoothing=10, min_samples_leaf=5)),
        ("scale",  StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ],
        remainder="drop",
    )

    models = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced", n_jobs=-1, random_state=SEED
        ),
        "lightgbm": lgb.LGBMClassifier(
            class_weight="balanced", n_jobs=-1, random_state=SEED, verbose=-1
        ),
        "xgboost": xgb.XGBClassifier(
            eval_metric="logloss", n_jobs=-1, random_state=SEED, verbosity=0
        ),
    }

    if model_name not in models:
        raise ValueError(
            f"[Modeling] Unknown model '{model_name}'. "
            f"Choose from: {list(models.keys())}"
        )
    estimator = models[model_name]

    steps = [("preprocessor", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=SEED, k_neighbors=5)))
    steps.append(("clf", estimator))

    return ImbPipeline(steps)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5.  HYPERPARAMETER SEARCH                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_hyperparameter_search(
    pipeline:  ImbPipeline,
    X_train:   pd.DataFrame,
    y_train:   pd.Series,
    param_grid: dict,
    n_iter:    int = 20,
) -> ImbPipeline:
    """
    Run RandomizedSearchCV over *param_grid* optimising for ROC-AUC with
    5-fold stratified cross-validation.

    Parameters
    ----------
    pipeline   : ImbPipeline to search over.
    X_train    : Training features.
    y_train    : Training labels.
    param_grid : Dict of parameter distributions.
    n_iter     : Number of random combinations to try.

    Returns
    -------
    best_estimator_ from RandomizedSearchCV (already refitted on full train set).
    """
    log.info("[Modeling] Starting RandomizedSearchCV (n_iter=%d) …", n_iter)
    t0 = time.time()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=SEED,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    elapsed = time.time() - t0
    log.info(
        "[Modeling] Search done in %.1fs | Best ROC-AUC: %.4f | Best params: %s",
        elapsed, search.best_score_, search.best_params_
    )
    return search.best_estimator_


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6.  PROBABILITY CALIBRATION                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def calibrate_model(
    best_pipeline: ImbPipeline,
    X_val:         pd.DataFrame,
    y_val:         pd.Series,
) -> CalibratedClassifierCV:
    """
    Wrap the already-fitted best pipeline with isotonic calibration.

    Calibration is performed on the **validation set** — completely
    independent from training data so the test set stays unseen.

    Parameters
    ----------
    best_pipeline : Fitted ImbPipeline from hyperparameter search.
    X_val         : Validation features.
    y_val         : Validation labels.

    Returns
    -------
    CalibratedClassifierCV — calibrated model ready for inference.
    """
    log.info("[Modeling] Calibrating model on validation set (isotonic) …")
    calibrated = CalibratedClassifierCV(best_pipeline, method="isotonic", cv="prefit")
    calibrated.fit(X_val, y_val)
    log.info("[Modeling] Calibration complete.")
    return calibrated


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  7.  THRESHOLD TUNING                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def tune_threshold(
    calibrated_model: CalibratedClassifierCV,
    X_val:            pd.DataFrame,
    y_val:            pd.Series,
) -> float:
    """
    Select the decision threshold that maximises F1 on the validation set.

    Scans thresholds from 0.10 to 0.90 in 0.01 increments.  The test set
    is never touched during this step.

    Returns
    -------
    float — optimal threshold value.
    """
    log.info("[Modeling] Tuning decision threshold on validation set …")
    proba_val  = calibrated_model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.10, 0.91, 0.01)
    rows = []

    for t in thresholds:
        preds = (proba_val >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 2),
            "f1":        f1_score(y_val, preds, zero_division=0),
            "precision": precision_score(y_val, preds, zero_division=0),
            "recall":    recall_score(y_val, preds, zero_division=0),
        })

    thresh_df     = pd.DataFrame(rows)
    best_row      = thresh_df.loc[thresh_df["f1"].idxmax()]
    best_threshold = float(best_row["threshold"])

    log.info(
        "[Modeling] Best threshold: %.2f | F1=%.4f | Precision=%.4f | Recall=%.4f",
        best_threshold, best_row["f1"], best_row["precision"], best_row["recall"]
    )
    return best_threshold


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  8.  EVALUATION                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def evaluate_model(
    calibrated_model: CalibratedClassifierCV,
    X_test:           pd.DataFrame,
    y_test:           pd.Series,
    best_threshold:   float,
) -> dict:
    """
    Compute all evaluation metrics on the held-out test set.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc,
                    avg_precision, threshold, proba_test, y_pred_test.
    """
    log.info("[Modeling] Evaluating on test set (threshold=%.2f) …", best_threshold)

    proba_test  = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred_test = (proba_test >= best_threshold).astype(int)

    metrics = {
        "accuracy":      accuracy_score(y_test, y_pred_test),
        "precision":     precision_score(y_test, y_pred_test, zero_division=0),
        "recall":        recall_score(y_test, y_pred_test, zero_division=0),
        "f1":            f1_score(y_test, y_pred_test, zero_division=0),
        "roc_auc":       roc_auc_score(y_test, proba_test),
        "avg_precision": average_precision_score(y_test, proba_test),
        "threshold":     best_threshold,
    }

    log.info("[Modeling] ── Evaluation Report ──")
    for k, v in metrics.items():
        if k not in ("proba_test", "y_pred_test"):
            log.info("  %-20s: %.4f", k, v)

    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    log.info(
        "[Modeling] Confusion Matrix — TN=%d | FP=%d | FN=%d | TP=%d",
        tn, fp, fn, tp
    )

    # Business KPI report
    total_targeted  = tp + fp
    campaign_spend  = total_targeted * COST_PER_CAMPAIGN
    revenue_saved   = tp * CLV
    net_value       = revenue_saved - campaign_spend
    roi             = net_value / (campaign_spend + 1e-9) * 100
    log.info(
        "[Modeling] Business KPIs — Targeted: %d | Retained (TP): %d | "
        "Campaign spend: $%.0f | Revenue saved: $%.0f | Net value: $%.0f | ROI: %.1f%%",
        total_targeted, tp, campaign_spend, revenue_saved, net_value, roi
    )

    # Attach arrays for use by the segmentation step
    metrics["proba_test"]  = proba_test
    metrics["y_pred_test"] = y_pred_test
    return metrics


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  9.  CUSTOMER SEGMENTATION                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _risk_tier(prob: float) -> str:
    for tier, (lo, hi) in RISK_TIERS.items():
        if lo <= prob < hi:
            return tier
    return "Low"


def segment_customers(
    X_test:      pd.DataFrame,
    y_test:      pd.Series,
    proba_test:  np.ndarray,
    y_pred_test: np.ndarray,
) -> pd.DataFrame:
    """
    Build the scored predictions DataFrame with risk tier and action
    segment columns (matches notebook output schema exactly).

    Columns produced
    ----------------
    All original X_test columns + actual_churn, churn_probability,
    churn_prediction, risk_segment, value_segment, action_segment.

    Returns
    -------
    pd.DataFrame — sorted by churn_probability descending.
    """
    log.info("[Modeling] Segmenting customers …")
    scored = X_test.copy()
    scored["actual_churn"]      = y_test.values
    scored["churn_probability"] = proba_test
    scored["churn_prediction"]  = y_pred_test
    scored["risk_segment"]      = scored["churn_probability"].map(_risk_tier)

    # Value segmentation on median revenue
    rev_col = "revenue" if "revenue" in scored.columns else None
    if rev_col:
        median_rev = scored[rev_col].median()
        scored["value_segment"] = np.where(
            scored[rev_col] >= median_rev, "High Value", "Low Value"
        )
        conditions = [
            (scored["risk_segment"] == "High") & (scored["value_segment"] == "High Value"),
            (scored["risk_segment"] == "High") & (scored["value_segment"] == "Low Value"),
            scored["risk_segment"].isin(["Medium", "Low"]) & (scored["value_segment"] == "High Value"),
        ]
        choices = ["Priority Retention", "Low-cost Campaign", "Loyalty Program"]
        scored["action_segment"] = np.select(conditions, choices, default="Monitor")
    else:
        scored["action_segment"] = scored["risk_segment"].map({
            "High": "Priority Retention", "Medium": "Low-cost Campaign", "Low": "Monitor"
        })

    scored = scored.sort_values("churn_probability", ascending=False).reset_index(drop=True)

    log.info("[Modeling] Risk tier distribution:\n%s",
             scored["risk_segment"].value_counts().to_string())
    log.info("[Modeling] Action segment distribution:\n%s",
             scored["action_segment"].value_counts().to_string())
    return scored


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  10.  AIRFLOW ENTRYPOINT                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_model_inference(**kwargs) -> dict:
    """
    Airflow ``PythonOperator`` callable — full modeling stage:

    1. Load 100 k training data.
    2. Feature engineering (notebook-identical).
    3. Define feature sets and perform three-way split.
    4. Build LightGBM ImbPipeline with SMOTE.
    5. Run RandomizedSearchCV.
    6. Calibrate on validation set.
    7. Tune threshold on validation set.
    8. Evaluate on test set.
    9. Segment customers.
    10. Save calibrated model as a versioned joblib payload.
    11. Save evaluation metrics CSV.
    12. Return paths for XCom.

    Returns
    -------
    dict with keys: model_path, metrics.
    """
    log.info("[Modeling] ══════ MODELING STAGE STARTED ══════")
    t0 = time.time()

    # ── 1-3. Load → engineer → split ────────────────────────────────────────
    df_raw   = load_training_data(TELECOM_CHURN_100K_FILE)
    df       = engineer_features(df_raw)
    num_cols, cat_cols = define_feature_sets(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, num_cols, cat_cols)

    # ── 4-5. Pipeline + hyperparameter search ────────────────────────────────
    lgb_pipeline = build_pipeline("lightgbm", num_cols, cat_cols, use_smote=True)
    best_lgb     = run_hyperparameter_search(lgb_pipeline, X_train, y_train, LGB_PARAM_GRID)

    # ── 6. Calibration ───────────────────────────────────────────────────────
    calibrated_model = calibrate_model(best_lgb, X_val, y_val)

    # ── 7. Threshold tuning ──────────────────────────────────────────────────
    best_threshold = tune_threshold(calibrated_model, X_val, y_val)

    # ── 8. Test evaluation ───────────────────────────────────────────────────
    metrics = evaluate_model(calibrated_model, X_test, y_test, best_threshold)
    proba_test  = metrics.pop("proba_test")
    y_pred_test = metrics.pop("y_pred_test")

    # ── 9. Customer segmentation ─────────────────────────────────────────────
    scored_df = segment_customers(X_test, y_test, proba_test, y_pred_test)

    # ── 10. Save model ───────────────────────────────────────────────────────
    version    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_filename = MODEL_FILE_TEMPLATE.format(version=version)
    model_path     = MODELS_DIR / model_filename
    payload = {
        "model": calibrated_model,
        "metadata": {
            "version":    version,
            "model_name": "lightgbm",
            "threshold":  best_threshold,
            "roc_auc":    metrics["roc_auc"],
            "f1":         metrics["f1"],
            "num_cols":   num_cols,
            "cat_cols":   cat_cols,
            "smote":      True,
            "calibrated": True,
        },
    }
    joblib.dump(payload, str(model_path))
    log.info("[Modeling] ✓ Model saved → %s", model_path)

    # Update the stable symlink (churn_model.joblib) that Streamlit uses
    symlink = Path(LATEST_MODEL_SYMLINK)
    if symlink.is_symlink() or symlink.exists():
        symlink.unlink()
    symlink.symlink_to(model_path.resolve())
    log.info("[Modeling] ✓ Stable symlink updated → %s", symlink)

    # ── 11. Save metrics ─────────────────────────────────────────────────────
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame([
        {"metric": k, "value": v}
        for k, v in metrics.items()
    ])
    metrics_df.to_csv(str(EVAL_METRICS_FILE), index=False)
    log.info("[Modeling] ✓ Metrics saved → %s", EVAL_METRICS_FILE)

    elapsed = time.time() - t0
    result = {
        "model_path":    str(model_path),
        "metrics":       metrics,
        "scored_df_len": len(scored_df),
    }

    # Pass scored_df via XCom-compatible serialisation (path reference)
    scored_df.to_csv(str(PREDICTIONS_FILE), index=False)
    log.info("[Modeling] ✓ Scored predictions temporarily saved → %s", PREDICTIONS_FILE)

    log.info("[Modeling] ══════ MODELING STAGE COMPLETE (%.1fs) ══════", elapsed)
    return result


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  11.  STANDALONE EXECUTION                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    import pprint
    pprint.pprint(run_model_inference())
