"""LightGBM training pipeline with calibration + threshold tuning.

Source notebook: notebooks/Telecom Customer Churn - Training, Inference
& Evaluation - Project Team A.ipynb

The training entrypoint is :func:`train`. It produces:

* a versioned joblib (``models/lgb_churn_pipeline_<ts>.joblib``)
* a stable joblib (``models/churn_model.joblib``) — what the dashboard reads
* ``outputs/metrics/evaluation_metrics.csv``
* ``outputs/metrics/feature_importances.csv``

Joblib payload schema (matches dashboards/churn_dashboard_app.py):

::

    {
        "model":    CalibratedClassifierCV(...),
        "metadata": {
            "version": "20260506_120000",
            "model_name": "lightgbm",
            "threshold": 0.42,
            "roc_auc": 0.85,
            "f1": 0.69,
            "num_cols": [...],
            "cat_cols": [...],
            "smote": True,
            "calibrated": True,
        },
    }
"""
from __future__ import annotations

import datetime as dt
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config, data_io, feature_engineering, paths
from .logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Optional MLflow tracking
# ---------------------------------------------------------------------------
def _maybe_mlflow_start():
    """Return an MLflow run context if MLFLOW_TRACKING_URI is set, else None."""
    import os
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return None
    try:
        import mlflow
    except ImportError:
        log.warning("MLFLOW_TRACKING_URI is set but mlflow is not installed.")
        return None
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "telecom_churn"))
    return mlflow


def _mlflow_log(mlflow, params: dict, metrics: dict, artifacts: list[Path]) -> None:
    if mlflow is None:
        return
    mlflow.log_params({k: str(v) for k, v in params.items()})
    mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
    for art in artifacts:
        if Path(art).exists():
            mlflow.log_artifact(str(art))

# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------
def build_pipeline(
    num_cols: list[str],
    cat_cols: list[str],
    use_smote: bool = True,
    seed: int | None = None,
) -> Any:
    """Construct the (preprocessor → SMOTE? → LightGBM) imblearn Pipeline."""
    import lightgbm as lgb
    from category_encoders import TargetEncoder
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    seed = config.SEED if seed is None else seed

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", TargetEncoder(smoothing=10, min_samples_leaf=5)),
        ("scale",  StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")

    clf = lgb.LGBMClassifier(
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
        verbose=-1,
    )

    steps = [("preprocessor", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=seed, k_neighbors=config.CONFIG["model"]["smote_k_neighbors"])))
    steps.append(("clf", clf))
    return ImbPipeline(steps)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    input_path: Path,
    model_dir: Path | None = None,
    metrics_dir: Path | None = None,
    run_shap: bool = True,
) -> dict[str, Any]:
    """Train the full pipeline and persist models + metrics."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        classification_report,
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

    input_path = Path(input_path)
    model_dir = Path(model_dir or paths.MODELS_DIR)
    metrics_dir = Path(metrics_dir or paths.METRICS_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    cfg = config.CONFIG["model"]
    seed = config.SEED
    np.random.seed(seed)

    # --------------------------------------------------- Load + feature engineer
    df_raw = data_io.read_csv(input_path)
    df = feature_engineering.prepare_features(df_raw)
    num_cols, cat_cols = feature_engineering.split_feature_columns(df)
    log.info("num_cols=%s, cat_cols=%s", len(num_cols), len(cat_cols))

    target = config.TARGET
    X = df[num_cols + cat_cols]
    y = df[target].astype(int)

    # --------------------------------------------------- Split
    test_size = cfg["test_size"]
    val_size = cfg["val_size"] / (1.0 - cfg["test_size"])  # of remainder
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=seed)
    log.info("sizes  train=%s  val=%s  test=%s",
             f"{len(X_train):,}", f"{len(X_val):,}", f"{len(X_test):,}")

    # --------------------------------------------------- LightGBM search
    pipe = build_pipeline(num_cols, cat_cols, use_smote=True, seed=seed)
    cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=cfg["lgbm_param_grid"],
        n_iter=cfg["n_iter_search"],
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=seed,
        verbose=1,
        refit=True,
    )
    log.info("starting RandomizedSearchCV …")
    t0 = time.time()
    search.fit(X_train, y_train)
    log.info("search done in %.1fs | best ROC-AUC=%.4f",
             time.time() - t0, search.best_score_)
    best_pipe = search.best_estimator_

    # --------------------------------------------------- Calibration on val
    calibrated = CalibratedClassifierCV(
        best_pipe, method=cfg["calibration_method"], cv="prefit",
    )
    calibrated.fit(X_val, y_val)

    # --------------------------------------------------- Threshold tuning
    proba_val = calibrated.predict_proba(X_val)[:, 1]
    rows = []
    for t in np.arange(0.10, 0.91, 0.01):
        preds = (proba_val >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 2),
            "precision": precision_score(y_val, preds, zero_division=0),
            "recall":    recall_score(y_val, preds, zero_division=0),
            "f1":        f1_score(y_val, preds, zero_division=0),
        })
    thresh_df = pd.DataFrame(rows)
    best_threshold = float(thresh_df.loc[thresh_df["f1"].idxmax(), "threshold"])
    log.info("best threshold (F1) = %.2f", best_threshold)

    # --------------------------------------------------- Test evaluation
    proba_test = calibrated.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= best_threshold).astype(int)

    metrics = {
        "accuracy":      float(accuracy_score(y_test, y_pred)),
        "precision":     float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":        float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":            float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":       float(roc_auc_score(y_test, proba_test)),
        "avg_precision": float(average_precision_score(y_test, proba_test)),
        "threshold":     best_threshold,
    }
    log.info("metrics: %s", metrics)
    log.info("\n%s", classification_report(
        y_test, y_pred, target_names=["No Churn", "Churn"]))

    # --------------------------------------------------- Feature importances
    fi_path = metrics_dir / "feature_importances.csv"
    try:
        clf = best_pipe.named_steps["clf"]
        pre = best_pipe.named_steps["preprocessor"]
        try:
            feat_names = list(pre.get_feature_names_out())
        except Exception:
            feat_names = num_cols + cat_cols
        if hasattr(clf, "feature_importances_") and len(clf.feature_importances_) == len(feat_names):
            fi = pd.DataFrame({
                "feature": feat_names,
                "importance": clf.feature_importances_,
            }).sort_values("importance", ascending=False)
            fi.to_csv(fi_path, index=False)
            log.info("wrote %s", fi_path)
    except Exception as exc:
        log.warning("feature-importance export failed: %s", exc)

    # --------------------------------------------------- SHAP (optional)
    if run_shap:
        try:
            _shap_summary(best_pipe, X_test, num_cols, cat_cols, metrics_dir)
        except Exception as exc:
            log.warning("SHAP step skipped: %s", exc)

    # --------------------------------------------------- Persist metrics
    metrics_csv = metrics_dir / "evaluation_metrics.csv"
    pd.DataFrame(
        [{"metric": k, "value": v} for k, v in metrics.items()]
    ).to_csv(metrics_csv, index=False)
    log.info("wrote %s", metrics_csv)

    # --------------------------------------------------- Persist model
    version = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned = model_dir / f"lgb_churn_pipeline_{version}.joblib"
    payload: dict[str, Any] = {
        "model": calibrated,
        "metadata": {
            "version":     version,
            "model_name":  "lightgbm",
            "threshold":   best_threshold,
            "roc_auc":     metrics["roc_auc"],
            "f1":          metrics["f1"],
            "num_cols":    num_cols,
            "cat_cols":    cat_cols,
            "smote":       True,
            "calibrated":  True,
            "trained_at":  dt.datetime.utcnow().isoformat() + "Z",
        },
    }
    data_io.dump_joblib(payload, versioned)

    # The dashboard reads this stable filename; copy not symlink (cross-OS).
    # ``copyfile`` (not ``copy2``) so we don't try to preserve mtime — Docker
    # bind mounts on Windows hosts deny ``os.utime`` (Operation not permitted).
    stable = paths.CHURN_MODEL_JOBLIB
    stable.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(versioned, stable)
    log.info("copied %s -> %s", versioned.name, stable)

    # ---------------------------------- MLflow logging (no-op if not configured)
    mlflow = _maybe_mlflow_start()
    if mlflow is not None:
        with mlflow.start_run(run_name=f"churn_{version}"):
            _mlflow_log(
                mlflow,
                params={
                    "model": "lightgbm",
                    "calibration": cfg["calibration_method"],
                    "smote": True,
                    "test_size": cfg["test_size"],
                    "val_size": cfg["val_size"],
                    "n_iter_search": cfg["n_iter_search"],
                    "cv_folds": cfg["cv_folds"],
                    "best_threshold": best_threshold,
                    "version": version,
                },
                metrics=metrics,
                artifacts=[versioned, metrics_csv, fi_path,
                           metrics_dir / "shap_summary.png"],
            )
            log.info("MLflow run logged")

    return {
        "metrics": metrics,
        "model_path": str(stable),
        "versioned_model_path": str(versioned),
    }


# ---------------------------------------------------------------------------
# SHAP helper
# ---------------------------------------------------------------------------
def _shap_summary(best_pipe, X_test, num_cols, cat_cols, metrics_dir: Path) -> None:
    import matplotlib
    import shap
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pre = best_pipe.named_steps["preprocessor"]
    clf = best_pipe.named_steps["clf"]
    X_t = pd.DataFrame(
        pre.transform(X_test),
        columns=list(getattr(pre, "get_feature_names_out", lambda: num_cols + cat_cols)()),
    )
    n = min(config.CONFIG["model"]["shap_sample_size"], len(X_t))
    sample = X_t.sample(n=n, random_state=config.SEED)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(sample)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    fig = plt.figure()
    shap.summary_plot(sv, sample, plot_type="bar", show=False)
    out = metrics_dir / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)
