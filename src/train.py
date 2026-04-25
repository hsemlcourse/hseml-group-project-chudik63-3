"""Train weather rain prediction models.

Example:
    python -m src.train --quick
    python -m src.train
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier

from src.config import MODELS_DIR, RANDOM_STATE
from src.data_loading import find_weather_csv, load_weather_data
from src.evaluate import (
    evaluate_binary_classifier,
    find_best_threshold,
    metrics_to_row,
    predict_proba_positive,
)
from src.preprocessing import (
    basic_clean,
    build_pipeline,
    chronological_train_val_test_split,
    split_features_target,
)
from src.utils import ensure_project_dirs, set_seed

warnings.filterwarnings("ignore", category=UserWarning)


def _optional_boosters() -> dict[str, object]:
    """Return optional gradient boosting models if installed."""
    models: dict[str, object] = {}
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        pass
    return models


def get_candidate_models(quick: bool = False) -> dict[str, object]:
    """Models required by the project criteria: baseline, 4-5 models and ensembles."""
    n_estimators = 60 if quick else 250
    models: dict[str, object] = {
        "Baseline LogisticRegression": LogisticRegression(
            max_iter=700,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=35 if not quick else 15),
        "RandomForest": RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=18,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    if not quick:
        models.update(_optional_boosters())
    return models


def tune_random_forest(X_train, y_train, X_val, y_val, quick: bool = False):
    """Small hyperparameter search for RandomForest using validation PR-AUC."""
    grid = {
        "n_estimators": [80] if quick else [150, 250],
        "max_depth": [10, 18] if quick else [10, 18, None],
        "min_samples_leaf": [2, 5],
    }
    best_pipeline = None
    best_score = -1.0
    best_params = None

    for params in ParameterGrid(grid):
        model = RandomForestClassifier(
            **params,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        pipeline = build_pipeline(model, use_feature_engineering=True)
        pipeline.fit(X_train, y_train)
        metrics = evaluate_binary_classifier(pipeline, X_val, y_val)
        score = float(metrics["pr_auc"])
        if score > best_score:
            best_score = score
            best_pipeline = pipeline
            best_params = params

    return best_pipeline, best_score, best_params


def run_training(csv_path: str | Path | None = None, quick: bool = False) -> pd.DataFrame:
    """Run full experiment loop and save the best model."""
    set_seed(RANDOM_STATE)
    ensure_project_dirs()

    if csv_path is not None:
        csv_path = find_weather_csv(csv_path) if Path(csv_path).is_dir() else csv_path
    df = basic_clean(load_weather_data(csv_path))

    train_df, val_df, test_df = chronological_train_val_test_split(df)
    X_train, y_train = split_features_target(train_df)
    X_val, y_val = split_features_target(val_df)
    X_test, y_test = split_features_target(test_df)

    rows = []
    fitted_models: dict[str, object] = {}

    for name, model in get_candidate_models(quick=quick).items():
        # Baseline is intentionally without feature engineering; all other models use FE.
        use_fe = name != "Baseline LogisticRegression"
        pipeline = build_pipeline(model, use_feature_engineering=use_fe)
        pipeline.fit(X_train, y_train)

        val_score = predict_proba_positive(pipeline, X_val)
        threshold, _ = find_best_threshold(y_val, val_score)
        val_metrics = evaluate_binary_classifier(pipeline, X_val, y_val, threshold)
        test_metrics = evaluate_binary_classifier(pipeline, X_test, y_test, threshold)

        rows.append(metrics_to_row(name, "val", val_metrics, threshold))
        rows.append(metrics_to_row(name, "test", test_metrics, threshold))
        fitted_models[name] = pipeline
        print(f"{name}: val PR-AUC={val_metrics['pr_auc']:.4f}, test PR-AUC={test_metrics['pr_auc']:.4f}")

    tuned_rf, tuned_pr_auc, tuned_params = tune_random_forest(X_train, y_train, X_val, y_val, quick=quick)
    threshold, _ = find_best_threshold(y_val, predict_proba_positive(tuned_rf, X_val))
    tuned_name = f"Tuned RandomForest {tuned_params}"
    rows.append(
        metrics_to_row(
            tuned_name,
            "val",
            evaluate_binary_classifier(tuned_rf, X_val, y_val, threshold),
            threshold,
        )
    )
    rows.append(
        metrics_to_row(
            tuned_name,
            "test",
            evaluate_binary_classifier(tuned_rf, X_test, y_test, threshold),
            threshold,
        )
    )
    fitted_models[tuned_name] = tuned_rf
    print(f"Tuned RandomForest: best validation PR-AUC={tuned_pr_auc:.4f}, params={tuned_params}")

    # Ensemble over stable models. It is trained after individual models, still only on train split.
    ensemble_estimators = [
        (
            "lr",
            build_pipeline(
                LogisticRegression(max_iter=700, class_weight="balanced", random_state=RANDOM_STATE),
                use_feature_engineering=True,
            ),
        ),
        (
            "rf",
            build_pipeline(
                RandomForestClassifier(
                    n_estimators=80 if quick else 200,
                    class_weight="balanced_subsample",
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                use_feature_engineering=True,
            ),
        ),
        (
            "et",
            build_pipeline(
                ExtraTreesClassifier(
                    n_estimators=80 if quick else 200,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                use_feature_engineering=True,
            ),
        ),
    ]
    voting = VotingClassifier(estimators=ensemble_estimators, voting="soft", n_jobs=-1)
    voting.fit(X_train, y_train)
    threshold, _ = find_best_threshold(y_val, predict_proba_positive(voting, X_val))
    rows.append(metrics_to_row("SoftVoting ensemble", "val", evaluate_binary_classifier(voting, X_val, y_val, threshold), threshold))
    rows.append(metrics_to_row("SoftVoting ensemble", "test", evaluate_binary_classifier(voting, X_test, y_test, threshold), threshold))
    fitted_models["SoftVoting ensemble"] = voting

    if not quick:
        stacking = StackingClassifier(
            estimators=ensemble_estimators,
            final_estimator=LogisticRegression(max_iter=500, class_weight="balanced"),
            passthrough=False,
            n_jobs=-1,
        )
        stacking.fit(X_train, y_train)
        threshold, _ = find_best_threshold(y_val, predict_proba_positive(stacking, X_val))
        rows.append(metrics_to_row("Stacking ensemble", "val", evaluate_binary_classifier(stacking, X_val, y_val, threshold), threshold))
        rows.append(metrics_to_row("Stacking ensemble", "test", evaluate_binary_classifier(stacking, X_test, y_test, threshold), threshold))
        fitted_models["Stacking ensemble"] = stacking

    results = pd.DataFrame(rows).sort_values(["split", "pr_auc"], ascending=[True, False])
    val_results = results[results["split"] == "val"].sort_values("pr_auc", ascending=False)
    best_name = str(val_results.iloc[0]["model"])
    best_model = fitted_models[best_name]

    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)
    results.to_csv(MODELS_DIR / "experiment_results.csv", index=False)
    print(f"Saved best model to {model_path}")
    print(f"Saved experiment table to {MODELS_DIR / 'experiment_results.csv'}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, default=None, help="Path to weatherAUS.csv or dataset folder")
    parser.add_argument("--quick", action="store_true", help="Run shorter experiment for CI/Docker smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(csv_path=args.csv_path, quick=args.quick)
