"""
Train the credit scoring model.

- 64/16/20 split: train / validation (early stopping) / held-out test.
- LightGBM with class weighting for the ~8% default rate.
- Early stopping on validation AUC to prevent overfitting.
- Saves the model bundle to results/model/my_own_model.pkl
- Saves the learning curve to results/model/learning_curve.png
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from preprocess import make_datasets, get_feature_columns

RANDOM_STATE = 42
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "model"


def split_data(train_df):
    """64/16/20 split, stratified so each part keeps the ~8% default rate."""
    train_val, test = train_test_split(
        train_df, test_size=0.20, stratify=train_df["TARGET"],
        random_state=RANDOM_STATE,
    )
    train, val = train_test_split(
        train_val, test_size=0.20, stratify=train_val["TARGET"],
        random_state=RANDOM_STATE,
    )
    return train, val, test


def train_model(train, val, features):
    """LightGBM with early stopping on validation AUC."""
    model = lgb.LGBMClassifier(
        n_estimators=5000,           # ceiling; early stopping decides the real number
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,       # regularization: no leaf smaller than 100 clients
        colsample_bytree=0.8,        # regularization: random 80% of features per tree
        subsample=0.8,               # regularization: random 80% of rows per tree
        subsample_freq=1,
        class_weight="balanced",     # EDA section 1: ~8% default rate
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        train[features], train["TARGET"],
        eval_set=[
            (train[features], train["TARGET"]),
            (val[features], val["TARGET"]),
        ],
        eval_names=["train", "validation"],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100),
        ],
    )
    return model


def plot_learning_curve(model, out_path):
    """Training vs validation AUC per boosting round - the overfitting evidence."""
    history = model.evals_result_
    rounds = range(1, len(history["train"]["auc"]) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rounds, history["train"]["auc"], label="Train AUC")
    ax.plot(rounds, history["validation"]["auc"], label="Validation AUC")
    ax.axvline(model.best_iteration_, color="gray", linestyle="--",
               label=f"Early stopping (round {model.best_iteration_})")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("AUC")
    ax.set_title("Learning curves - training stopped at the validation peak")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Learning curve saved to {out_path}")


def main():
    print("Loading and preprocessing data...")
    train_full, _ = make_datasets()
    features = get_feature_columns(train_full)

    train, val, test = split_data(train_full)
    print(f"Split sizes - train: {len(train)}, val: {len(val)}, test: {len(test)}")

    model = train_model(train, val, features)

    print(f"\nBest iteration: {model.best_iteration_}")
    print(f"Train AUC:      {roc_auc_score(train['TARGET'], model.predict_proba(train[features])[:, 1]):.4f}")
    print(f"Validation AUC: {roc_auc_score(val['TARGET'], model.predict_proba(val[features])[:, 1]):.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_learning_curve(model, RESULTS_DIR / "learning_curve.png")

    # Bundle everything predict.py and explain.py will need.
    # The held-out test is identified by SK_ID_CURR so it can never
    # leak into any future training run by accident.
    bundle = {
        "model": model,
        "features": features,
        "test_ids": test["SK_ID_CURR"].tolist(),
        "random_state": RANDOM_STATE,
    }
    joblib.dump(bundle, RESULTS_DIR / "my_own_model.pkl")
    print(f"Model bundle saved to {RESULTS_DIR / 'my_own_model.pkl'}")


if __name__ == "__main__":
    main()
    