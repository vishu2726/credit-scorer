"""
Evaluate the trained model on the held-out test set and produce the
Kaggle submission file.

The held-out test set (20% of the labelled data) was carved out in
train.py BEFORE any training or tuning and identified by SK_ID_CURR
in the model bundle. The model sees it here for the first time.

Kaggle's own test set (application_test.csv) has private labels, so the
AUC printed below is computed on our internal held-out split; the Kaggle
leaderboard score provides independent confirmation.

Usage:
    python predict.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

from preprocess import make_datasets, get_feature_columns

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "model"


def main():
    bundle = joblib.load(RESULTS_DIR / "my_own_model.pkl")
    model = bundle["model"]
    features = bundle["features"]
    test_ids = set(bundle["test_ids"])

    train_full, kaggle_test = make_datasets()

    # Sanity check: the pipeline must produce the features the model expects.
    assert features == get_feature_columns(train_full), \
        "Feature mismatch between saved model and current preprocessing!"

    # Rebuild the exact held-out test set from the saved client IDs.
    held_out = train_full[train_full["SK_ID_CURR"].isin(test_ids)]

    proba = model.predict_proba(held_out[features])[:, 1]
    auc = roc_auc_score(held_out["TARGET"], proba)
    print(f"AUC on test set: {auc:.2f}")

    # Kaggle submission on the real unlabelled test set.
    kaggle_proba = model.predict_proba(kaggle_test[features])[:, 1]
    submission = pd.DataFrame({
        "SK_ID_CURR": kaggle_test["SK_ID_CURR"],
        "TARGET": kaggle_proba,
    })
    out_path = RESULTS_DIR / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"Kaggle submission saved to {out_path} ({len(submission)} rows)")


if __name__ == "__main__":
    main()