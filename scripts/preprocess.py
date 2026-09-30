"""
Preprocessing and feature engineering for the credit scoring model.

Single source of truth: train.py and predict.py both import make_datasets()
so the exact same pipeline is applied everywhere.

All decisions implemented here are justified in results/feature_engineering/EDA.ipynb:
- DAYS_EMPLOYED sentinel 365243 -> NaN + anomaly flag
- EXT_SOURCE combination features
- Domain ratio features (credit/income, annuity/income, ...)
- Per-client aggregation of bureau.csv and previous_application.csv
- Categoricals cast to 'category' dtype for LightGBM native handling
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DAYS_EMPLOYED_SENTINEL = 365243


def clean_application(df):
    """Fix known data anomalies in an application table (train or test).

    These are fixed rules, not statistics learned from data, so applying
    them to both train and test carries no leakage risk.
    """
    df = df.copy()

    # EDA section 4: sentinel value means "not currently employed"
    df["DAYS_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_SENTINEL, np.nan)

    # 4 rows with unknown gender; make it explicit missing
    df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)

    return df


def add_application_features(df):
    """Feature engineering on application columns only (EDA sections 3-4)."""
    df = df.copy()

    # Age and employment length in human units
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25

    # EXT_SOURCE combinations - our strongest raw predictors
    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    df["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    df["EXT_SOURCE_MIN"] = ext.min(axis=1)
    df["EXT_SOURCE_COUNT_MISSING"] = ext.isna().sum(axis=1)

    # Domain ratios: how heavy is the loan relative to the client's means
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM_YEARS"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["EMPLOYED_AGE_RATIO"] = df["EMPLOYED_YEARS"] / df["AGE_YEARS"]

    return df


def aggregate_bureau(bureau):
    """One row per client summarising their credit history at other banks."""
    agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_CREDIT_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("CREDIT_ACTIVE", lambda s: (s == "Active").sum()),
        BUREAU_DAYS_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_DAYS_OVERDUE_MEAN=("CREDIT_DAY_OVERDUE", "mean"),
        BUREAU_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
        BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
        BUREAU_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
    )
    # Share of the client's total credit that is still debt
    agg["BUREAU_DEBT_CREDIT_RATIO"] = agg["BUREAU_DEBT_SUM"] / agg["BUREAU_CREDIT_SUM"].replace(0, np.nan)
    return agg.reset_index()


def aggregate_previous(prev):
    """One row per client summarising their past applications at Home Credit."""
    agg = prev.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_REFUSED_SHARE=("NAME_CONTRACT_STATUS", lambda s: (s == "Refused").mean()),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PREV_DAYS_DECISION_MEAN=("DAYS_DECISION", "mean"),
    )
    # Did they historically get less than they asked for?
    agg["PREV_GRANTED_RATIO"] = agg["PREV_AMT_CREDIT_MEAN"] / agg["PREV_AMT_APPLICATION_MEAN"].replace(0, np.nan)
    return agg.reset_index()


def build_features(app, bureau_agg, prev_agg):
    """Clean an application table, engineer features, join aggregates."""
    df = clean_application(app)
    df = add_application_features(df)
    df = df.merge(bureau_agg, on="SK_ID_CURR", how="left")
    df = df.merge(prev_agg, on="SK_ID_CURR", how="left")

    # LightGBM native categorical handling
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].astype("category")

    return df


def make_datasets(data_dir=DATA_DIR):
    """Load raw CSVs and return fully processed (train, test) DataFrames.

    train keeps TARGET; both keep SK_ID_CURR for identification.
    Feature columns and dtypes are guaranteed identical between the two.
    """
    data_dir = Path(data_dir)

    app_train = pd.read_csv(data_dir / "application_train.csv")
    app_test = pd.read_csv(data_dir / "application_test.csv")
    bureau = pd.read_csv(data_dir / "bureau.csv")
    prev = pd.read_csv(data_dir / "previous_application.csv")

    bureau_agg = aggregate_bureau(bureau)
    prev_agg = aggregate_previous(prev)

    train = build_features(app_train, bureau_agg, prev_agg)
    test = build_features(app_test, bureau_agg, prev_agg)

    # Align categorical categories between train and test so LightGBM
    # sees a consistent encoding (fit on train's world view).
    for col in train.select_dtypes("category").columns:
        if col in test.columns:
            test[col] = test[col].cat.set_categories(train[col].cat.categories)

    return train, test


def get_feature_columns(df):
    """Columns the model trains on: everything except identifiers and target."""
    return [c for c in df.columns if c not in ("SK_ID_CURR", "TARGET")]


if __name__ == "__main__":
    train, test = make_datasets()
    features = get_feature_columns(train)
    print(f"Train: {train.shape}, Test: {test.shape}")
    print(f"Feature columns: {len(features)}")
    print(f"Categorical features: {len(train[features].select_dtypes('category').columns)}")
    