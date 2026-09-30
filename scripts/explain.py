"""
Model interpretability - part 1: global feature importance.

Answers the question "what are the key variables to the model overall?"
(global transparency level from the project README).

The feature names come from the model bundle saved by train.py - the
exact list the model was fitted on - so the importance <-> name mapping
cannot drift out of sync with the model.

Usage:
    python explain.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import make_datasets, get_feature_columns, DATA_DIR

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODEL_DIR = RESULTS_DIR / "model"

# Raw variables shown in the client dashboards (audit: >= 10, raw values)
DESCRIPTIVE_CATEGORICAL = [
    ("CODE_GENDER", "Gender"),
    ("NAME_CONTRACT_TYPE", "Contract type"),
    ("NAME_EDUCATION_TYPE", "Education"),
    ("NAME_FAMILY_STATUS", "Family status"),
    ("NAME_INCOME_TYPE", "Income type"),
    ("OCCUPATION_TYPE", "Occupation"),
    ("CNT_CHILDREN", "Number of children"),
]
DESCRIPTIVE_NUMERIC = [
    ("AGE_YEARS", "Age (years)"),
    ("EMPLOYED_YEARS", "Years employed"),
    ("AMT_INCOME_TOTAL", "Annual income"),
    ("AMT_CREDIT", "Credit amount"),
    ("AMT_ANNUITY", "Loan annuity"),
    ("AMT_GOODS_PRICE", "Goods price"),
    ("EXT_SOURCE_2", "External score 2"),
    ("EXT_SOURCE_3", "External score 3"),
]

def load_bundle():
    return joblib.load(MODEL_DIR / "my_own_model.pkl")


def compute_feature_importance(bundle):
    """Gain importance of every feature the model was trained on.

    LightGBM's sklearn API guarantees feature_importances_[i] corresponds
    to the i-th training column, i.e. bundle['features'][i].
    """
    model = bundle["model"]
    importance = pd.DataFrame({
        "feature": bundle["features"],
        "importance": model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return importance


def plot_feature_importance(importance, out_path, top_n=25):
    top = importance.head(top_n).iloc[::-1]  # reversed so #1 sits on top

    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(top["feature"], top["importance"], color="#4c72b0")
    ax.set_xlabel("Gain importance")
    ax.set_title(f"Top {top_n} features by gain - global model transparency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Feature importance plot saved to {out_path}")

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def select_clients(model, features, train_full, test_ids, kaggle_test):
    """Pick the 3 clients required by the project.

    - correct: held-out client, true defaulter, confidently caught
    - wrong:   held-out client predicted to default who actually repaid
    - test:    a Kaggle test client with rich data (few missing values)

    Both labelled clients come from the held-out test split (never seen
    in training), so 'correct' and 'wrong' reflect real generalization.
    """
    held_out = train_full[train_full["SK_ID_CURR"].isin(test_ids)].copy()
    held_out["proba"] = model.predict_proba(held_out[features])[:, 1]

    defaulters = held_out[held_out["TARGET"] == 1]
    client_correct = defaulters.loc[[defaulters["proba"].idxmax()]]

    repayers = held_out[held_out["TARGET"] == 0]
    client_wrong = repayers.loc[[repayers["proba"].idxmax()]]

    kt = kaggle_test.copy()
    kt["n_missing"] = kt[features].isna().sum(axis=1)
    client_test = kt.loc[[kt["n_missing"].idxmin()]]

    return client_correct, client_wrong, client_test


def shap_force_plot(model, features, client_row, title, out_path):
    """SHAP force plot for one client, with the log-odds -> probability
    conversion made explicit and verified against the model's output."""
    explainer = shap.TreeExplainer(model.booster_)
    x = client_row[features]  # one-row DataFrame, dtypes intact
    shap_values = explainer.shap_values(x)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    sv = sv.reshape(-1)
    base = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

    # The audit check: force plot output == model output
    log_odds = base + sv.sum()
    proba_from_shap = sigmoid(log_odds)
    proba_from_model = model.predict_proba(x)[0, 1]
    assert abs(proba_from_shap - proba_from_model) < 1e-6, \
        f"SHAP ({proba_from_shap}) != model ({proba_from_model})"

    print(f"  base value (log-odds):        {base:+.4f}")
    print(f"  sum of SHAP contributions:    {sv.sum():+.4f}")
    print(f"  total (log-odds):             {log_odds:+.4f}")
    print(f"  sigmoid(total) = probability: {proba_from_shap:.4f}")
    print(f"  model.predict_proba:          {proba_from_model:.4f}  ✓ identical")

    fig = shap.force_plot(
        base, sv, x.iloc[0], matplotlib=True, show=False, figsize=(20, 4),
        contribution_threshold=0.12,   # label only features driving >=12% of the push
        text_rotation=30,              # slant labels so neighbours don't collide
    )
    plt.title(title)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  force plot saved to {out_path}")

def load_raw_applications():
    """Raw application tables for the descriptive dashboards.

    The audit requires RAW values here (real income, real age), not the
    model-transformed features. Days are converted to years purely for
    readability; the underlying values are untouched raw data.
    """
    raw_train = pd.read_csv(DATA_DIR / "application_train.csv")
    raw_test = pd.read_csv(DATA_DIR / "application_test.csv")
    for df in (raw_train, raw_test):
        df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
        df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"].replace(365243, np.nan) / 365.25
    return raw_train, raw_test


def client_dashboard(raw_row, population, proba, title):
    """Plotly figure: client facts table + 8 numeric variables compared
    to the whole client population (red line = this client)."""
    fig = make_subplots(
        rows=3, cols=4,
        specs=[[{"type": "table", "colspan": 4}, None, None, None],
               [{}, {}, {}, {}],
               [{}, {}, {}, {}]],
        subplot_titles=[""] + [label for _, label in DESCRIPTIVE_NUMERIC],
        vertical_spacing=0.09,
    )

    fields = ["Predicted default probability"]
    values = [f"{proba:.1%}"]
    for col, label in DESCRIPTIVE_CATEGORICAL:
        fields.append(label)
        v = raw_row[col]
        values.append("not provided" if pd.isna(v) else str(v))

    fig.add_trace(go.Table(
        header=dict(values=["Client information", "Value"],
                    fill_color="#4c72b0", font=dict(color="white", size=13)),
        cells=dict(values=[fields, values], height=24),
    ), row=1, col=1)

    positions = [(2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)]
    for i, ((col, label), (r, c)) in enumerate(zip(DESCRIPTIVE_NUMERIC, positions)):
        pop = population[col].dropna()
        pop = pop[pop <= pop.quantile(0.99)]  # clip extreme outliers for readable axes
        fig.add_trace(go.Histogram(x=pop, nbinsx=40, marker_color="#4c72b0",
                                   showlegend=False), row=r, col=c)
        v = raw_row[col]
        if not pd.isna(v):
            # add_vline breaks when the figure also contains a go.Table
            # (known plotly bug), so draw the line manually on this
            # subplot's own axes. Histograms own axes x1..x8 in order.
            axis = "" if i == 0 else str(i + 1)
            fig.add_shape(
                type="line",
                x0=float(v), x1=float(v), y0=0, y1=1,
                xref=f"x{axis}", yref=f"y{axis} domain",
                line=dict(color="#c44e52", width=3),
            )

    fig.update_layout(height=1000, width=1400, title_text=title,
                      title_font_size=18, showlegend=False)
    return fig


def assemble_pdf(png_paths, out_path):
    """Combine PNG pages (force plot + dashboard) into one PDF deliverable."""
    with PdfPages(out_path) as pdf:
        for png in png_paths:
            img = mpimg.imread(png)
            h, w = img.shape[:2]
            fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
    print(f"  assembled {out_path}")

def main():
    bundle = load_bundle()
    model = bundle["model"]
    features = bundle["features"]

    # --- Global interpretability ---
    importance = compute_feature_importance(bundle)
    n_unused = (importance["importance"] == 0).sum()
    print(f"Features in model: {len(importance)} "
          f"({n_unused} never used by any split, importance = 0)")
    print("\nTop 15 features by gain:")
    print(importance.head(15).to_string(index=False))
    plot_feature_importance(importance, MODEL_DIR / "feature_importance.png")
    importance.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    # --- Local interpretability: the 3 required clients ---
    print("\nLoading data to select the 3 clients...")
    train_full, kaggle_test = make_datasets()

    client_correct, client_wrong, client_test = select_clients(
        model, features, train_full, set(bundle["test_ids"]), kaggle_test
    )

    out_dir = RESULTS_DIR / "clients_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for row, label, fname in [
        (client_correct, "correct (true default, caught)", "client1_correct_train_force.png"),
        (client_wrong, "wrong (false positive)", "client2_wrong_train_force.png"),
        (client_test, "test set client", "client_test_force.png"),
    ]:
        cid = int(row["SK_ID_CURR"].iloc[0])
        target = row["TARGET"].iloc[0] if "TARGET" in row.columns else None
        target_str = f", TARGET={int(target)}" if target is not None and not pd.isna(target) else ""
        print(f"\nClient {cid} - {label}{target_str}")
        shap_force_plot(model, features, row,
                        f"Client {cid} - {label}", out_dir / fname)

    # Persist the selection so the Plotly visualizations (next step)
    # and the final PDFs use the same clients.
    selection = pd.DataFrame({
        "role": ["correct_train", "wrong_train", "test"],
        "SK_ID_CURR": [int(client_correct["SK_ID_CURR"].iloc[0]),
                       int(client_wrong["SK_ID_CURR"].iloc[0]),
                       int(client_test["SK_ID_CURR"].iloc[0])],
    })
    selection.to_csv(out_dir / "selected_clients.csv", index=False)
    print(f"\nClient selection saved to {out_dir / 'selected_clients.csv'}")

    # --- Descriptive dashboards (raw values) and final PDFs ---
    print("\nBuilding descriptive dashboards from raw data...")
    raw_train, raw_test = load_raw_applications()

    jobs = [
        (client_correct, raw_train, "client1_correct_train", "correct prediction (train)"),
        (client_wrong, raw_train, "client2_wrong_train", "wrong prediction (train)"),
        (client_test, raw_test, "client_test", "test set client"),
    ]
    for row, raw_df, stem, label in jobs:
        cid = int(row["SK_ID_CURR"].iloc[0])
        proba = model.predict_proba(row[features])[0, 1]
        raw_row = raw_df.loc[raw_df["SK_ID_CURR"] == cid].iloc[0]

        fig = client_dashboard(raw_row, raw_train, proba,
                               f"Client {cid} - {label}")
        dash_png = out_dir / f"{stem}_dashboard.png"
        fig.write_image(str(dash_png), scale=2)

        assemble_pdf([out_dir / f"{stem}_force.png", dash_png],
                     out_dir / f"{stem}.pdf")


if __name__ == "__main__":
    main()