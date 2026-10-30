import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

st.set_page_config(
    page_title="Student Dropout Classification",
    page_icon="🎓",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TARGET = "Target"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Apply exactly the feature engineering used during notebook training."""
    x = data.copy()

    def safe_ratio(numerator, denominator):
        return numerator / denominator.replace(0, np.nan)

    x["FE_1st_approval_rate"] = safe_ratio(
        x["Curricular units 1st sem (approved)"],
        x["Curricular units 1st sem (enrolled)"],
    )
    x["FE_2nd_approval_rate"] = safe_ratio(
        x["Curricular units 2nd sem (approved)"],
        x["Curricular units 2nd sem (enrolled)"],
    )
    x["FE_1st_evaluation_rate"] = safe_ratio(
        x["Curricular units 1st sem (evaluations)"],
        x["Curricular units 1st sem (enrolled)"],
    )
    x["FE_2nd_evaluation_rate"] = safe_ratio(
        x["Curricular units 2nd sem (evaluations)"],
        x["Curricular units 2nd sem (enrolled)"],
    )
    x["FE_total_approved_units"] = (
        x["Curricular units 1st sem (approved)"]
        + x["Curricular units 2nd sem (approved)"]
    )
    x["FE_total_enrolled_units"] = (
        x["Curricular units 1st sem (enrolled)"]
        + x["Curricular units 2nd sem (enrolled)"]
    )
    x["FE_average_semester_grade"] = (
        x["Curricular units 1st sem (grade)"]
        + x["Curricular units 2nd sem (grade)"]
    ) / 2
    x["FE_grade_change"] = (
        x["Curricular units 2nd sem (grade)"]
        - x["Curricular units 1st sem (grade)"]
    )
    x["FE_overall_approval_rate"] = (
        x["FE_total_approved_units"]
        / (x["FE_total_enrolled_units"] + 1)
    )

    risk = pd.Series(0, index=x.index, dtype=float)
    if "Debtor" in x.columns:
        risk += (x["Debtor"] == 1).astype(int)
    if "Tuition fees up to date" in x.columns:
        risk += (x["Tuition fees up to date"] == 0).astype(int)
    x["FE_financial_admin_risk"] = risk

    return x.replace([np.inf, -np.inf], np.nan)


st.title("Student Dropout Classification")
st.caption(
    "Streamlit deployment for the five trained classifiers required by the assignment."
)
st.info(
    "Upload TEST data only. This app loads saved trained models and does not retrain "
    "or fit anything on the uploaded file."
)

required_features = [
    "Marital Status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification",
    "Previous qualification (grade)", "Nacionality", "Mother's qualification",
    "Father's qualification", "Mother's occupation", "Father's occupation",
    "Admission grade", "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder",
    "Age at enrollment", "International",
    "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP"
]

class_names = ["Dropout", "Enrolled", "Graduate"]
label_to_int = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}

uploaded_file = st.file_uploader(
    "Upload the held-out test CSV",
    type=["csv"],
    help="The CSV must contain all 36 original input features and the Target column.",
)

model_name = st.selectbox(
    "Select a trained classification model",
    list(MODEL_FILES.keys()),
)

if uploaded_file is None:
    st.warning("Please upload test_data.csv to continue.")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

if TARGET not in data.columns:
    st.error("The uploaded CSV must contain the target column named 'Target'.")
    st.stop()

missing = [column for column in required_features if column not in data.columns]
if missing:
    st.error("The uploaded CSV is missing required input columns:")
    st.write(missing)
    st.stop()

unknown_targets = sorted(set(data[TARGET].astype(str)) - set(label_to_int))
if unknown_targets:
    st.error(
        "The Target column contains unknown class labels. "
        f"Expected: {class_names}. Found: {unknown_targets}"
    )
    st.stop()

st.subheader("Uploaded test-data validation")
validation_cols = st.columns(4)
validation_cols[0].metric("Rows", f"{len(data):,}")
validation_cols[1].metric("Input features", len(required_features))
validation_cols[2].metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
validation_cols[3].metric("Duplicate rows", f"{int(data.duplicated().sum()):,}")

with st.expander("Preview uploaded test data"):
    st.dataframe(data.head(10), use_container_width=True)

# Convert target labels to the numeric labels used when the models were trained.
y = data[TARGET].astype(str).map(label_to_int).astype(int)

# Apply the same row-wise feature engineering used during training.
X = add_features(data[required_features])

model_path = MODEL_DIR / MODEL_FILES[model_name]
if not model_path.exists():
    st.error(
        f"Saved model not found: {model_path.name}. "
        "Run the notebook first so all five .joblib files are created."
    )
    st.stop()

model = joblib.load(model_path)

try:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    classes = np.asarray(model.named_steps["model"].classes_)

    # The assignment requires multiclass macro one-vs-rest AUC.
    if len(np.unique(y)) < len(classes):
        raise ValueError(
            "AUC cannot be calculated because the uploaded test file does not "
            "contain all three target classes."
        )

    auc_score = roc_auc_score(
        y,
        probabilities,
        multi_class="ovr",
        average="macro",
        labels=classes,
    )

    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": auc_score,
        "Precision": precision_score(
            y, predictions, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, average="macro", zero_division=0
        ),
        "F1 Score": f1_score(
            y, predictions, average="macro", zero_division=0
        ),
        "MCC": matthews_corrcoef(y, predictions),
    }

    st.subheader(f"Evaluation results — {model_name}")
    metric_cols = st.columns(6)
    for column, (metric_name, value) in zip(metric_cols, metrics.items()):
        column.metric(metric_name, f"{value:.4f}")

    st.caption(
        "Precision, Recall, and F1 are macro averages. AUC is macro "
        "one-vs-rest multiclass ROC-AUC."
    )

    st.subheader("Classification Report")
    report = classification_report(
        y,
        predictions,
        labels=classes,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    st.dataframe(
        pd.DataFrame(report).transpose().round(4),
        use_container_width=True,
    )

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, predictions, labels=classes)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

except Exception as exc:
    st.error(f"Could not evaluate the selected model: {exc}")
