import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

st.set_page_config(page_title="Clustering Results Dashboard", layout="wide")

# ===============================
# Results extracted from your PDF
# ===============================
results = {
    "Thyroid": {
        "K-Means": {
            "Validation": {"accuracy": 71.88, "cm": [[15, 0, 7], [1, 4, 0], [1, 0, 4]]},
            "Final Test": {"accuracy": 90.91, "cm": [[23, 0, 0], [3, 3, 0], [0, 0, 4]]},
        },
        "Fuzzy C-Means": {
            "Validation": {"accuracy": 84.38, "cm": [[22, 0, 0], [3, 2, 0], [2, 0, 3]]},
            "Final Test": {"accuracy": 84.85, "cm": [[23, 0, 0], [0, 6, 0], [2, 0, 2]]},
        },
        "Genetic Algorithm - Fuzzy C-Means": {
            "Validation": {"accuracy": 96.88, "cm": [[22, 0, 0], [0, 5, 0], [1, 0, 4]]},
            "Final Test": {"accuracy": 100.00, "cm": [[23, 0, 0], [0, 6, 0], [0, 0, 4]]},
        },
    },
    "Heart Disease": {
        "K-Means": {
            "Validation": {"accuracy": 81.11, "cm": [[51, 24], [5, 74]]},
            "Final Test": {"accuracy": 79.87, "cm": [[51, 24], [7, 72]]},
        },
        "Fuzzy C-Means": {
            "Validation": {"accuracy": 81.81, "cm": [[59, 16], [8, 71]]},
            "Final Test": {"accuracy": 80.51, "cm": [[57, 18], [12, 67]]},
        },
        "Genetic Algorithm - Fuzzy C-Means": {
            "Validation": {"accuracy": 80.52, "cm": [[62, 13], [13, 66]]},
            "Final Test": {"accuracy": 83.12, "cm": [[59, 16], [14, 65]]},
        },
    },
    "Diabetes": {
        "K-Means": {
            "Validation": {"accuracy": 69.57, "cm": [[63, 12], [23, 17]]},
            "Final Test": {"accuracy": 68.97, "cm": [[64, 11], [25, 16]]},
        },
        "Fuzzy C-Means": {
            "Train": {"accuracy": 68.16, "cm": [[234, 116], [55, 132]]},
            "Validation": {"accuracy": 65.22, "cm": [[46, 29], [11, 29]]},
            "Final Test": {"accuracy": 68.97, "cm": [[54, 21], [15, 26]]},
        },
        "Genetic Algorithm - Fuzzy C-Means": {
            "Validation": {"accuracy": 70.43, "cm": [[66, 9], [25, 15]]},
            "Final Test": {"accuracy": 71.55, "cm": [[66, 6], [27, 14]]},
        },
    },
    "Kidney Disease": {
        "K-Means": {
            "Validation": {"accuracy": 96.66, "cm": [[11, 1], [0, 18]]},
            "Final Test": {"accuracy": 100.00, "cm": [[12, 0], [0, 19]]},
        },
        "Fuzzy C-Means": {
            "Train": {"accuracy": 96.66, "cm": [[11, 1], [0, 18]]},
            "Final Test": {"accuracy": 100.00, "cm": [[12, 0], [0, 19]]},
        },
        "Genetic Algorithm - Fuzzy C-Means": {
            "Validation": {"accuracy": 96.67, "cm": [[11, 1], [0, 18]]},
            "Final Test": {"accuracy": 100.00, "cm": [[12, 0], [0, 19]]},
        },
    },
    "Liver": {
        "K-Means": {
            "Validation": {"accuracy": 71.26, "cm": [[62, 0], [25, 0]]},
            "Final Test": {"accuracy": 71.59, "cm": [[63, 0], [25, 0]]},
        },
        "Fuzzy C-Means": {
            "Validation": {"accuracy": 71.32, "cm": [[62, 0], [25, 0]]},
            "Final Test": {"accuracy": 71.59, "cm": [[63, 0], [25, 0]]},
        },
        "Genetic Algorithm - Fuzzy C-Means": {
            "Validation": {"accuracy": 70.11, "cm": [[62, 0], [25, 0]]},
            "Final Test": {"accuracy": 72.73, "cm": [[62, 2], [22, 3]]},
        },
    },
}

# ===============================
# Helper functions
# ===============================
def plot_confusion_matrix(cm, title="Confusion Matrix"):
    cm = np.array(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    return fig

def classification_report_from_cm(cm):
    cm = np.array(cm)
    y_true = []
    y_pred = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            y_true.extend([i] * cm[i, j])
            y_pred.extend([j] * cm[i, j])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    df_report = pd.DataFrame({
        "Class": [f"Class {i}" for i in range(len(precision))],
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Support": support
    })
    macro = df_report[["Precision", "Recall", "F1-Score"]].mean()
    weighted = (
        (df_report[["Precision", "Recall", "F1-Score"]]
         .T @ df_report["Support"]) / df_report["Support"].sum()
    )
    df_report = pd.concat([
        df_report,
        pd.DataFrame([["Macro Avg", macro["Precision"], macro["Recall"], macro["F1-Score"], df_report["Support"].sum()]], 
                     columns=df_report.columns),
        pd.DataFrame([["Weighted Avg", weighted["Precision"], weighted["Recall"], weighted["F1-Score"], df_report["Support"].sum()]], 
                     columns=df_report.columns),
    ], ignore_index=True)
    return df_report

# ===============================
# Sidebar Controls
# ===============================
st.sidebar.title("Controls")
dataset = st.sidebar.selectbox("Dataset", list(results.keys()))
methods = list(results[dataset].keys())
method = st.sidebar.selectbox("Method", methods)
splits = list(results[dataset][method].keys())
split = st.sidebar.selectbox("Split", splits)

# ===============================
# Main Dashboard
# ===============================
st.markdown(
    """
    <h1 style='text-align: center; font-size: 36px; color: #2c3e50;'>
        Comparative Study of K-Means, Fuzzy C-Means and Genetic Algorithm based clustering on Medical Dataset
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)


st.markdown(
    """
    <p style='text-align: center; font-size:20px; color: #2c3e50;'>
        Comparison of K-Means, Fuzzy C-Means, and Genetic Algorithm - Fuzzy C-Means across 5 medical datasets.
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

info = results[dataset][method][split]

# Dataset — Method — Split
st.markdown(
    f"<h2 style='color:#34495e;text-align:center;'><b>{dataset} Dataset</b> with <b>{method}</b> on <b>{split}</b> set</h2>",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)



# Accuracy & Total Samples
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        f"<p style='font-size:28px; color:#16a085;'><b>Accuracy (%) :</b> {info['accuracy']:.2f}</p>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

with col2:
    if "cm" in info and info["cm"] is not None:
        total_samples = np.sum(info["cm"])
        st.markdown(
            f"<p style='font-size:28px; color:#2980b9;'><b>Total Samples :</b> {total_samples}</p>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)


# Confusion Matrix
if "cm" in info and info["cm"] is not None:
    st.subheader("Confusion Matrix")
    fig = plot_confusion_matrix(info["cm"], title=f"{method} — {split}")
    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")
    try:
        df_report = classification_report_from_cm(info["cm"])
        st.dataframe(df_report, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate classification report: {e}")

# Method Comparison
st.markdown(
    f"<h3 style='color:#2c3e50;'>Comparison of K-Means, FCM and GA-FCM for {dataset}  Dataset ({split})</h3>",
    unsafe_allow_html=True
)
compare = []
for mname, mdata in results[dataset].items():
    if split in mdata:
        compare.append((mname, mdata[split]["accuracy"]))
if compare:
    df_compare = pd.DataFrame(compare, columns=["Method", "Accuracy (%)"])
    st.dataframe(df_compare.style.set_properties(**{'font-size': '16px'}), use_container_width=True)

# Overview Table
st.subheader(f"Overview for {dataset}  Dataset")
overview_rows = []
for mname, mdata in results[dataset].items():
    for sname, sinfo in mdata.items():
        overview_rows.append((mname, sname, sinfo["accuracy"]))
if overview_rows:
    df = pd.DataFrame(overview_rows, columns=["Method", "Split", "Accuracy (%)"])
    st.dataframe(df.style.set_properties(**{'font-size': '15px'}), use_container_width=True)
