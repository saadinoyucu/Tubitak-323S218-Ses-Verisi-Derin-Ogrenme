from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "Results-And-Graph-2"
OUTPUT_DIR = BASE_DIR / "Grafikler"
FLOAT_FORMAT = "%.2f"
METRIC_COLUMNS = ["accuracy", "f1", "precision", "recall", "roc_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "f1": "F1 Score",
    "precision": "Precision",
    "recall": "Recall",
    "roc_auc": "ROC-AUC",
}
FOCUSED_METRICS = ["f1", "roc_auc"]

DATASET_LABELS = {"tat": "TAT", "spontane": "Spontaneous"}
FEATURE_LABELS = {
    "googlehear": "Google-HeAR",
    "hubert": "HuBERT",
    "mfcc": "MFCC",
    "wav2vec": "Wav2Vec",
}
MODEL_LABELS = {
    "TCN": "TCN",
    "TCN_LSTM": "TCN-LSTM",
    "CNN_LSTM": "CNN-LSTM",
}
SCENARIO_ORDER = [
    ("tat", "tat"),
    ("tat", "spontane"),
    ("spontane", "tat"),
    ("spontane", "spontane"),
]


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (12, 7)
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["savefig.dpi"] = 200


def reset_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_combined_results() -> pd.DataFrame:
    frames = []
    for csv_path in sorted(RESULTS_DIR.glob("*/results/summary/all_models_comparison.csv")):
        feature = csv_path.parts[-4]
        df = pd.read_csv(csv_path)
        df["feature"] = feature
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No all_models_comparison.csv files were found.")

    combined = pd.concat(frames, ignore_index=True)
    combined["feature_label"] = combined["feature"].map(FEATURE_LABELS).fillna(combined["feature"])
    combined["model_label"] = combined["model"].map(MODEL_LABELS).fillna(combined["model"])
    combined["train_label"] = combined["train_scenario"].map(DATASET_LABELS)
    combined["test_label"] = combined["test_dataset"].map(DATASET_LABELS)
    combined["scenario_label"] = (
        "Train: " + combined["train_label"] + " | Test: " + combined["test_label"]
    )
    return combined


def build_average_metric_table(
    df: pd.DataFrame, group_col: str, label_col: str, key_name: str, label_name: str
) -> pd.DataFrame:
    summary = (
        df.groupby([group_col, label_col])[METRIC_COLUMNS]
        .mean()
        .reset_index()
        .rename(columns={group_col: key_name, label_col: label_name})
    )
    summary["overall_mean"] = summary[METRIC_COLUMNS].mean(axis=1)
    summary = summary.sort_values("overall_mean", ascending=False).reset_index(drop=True)
    return summary


def plot_average_metric_ranking(
    summary_df: pd.DataFrame,
    entity_col: str,
    title: str,
    output_name: str,
) -> None:
    plot_df = summary_df[[entity_col] + METRIC_COLUMNS].melt(
        id_vars=entity_col,
        var_name="Metric",
        value_name="Score",
    )
    plot_df["Metric"] = plot_df["Metric"].map(METRIC_LABELS)

    plt.figure(figsize=(16, 9))
    ax = sns.barplot(
        data=plot_df,
        x=entity_col,
        y="Score",
        hue="Metric",
        palette="Set2",
    )
    ax.set_title(title, pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("Average Score")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(
        title="Metric",
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
    )

    best_name = summary_df.iloc[0][entity_col]
    best_score = summary_df.iloc[0]["overall_mean"]
    ax.text(
        0.01,
        0.02,
        f"Best: {best_name} ({best_score:.2f})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    for container in ax.containers:
        ax.bar_label(container, fmt=FLOAT_FORMAT, padding=2, fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, bbox_inches="tight")
    plt.close()


def plot_single_metric_ranking(
    summary_df: pd.DataFrame,
    entity_col: str,
    metric_col: str,
    title: str,
    output_name: str,
) -> None:
    plot_df = summary_df[[entity_col, metric_col]].sort_values(metric_col, ascending=False)

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        data=plot_df,
        x=entity_col,
        y=metric_col,
        hue=entity_col,
        dodge=False,
        palette="Set2",
        legend=False,
    )
    ax.set_title(title, pad=14)
    ax.set_xlabel("")
    ax.set_ylabel(METRIC_LABELS[metric_col])
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=0)

    best_name = plot_df.iloc[0][entity_col]
    best_score = plot_df.iloc[0][metric_col]
    ax.text(
        0.01,
        0.02,
        f"Best: {best_name} ({best_score:.2f})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    for container in ax.containers:
        ax.bar_label(container, fmt=FLOAT_FORMAT, padding=2, fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, bbox_inches="tight")
    plt.close()


def save_average_tables(feature_summary: pd.DataFrame, model_summary: pd.DataFrame) -> None:
    feature_summary.to_csv(
        OUTPUT_DIR / "feature_average_metrics.csv",
        index=False,
        float_format=FLOAT_FORMAT,
    )
    model_summary.to_csv(
        OUTPUT_DIR / "model_average_metrics.csv",
        index=False,
        float_format=FLOAT_FORMAT,
    )


def save_all_scenario_results(df: pd.DataFrame) -> None:
    export_df = (
        df[
            [
                "feature",
                "feature_label",
                "model",
                "model_label",
                "train_scenario",
                "train_label",
                "test_dataset",
                "test_label",
                "scenario_label",
                "accuracy",
                "f1",
                "precision",
                "recall",
                "roc_auc",
                "n_samples",
                "n_patient",
                "n_control",
            ]
        ]
        .copy()
        .rename(
            columns={
                "feature": "feature_key",
                "feature_label": "feature_extractor",
                "model": "model_key",
                "model_label": "model",
                "train_scenario": "train_scenario_key",
                "train_label": "train_scenario",
                "test_dataset": "test_set_key",
                "test_label": "test_set",
                "scenario_label": "scenario",
                "f1": "f1_score",
            }
        )
        .sort_values(
            ["feature_extractor", "model", "train_scenario_key", "test_set_key"]
        )
        .reset_index(drop=True)
    )
    export_df.to_csv(
        OUTPUT_DIR / "all_scenario_results.csv",
        index=False,
        float_format=FLOAT_FORMAT,
    )


def build_scenario_metric_table(
    df: pd.DataFrame,
    train_key: str,
    test_key: str,
    group_col: str,
    label_col: str,
    key_name: str,
    label_name: str,
) -> pd.DataFrame:
    scenario_df = df[
        (df["train_scenario"] == train_key) & (df["test_dataset"] == test_key)
    ].copy()
    if scenario_df.empty:
        raise ValueError(f"No rows found for scenario train={train_key}, test={test_key}.")

    summary = (
        scenario_df.groupby([group_col, label_col])[METRIC_COLUMNS]
        .mean()
        .reset_index()
        .rename(columns={group_col: key_name, label_col: label_name})
    )
    summary["overall_mean"] = summary[METRIC_COLUMNS].mean(axis=1)
    summary = summary.sort_values("overall_mean", ascending=False).reset_index(drop=True)
    return summary


def build_filtered_metric_table(
    df: pd.DataFrame,
    filters: dict[str, str],
    group_col: str,
    label_col: str,
    key_name: str,
    label_name: str,
) -> pd.DataFrame:
    filtered_df = df.copy()
    for column, value in filters.items():
        filtered_df = filtered_df[filtered_df[column] == value]

    if filtered_df.empty:
        raise ValueError(f"No rows found for filters: {filters}")

    summary = (
        filtered_df.groupby([group_col, label_col])[METRIC_COLUMNS]
        .mean()
        .reset_index()
        .rename(columns={group_col: key_name, label_col: label_name})
    )
    summary["overall_mean"] = summary[METRIC_COLUMNS].mean(axis=1)
    summary = summary.sort_values("overall_mean", ascending=False).reset_index(drop=True)
    return summary


def get_best_pair_subset(df: pd.DataFrame, feature_key: str, model_key: str) -> pd.DataFrame:
    return (
        df[(df["feature"] == feature_key) & (df["model"] == model_key)]
        .copy()
        .sort_values(["train_scenario", "test_dataset"])
        .reset_index(drop=True)
    )


def build_synthetic_roc_curve(auc_value: float, n_points: int = 300) -> tuple[np.ndarray, np.ndarray]:
    auc_value = float(np.clip(auc_value, 0.5001, 0.9999))
    exponent = (1.0 / auc_value) - 1.0
    fpr = np.linspace(0.0, 1.0, n_points)
    tpr = np.power(fpr, exponent)
    fpr[0], tpr[0] = 0.0, 0.0
    fpr[-1], tpr[-1] = 1.0, 1.0
    return fpr, tpr


def plot_reconstructed_roc_panel(
    best_pair_df: pd.DataFrame,
    feature_key: str,
    model_key: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    scenario_map = {
        (row["train_scenario"], row["test_dataset"]): row for _, row in best_pair_df.iterrows()
    }

    for index, (train_key, test_key) in enumerate(SCENARIO_ORDER):
        ax = axes[index]
        row = scenario_map.get((train_key, test_key))
        ax.set_title(
            f"Train: {DATASET_LABELS[train_key]} | Test: {DATASET_LABELS[test_key]}",
            fontsize=11,
        )

        if row is None:
            ax.text(0.5, 0.5, "Scenario not found", ha="center", va="center", fontsize=12)
            ax.axis("off")
            continue

        auc_value = float(row["roc_auc"])
        fpr, tpr = build_synthetic_roc_curve(auc_value)

        ax.plot(fpr, tpr, color="#2a7fdb", linewidth=2.5, label=f"AUC = {auc_value:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        (
            "Reconstructed ROC-AUC Curves: "
            f"{FEATURE_LABELS.get(feature_key, feature_key)} + {MODEL_LABELS.get(model_key, model_key)}"
        ),
        fontsize=18,
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "best_pair_roc_auc_panel.png", bbox_inches="tight")
    plt.close(fig)


def reconstruct_confusion_counts(row: pd.Series) -> tuple[int, int, int, int]:
    positives = int(row["n_patient"])
    negatives = int(row["n_control"])
    recall = float(row["recall"])
    precision = float(row["precision"])

    tp = round(recall * positives)
    fn = positives - tp
    predicted_positive = round(tp / precision) if precision > 0 else 0
    fp = max(0, predicted_positive - tp)
    tn = negatives - fp

    if tn < 0:
        fp = negatives
        tn = 0

    return tn, fp, fn, tp


def plot_reconstructed_confusion_panel(
    best_pair_df: pd.DataFrame,
    feature_key: str,
    model_key: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    scenario_map = {
        (row["train_scenario"], row["test_dataset"]): row for _, row in best_pair_df.iterrows()
    }

    for index, (train_key, test_key) in enumerate(SCENARIO_ORDER):
        ax = axes[index]
        row = scenario_map.get((train_key, test_key))
        ax.set_title(
            f"Train: {DATASET_LABELS[train_key]} | Test: {DATASET_LABELS[test_key]}",
            fontsize=11,
        )

        if row is None:
            ax.text(0.5, 0.5, "Scenario not found", ha="center", va="center", fontsize=12)
            ax.axis("off")
            continue

        tn, fp, fn, tp = reconstruct_confusion_counts(row)
        matrix = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["True Control", "True Patient"],
            columns=["Pred Control", "Pred Patient"],
        )
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle(
        (
            "Reconstructed Confusion Matrices: "
            f"{FEATURE_LABELS.get(feature_key, feature_key)} + {MODEL_LABELS.get(model_key, model_key)}"
        ),
        fontsize=18,
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "best_pair_confusion_panel.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    reset_output_dir()
    df = load_combined_results()
    feature_summary = build_average_metric_table(
        df,
        group_col="feature",
        label_col="feature_label",
        key_name="feature_key",
        label_name="feature_extractor",
    )
    model_summary = build_average_metric_table(
        df,
        group_col="model",
        label_col="model_label",
        key_name="model_key",
        label_name="model",
    )

    plot_average_metric_ranking(
        feature_summary,
        entity_col="feature_extractor",
        title="Average Metric Comparison of Feature Extractors",
        output_name="best_feature_extractor_metrics.png",
    )
    plot_average_metric_ranking(
        model_summary,
        entity_col="model",
        title="Average Metric Comparison of Models",
        output_name="best_model_metrics.png",
    )

    for metric in FOCUSED_METRICS:
        plot_single_metric_ranking(
            feature_summary,
            entity_col="feature_extractor",
            metric_col=metric,
            title=f"{METRIC_LABELS[metric]} Comparison of Feature Extractors",
            output_name=f"feature_{metric}_ranking.png",
        )
        plot_single_metric_ranking(
            model_summary,
            entity_col="model",
            metric_col=metric,
            title=f"{METRIC_LABELS[metric]} Comparison of Models",
            output_name=f"model_{metric}_ranking.png",
        )

    spontaneous_feature_summary = build_scenario_metric_table(
        df,
        train_key="spontane",
        test_key="spontane",
        group_col="feature",
        label_col="feature_label",
        key_name="feature_key",
        label_name="feature_extractor",
    )
    spontaneous_model_summary = build_scenario_metric_table(
        df,
        train_key="spontane",
        test_key="spontane",
        group_col="model",
        label_col="model_label",
        key_name="model_key",
        label_name="model",
    )

    plot_average_metric_ranking(
        spontaneous_feature_summary,
        entity_col="feature_extractor",
        title="Spontaneous Train / Spontaneous Test Feature Comparison",
        output_name="spontaneous_feature_metrics.png",
    )
    plot_average_metric_ranking(
        spontaneous_model_summary,
        entity_col="model",
        title="Spontaneous Train / Spontaneous Test Model Comparison",
        output_name="spontaneous_model_metrics.png",
    )

    tat_to_spontaneous_feature_summary = build_scenario_metric_table(
        df,
        train_key="tat",
        test_key="spontane",
        group_col="feature",
        label_col="feature_label",
        key_name="feature_key",
        label_name="feature_extractor",
    )
    tat_to_spontaneous_model_summary = build_scenario_metric_table(
        df,
        train_key="tat",
        test_key="spontane",
        group_col="model",
        label_col="model_label",
        key_name="model_key",
        label_name="model",
    )

    plot_average_metric_ranking(
        tat_to_spontaneous_feature_summary,
        entity_col="feature_extractor",
        title="TAT Train / Spontaneous Test Feature Comparison",
        output_name="tat_train_spontaneous_test_feature_metrics.png",
    )
    plot_average_metric_ranking(
        tat_to_spontaneous_model_summary,
        entity_col="model",
        title="TAT Train / Spontaneous Test Model Comparison",
        output_name="tat_train_spontaneous_test_model_metrics.png",
    )

    googlehear_model_summary = build_filtered_metric_table(
        df,
        filters={"feature": "googlehear"},
        group_col="model",
        label_col="model_label",
        key_name="model_key",
        label_name="model",
    )
    plot_average_metric_ranking(
        googlehear_model_summary,
        entity_col="model",
        title="Average Model Comparison for Google-HeAR",
        output_name="googlehear_model_average_metrics.png",
    )

    for metric in FOCUSED_METRICS:
        plot_single_metric_ranking(
            googlehear_model_summary,
            entity_col="model",
            metric_col=metric,
            title=f"Google-HeAR Model Comparison by {METRIC_LABELS[metric]}",
            output_name=f"googlehear_model_{metric}_ranking.png",
        )

    cnn_lstm_feature_summary = build_filtered_metric_table(
        df,
        filters={"model": "CNN_LSTM"},
        group_col="feature",
        label_col="feature_label",
        key_name="feature_key",
        label_name="feature_extractor",
    )
    plot_average_metric_ranking(
        cnn_lstm_feature_summary,
        entity_col="feature_extractor",
        title="Average Feature Comparison for CNN-LSTM",
        output_name="cnn_lstm_feature_average_metrics.png",
    )
    for metric in FOCUSED_METRICS:
        plot_single_metric_ranking(
            cnn_lstm_feature_summary,
            entity_col="feature_extractor",
            metric_col=metric,
            title=f"CNN-LSTM Feature Comparison by {METRIC_LABELS[metric]}",
            output_name=f"cnn_lstm_feature_{metric}_ranking.png",
        )

    save_average_tables(feature_summary, model_summary)
    save_all_scenario_results(df)

    best_feature_key = feature_summary.iloc[0]["feature_key"]
    best_model_key = model_summary.iloc[0]["model_key"]
    best_pair_df = get_best_pair_subset(df, best_feature_key, best_model_key)
    if best_pair_df.empty:
        raise ValueError("No rows found for the selected best feature/model combination.")

    plot_reconstructed_roc_panel(best_pair_df, best_feature_key, best_model_key)
    plot_reconstructed_confusion_panel(best_pair_df, best_feature_key, best_model_key)

    print(f"Loaded {len(df)} rows from combined results.")
    print(f"Best feature extractor: {feature_summary.iloc[0]['feature_extractor']}")
    print(f"Best model: {model_summary.iloc[0]['model']}")


if __name__ == "__main__":
    main()
