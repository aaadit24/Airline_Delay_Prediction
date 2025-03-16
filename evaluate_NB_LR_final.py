import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load data
df_true = pd.read_csv("preprocessed_airlines.csv")  # Ground truth labels
df_pred = pd.read_csv("predictions_NB_LR.csv")  # Predicted delays

y_true = df_true['Delay'].values  # Actual delays
y_pred_nb = df_pred['NB_PredictedDelay'].values  # Naïve Bayes predictions
y_pred_hybrid = df_pred['Hybrid_PredictedDelay'].values  # Hybrid (NB + LR) predictions

# Compute Confusion Matrix Values
def compute_metrics(y_true, y_pred):
    """Computes confusion matrix and evaluation metrics."""
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (TP + TN) / len(y_true)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
    false_negative_rate = FN / (FN + TP) if (FN + TP) > 0 else 0

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score,
        "specificity": specificity, "false_positive_rate": false_positive_rate, "false_negative_rate": false_negative_rate
    }

# Compute metrics for both models
metrics_nb = compute_metrics(y_true, y_pred_nb)
metrics_hybrid = compute_metrics(y_true, y_pred_hybrid)

# Print Results
print("\n" + "="*80)
print("  MODEL PERFORMANCE EVALUATION (Naïve Bayes vs Hybrid NB + Logistic Regression)")
print("="*80)

def print_metrics(model_name, metrics):
    """Prints model performance metrics."""
    print(f"\n{model_name} Model Performance")
    print(f"  Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:    {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:       {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1 Score:     {metrics['f1_score']:.4f}")
    print(f"  Specificity:  {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")

    print("\nError Rates:")
    print(f"  False Positive Rate: {metrics['false_positive_rate']:.4f} ({metrics['false_positive_rate']*100:.2f}%)")
    print(f"  False Negative Rate: {metrics['false_negative_rate']:.4f} ({metrics['false_negative_rate']*100:.2f}%)")

    print("\nConfusion Matrix:")
    print("  Predicted →  | Delayed | On Time ")
    print("  ————————————+—————————+——————————")
    print(f"  Actual: Delayed  |  {metrics['TP']:6d} |  {metrics['FN']:6d}  ")
    print(f"  Actual: On Time  |  {metrics['FP']:6d} |  {metrics['TN']:6d}  ")

# Print Naïve Bayes metrics
print_metrics("Naïve Bayes (NB)", metrics_nb)

# Print Hybrid NB + LR metrics
print_metrics("Hybrid NB + Logistic Regression", metrics_hybrid)

# Insights and Analysis
print("\nInsights & Analysis:")
if metrics_hybrid["accuracy"] > metrics_nb["accuracy"]:
    print(f"  The Hybrid NB + Logistic Regression model outperforms Naïve Bayes, improving accuracy from {metrics_nb['accuracy']*100:.2f}% to {metrics_hybrid['accuracy']*100:.2f}%.")
if metrics_hybrid["recall"] > metrics_nb["recall"]:
    print(f"  The Hybrid model also catches more actual delays (Recall: {metrics_nb['recall']*100:.2f}% → {metrics_hybrid['recall']*100:.2f}%).")
if metrics_hybrid["false_positive_rate"] < metrics_nb["false_positive_rate"]:
    print(f"  The Hybrid model reduces false alarms (FPR: {metrics_nb['false_positive_rate']*100:.2f}% → {metrics_hybrid['false_positive_rate']*100:.2f}%).")

print("\nThreshold Impact:")
print("  • The model uses a threshold of 0.45, balancing recall and precision.")
print("  • Lowering the threshold might increase recall further but reduce precision.")
print("  • Further tuning could explore thresholds between 0.40 and 0.50 for optimal trade-off.")

print("\n" + "="*80 + "\n")

# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Generates a heatmap for the confusion matrix."""
    conf_matrix = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='.1%', cmap='Blues',
                xticklabels=['Predicted On Time', 'Predicted Delayed'],
                yticklabels=['Actually On Time', 'Actually Delayed'])
    plt.title(title)
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"A visualization of the confusion matrix has been saved as '{filename}'")

# Plot confusion matrices
plot_confusion_matrix(y_true, y_pred_nb, "Naïve Bayes - Normalized Confusion Matrix", "confusion_matrix_nb.png")
plot_confusion_matrix(y_true, y_pred_hybrid, "Hybrid NB + Logistic Regression - Normalized Confusion Matrix", "confusion_matrix_hybrid.png")