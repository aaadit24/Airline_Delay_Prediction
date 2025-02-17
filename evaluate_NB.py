import pandas as pd
import numpy as np

df_true = pd.read_csv("preprocessed_airlines.csv")
df_pred = pd.read_csv("predictions_NB.csv")

y_true = df_true['Delay'].values
y_pred = df_pred['PredictedDelay'].values

# Computing confusion matrix values
# True Positives
TP = np.sum((y_true == 1) & (y_pred == 1))
# True Negatives
TN = np.sum((y_true == 0) & (y_pred == 0))
# False Positives
FP = np.sum((y_true == 0) & (y_pred == 1))
# False Negatives
FN = np.sum((y_true == 1) & (y_pred == 0))

# Computing metrics manually
accuracy = (TP + TN) / len(y_true)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Printing results
print("\nNaïve Bayes Model Performance:\n")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# Printing confusion matrix
print("\nConfusion Matrix:")
print(f"TP: {TP}, FP: {FP}")
print(f"FN: {FN}, TN: {TN}")
