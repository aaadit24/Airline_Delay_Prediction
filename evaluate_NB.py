import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

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

# Computing additional metrics
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
false_negative_rate = FN / (FN + TP) if (FN + TP) > 0 else 0

# Printing results with improved formatting
print("\n" + "="*75)
print("  NAÏVE BAYES MODEL PERFORMANCE EVALUATION (NOTE: UPDATED THRESHOLD = 0.45)")
print("="*75)

print("CLASSIFICATION METRICS:")
print(f"  Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision:    {precision:.4f} ({precision*100:.2f}%)")
print(f"  Recall:       {recall:.4f} ({recall*100:.2f}%)")
print(f"  F1 Score:     {f1_score:.4f}")
print(f"  Specificity:  {specificity:.4f} ({specificity*100:.2f}%)")

print("ERROR RATES:")
print(f"  False Positive Rate: {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)")
print(f"  False Negative Rate: {false_negative_rate:.4f} ({false_negative_rate*100:.2f}%)")

# Printing confusion matrix with improved formatting
print("CONFUSION MATRIX:")
conf_matrix = np.array([[TP, FN], [FP, TN]])
print("  Predicted →  | Delayed | On Time ")
print("  ————————————+—————————+——————————")
print(f"  Actual: Delayed  |  {TP:6d} |  {FN:6d}  ")
print(f"  Actual: On Time  |  {FP:6d} |  {TN:6d}  ")

# Analysis and insights
print("INSIGHTS & ANALYSIS:")
print("  • The model achieves moderate accuracy, correctly predicting")
print(f"    {accuracy*100:.2f}% of all flight outcomes.")
print(f"  • Of all flights predicted to be delayed, {precision*100:.2f}% actually were (precision).")
print(f"  • The model catches {recall*100:.2f}% of all actual delays (recall).")
print(f"  • There are {FN} flights with delays that the model failed to catch.")
print(f"  • The model correctly identifies {specificity*100:.2f}% of on-time flights.")

# Plotting the confusion matrix visually
plt.figure(figsize=(10, 8))
conf_matrix_percent = confusion_matrix(y_true, y_pred, normalize='true')
sns.heatmap(conf_matrix_percent, annot=True, fmt='.1%', cmap='Blues',
            xticklabels=['Predicted On Time', 'Predicted Delayed'],
            yticklabels=['Actually On Time', 'Actually Delayed'])
plt.title('Normalized Confusion Matrix - Naïve Bayes Model')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig('confusion_matrix_plot.png')
print("A visualization of the confusion matrix has been saved to 'confusion_matrix_plot.png'")

# Printing class distribution
print("CLASS DISTRIBUTION:")
print(f"  Actual delayed flights:    {np.sum(y_true == 1):6d} ({np.mean(y_true == 1)*100:.2f}%)")
print(f"  Actual on-time flights:    {np.sum(y_true == 0):6d} ({np.mean(y_true == 0)*100:.2f}%)")
print(f"  Predicted delayed flights: {np.sum(y_pred == 1):6d} ({np.mean(y_pred == 1)*100:.2f}%)")
print(f"  Predicted on-time flights: {np.sum(y_pred == 0):6d} ({np.mean(y_pred == 0)*100:.2f}%)")

# Analyzing the threshold effect
print("THRESHOLD ANALYSIS:")
print("  The current model uses a default threshold of 0.5 for classification.")
print("  Adjusting the threshold to 0.45 (as mentioned in the README) could:")
print("  • Increase recall (catch more actual delays)")
print("  • Decrease precision (more false positives)")
print("  • This tradeoff may be beneficial if the cost of missing delays is higher")
print("    than the cost of incorrectly predicting delays.")

print("\n" + "="*50 + "\n")