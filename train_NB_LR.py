import pandas as pd
import numpy as np

# Loading the preprocessed dataset
df = pd.read_csv("preprocessed_airlines.csv")

# Computing P(Delay)
p_delay = df['Delay'].value_counts(normalize=True)

# Computing P(Feature | Delay) dynamically from raw data
def compute_CP(feature):
    """Compute conditional probabilities for Naïve Bayes."""
    counts = df.groupby(['Delay', feature]).size().unstack(fill_value=0)
    probs = counts.div(counts.sum(axis=1), axis=0)
    return probs

# Computing CPTs on the fly
p_delay_airline = compute_CP('Airline')
p_delay_airport_from = compute_CP('AirportFrom')
p_delay_airport_to = compute_CP('AirportTo')
p_delay_day_of_week = compute_CP('DayOfWeek')
p_delay_time = compute_CP('Time')
p_delay_length = compute_CP('Length')

# Function to compute log probability for prediction
def predict_nb_probability(row, threshold=0.45):  # Using the new threshold 0.45
    log_p_delay_1 = np.log(p_delay[1] + 1e-6)
    log_p_delay_0 = np.log(p_delay[0] + 1e-6)

    for feature, cpt in zip(['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek', 'Time', 'Length'],
                            [p_delay_airline, p_delay_airport_from, p_delay_airport_to,
                             p_delay_day_of_week, p_delay_time, p_delay_length]):
        if row[feature] in cpt.columns:
            log_p_delay_1 += np.log(cpt.loc[1, row[feature]] + 1e-6)
            log_p_delay_0 += np.log(cpt.loc[0, row[feature]] + 1e-6)

    # Convert log probabilities back to normal probabilities
    p_delay_1 = np.exp(log_p_delay_1)
    p_delay_0 = np.exp(log_p_delay_0)

    # Normalize to get P(Delay=1 | Features)
    prob_delay_1 = p_delay_1 / (p_delay_1 + p_delay_0)

    # Apply threshold for Naïve Bayes classification
    return 1 if prob_delay_1 > threshold else 0, prob_delay_1

# Apply the function to get probabilities for Logistic Regression
df[['NB_PredictedDelay', 'NB_Probability']] = df.apply(lambda row: predict_nb_probability(row, threshold=0.45), axis=1, result_type="expand")

def sigmoid(z):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, learning_rate=0.01, iterations=1500):
    """Train logistic regression using gradient descent."""
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0

    for _ in range(iterations):
        # Compute predictions
        linear_model = np.dot(X, weights) + bias
        y_pred = sigmoid(linear_model)

        # Compute gradients
        error = y_pred - y
        weight_gradient = np.dot(X.T, error) / m
        bias_gradient = np.sum(error) / m

        # Update weights and bias
        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

    return weights, bias

def predict_logistic_regression(X, weights, bias, threshold=0.45):
    """Make predictions using trained logistic regression."""
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)
    return [1 if p >= threshold else 0 for p in y_pred]

# Using NB probability estimate as input for LR
X = df[['NB_Probability']].values
y = df['Delay'].values  # Target variable

# Manually split dataset into training and testing sets (80/20 split)
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Train Logistic Regression
weights, bias = train_logistic_regression(X_train, y_train, learning_rate=0.01, iterations=1500)

# Make predictions using Logistic Regression (threshold = 0.45)
y_pred = predict_logistic_regression(X_test, weights, bias, threshold=0.45)
accuracy = np.mean(y_pred == y_test)
print(f"Hybrid NB + Logistic Regression Accuracy (Threshold=0.45): {accuracy:.4f}")

# Save predictions
df['Hybrid_PredictedDelay'] = predict_logistic_regression(X, weights, bias, threshold=0.45)
df.to_csv("predictions_NB_LR.csv", index=False)