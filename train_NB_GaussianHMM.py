import pandas as pd
import numpy as np

# Load the preprocessed dataset
df = pd.read_csv("preprocessed_airlines.csv")

# Convert 'Time' column (HH:MM) to minutes since midnight for numerical processing
df['Time'] = df['Time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

# Ensure 'Length' column is numeric
df['Length'] = df['Length'].astype(float)

# Compute P(Delay)
p_delay = df['Delay'].value_counts(normalize=True)

# Compute the Naïve Bayes probabilities
def compute_CP(feature):
    """Compute conditional probabilities for Naïve Bayes."""
    counts = df.groupby(['Delay', feature]).size().unstack(fill_value=0)
    probs = counts.div(counts.sum(axis=1), axis=0)
    return probs

# Compute Conditional Probability Tables (CPTs)
p_delay_airline = compute_CP('Airline')
p_delay_airport_from = compute_CP('AirportFrom')
p_delay_airport_to = compute_CP('AirportTo')
p_delay_day_of_week = compute_CP('DayOfWeek')

# Function to compute Naïve Bayes probability
def predict_nb_probability(row):
    """Compute probability P(Delay=1 | Features) using Naïve Bayes."""
    log_p_delay_1 = np.log(p_delay[1] + 1e-6)
    log_p_delay_0 = np.log(p_delay[0] + 1e-6)

    for feature, cpt in zip(['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek'],
                            [p_delay_airline, p_delay_airport_from, p_delay_airport_to, p_delay_day_of_week]):
        if row[feature] in cpt.columns:
            log_p_delay_1 += np.log(cpt.loc[1, row[feature]] + 1e-6)
            log_p_delay_0 += np.log(cpt.loc[0, row[feature]] + 1e-6)

    # Convert log probabilities back to normal probabilities
    p_delay_1 = np.exp(log_p_delay_1)
    p_delay_0 = np.exp(log_p_delay_0)

    # Normalize to get P(Delay=1 | Features)
    return p_delay_1 / (p_delay_1 + p_delay_0)

# Apply the function to compute probabilities and store in df
df['NB_Probability'] = df.apply(predict_nb_probability, axis=1)

# Convert NB Probability to a discrete class label based on threshold
df['NB_PredictedDelay'] = (df['NB_Probability'] > 0.45).astype(int)

# Gaussian HMM Implementation
class GaussianHMM:
    """A simplified Gaussian Hidden Markov Model for predicting flight delays."""

    def __init__(self, n_states=2, n_iters=50):
        self.n_states = n_states
        self.n_iters = n_iters
        self.means = None
        self.variances = None
        self.transition_probs = np.ones((n_states, n_states)) / n_states

    def initialize_parameters(self, X):
        """Initialize Gaussian parameters (mean & variance) for each state."""
        np.random.seed(42)
        idx = np.random.choice(len(X), self.n_states, replace=False)
        self.means = X[idx]  # Randomly initialize means
        self.variances = np.var(X, axis=0) + 1e-6  # Initialize variances

    def gaussian_pdf(self, x, mean, var):
        """Compute Gaussian probability density function."""
        return np.exp(-0.5 * ((x - mean) ** 2) / (var + 1e-6)) / np.sqrt(2 * np.pi * (var + 1e-6))

    def e_step(self, X):
        """Expectation step: Compute responsibilities (posterior probabilities)."""
        responsibilities = np.zeros((len(X), self.n_states))
        for i in range(self.n_states):
            diff = X - self.means[i]
            responsibilities[:, i] = np.exp(-0.5 * np.sum(diff**2 / (self.variances + 1e-6), axis=1))
        responsibilities /= responsibilities.sum(axis=1, keepdims=True) + 1e-6
        return responsibilities

    def m_step(self, X, responsibilities):
        """Maximization step: Update mean and variance estimates."""
        for i in range(self.n_states):
            resp = responsibilities[:, i]
            self.means[i] = np.sum(resp[:, np.newaxis] * X, axis=0) / (np.sum(resp) + 1e-6)
            self.variances = np.sum(resp[:, np.newaxis] * (X - self.means[i])**2, axis=0) / (np.sum(resp) + 1e-6)

    def train(self, X):
        """Train the Gaussian HMM using Expectation-Maximization."""
        self.initialize_parameters(X)
        for _ in range(self.n_iters):
            responsibilities = self.e_step(X)
            self.m_step(X, responsibilities)

    def predict(self, X):
        """Predict the most likely state sequence using Viterbi algorithm."""
        responsibilities = self.e_step(X)
        return np.argmax(responsibilities, axis=1)

# Train Hybrid Model (NB + Gaussian HMM) using NB probability and numerical features (Time, Length)
X = df[['Time', 'Length']].values

# Train Gaussian HMM
hmm = GaussianHMM(n_states=2, n_iters=50)
hmm.train(X)

# Predict using Gaussian HMM
df['HMM_PredictedState'] = hmm.predict(X)

# Ensure that NB_Probability exists before Hybrid Decision
if 'NB_Probability' not in df.columns:
    raise ValueError("Error: 'NB_Probability' was not created in dataframe. Check Naïve Bayes function.")

# Final Hybrid Decision: Combining Naïve Bayes and HMM
def hybrid_prediction(nb_prob, hmm_state):
    """Combine NB probability and HMM state for final prediction."""
    if hmm_state == 1 and nb_prob > 0.5:
        return 1  # Delayed
    return 0  # On-time

df['Hybrid_PredictedDelay'] = df.apply(lambda row: hybrid_prediction(row['NB_Probability'], row['HMM_PredictedState']), axis=1)

# Evaluate Performance
accuracy = np.mean(df['Hybrid_PredictedDelay'] == df['Delay'])
print(f"Hybrid NB + Gaussian HMM Accuracy: {accuracy:.4f}")

# Save predictions
df[['NB_PredictedDelay', 'HMM_PredictedState', 'Hybrid_PredictedDelay']].to_csv("predictions_NB_GaussianHMM.csv", index=False)