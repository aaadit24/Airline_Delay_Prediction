# Airline Delay Prediction using Hybrid Naïve Bayes + Gaussian HMM

## Overview
In Milestone 2, we used a **Naïve Bayes (NB) model** to predict flight delays based on historical data. It provided useful probability estimates and helped identify key features affecting delays. However, the model assumes **feature independence**, which may not always hold true in real-world flight delays.

For Milestone 3, we introduced a **Hybrid model that combines Naïve Bayes with a Gaussian Hidden Markov Model (NB + Gaussian HMM):**
- **Naïve Bayes** computes the probability of a flight delay based on categorical features (Airline, Airports, Day of the Week).
- **Gaussian HMM** models underlying latent states of flight delays based on numerical features (Flight Time and Length), accounting for hidden dependencies between flight conditions.

This approach allows us to analyze **both categorical and continuous variables** together, introducing a **temporal component through HMM** while still leveraging NB’s probability estimates.

---

## PEAS (Performance, Environment, Actuators, Sensors) for Our AI Agent

| Component    | Explanation |
|-------------|-------------|
| **Performance** | The model is evaluated based on accuracy, precision, recall, and F1-score. We also compare different threshold values. |
| **Environment** | The model operates in the domain of flight departure data, including airline names, departure & arrival airports, day of the week, flight length, and departure time. |
| **Actuators** | The model outputs `PredictedDelay = {0 (On time), 1 (Delayed)}` for each flight. |
| **Sensors** | The model reads input features from the dataset, including categorical (`Airline, AirportFrom, AirportTo, DayOfWeek`) and continuous (`Time, Length`). |

---

## Why Gaussian HMM? How Does It Differ from Its Discrete Analog?

A **Hidden Markov Model (HMM)** is a **probabilistic model that captures temporal dependencies by modeling an observed sequence with hidden states.** In this case, we assume that **flight delays have underlying latent states** influenced by observed numerical features (Time and Length).

### Why Not Use a Discrete HMM?
A **Discrete HMM** models observations as categorical/discrete variables. Since **Time and Length are continuous**, using a discrete HMM would require **binning continuous data into categories**, which may lead to **loss of information and granularity.**

### Why Gaussian HMM?
A **Gaussian HMM** models **continuous variables using Gaussian distributions for each hidden state.** Instead of assuming fixed categories, we allow **each hidden state to follow a Gaussian distribution over numerical features.** This is more suitable for our dataset because:
- **Flight times and durations vary continuously**, making Gaussian HMMs a better choice.
- **Delays are influenced by hidden factors (weather, airport congestion, airline scheduling)**, which are best modeled using continuous probability distributions.
- **Gaussian emissions allow for smoother transitions between delay states** rather than rigid discrete categories.


---
## **Probability Formulas for Naïve Bayes Model**

- **Bayes' Theorem for classification:**
  
  $$
  P(Delay | Features) = \frac{P(Features | Delay) P(Delay)}{P(Features)}
  $$

- Since we assume feature independence, the formula simplifies to:
  `P(Delay | Airline, AirportFrom, AirportTo, DayOfWeek, Time, Length) = P(Airline | Delay) P(AirportFrom | Delay) P(AirportTo | Delay) P(DayOfWeek | Delay) P(Time | Delay) P(Length | Delay) P(Delay) / P(Airline, AirportFrom, AirportTo, DayOfWeek, Time, Length)`

- The denominator `P(Airline, AirportFrom, AirportTo, DayOfWeek, Time, Length)` is constant across classifications, so we only compute the numerator.

### **Computing Conditional Probabilities**
- For each feature \(X_i\), we estimate its probability using:

  `P(X_i | Delay) = \frac{\text{Count}(X_i, Delay)}{\sum_{X_i} \text{Count}(X_i, Delay)}`

- To avoid zero probabilities, Laplace Smoothing is applied:

  `P(X_i | Delay) = \frac{\text{Count}(X_i, Delay) + 1}{\sum_{X_i} \text{Count}(X_i, Delay) + k}`

  - \(k\) is the number of unique values in feature \(X_i\).

### **Log Probability Computation**
- Multiplying many small probabilities can lead to numerical instability, so we use log probabilities:

  
  `log P(Delay | Features) = log P(Delay) + \sum_{i} log P(X_i | Delay)`

  `log P(NoDelay | Features) = log P(NoDelay) + \sum_{i} log P(X_i | NoDelay)`

### **Prediction Decision**
- The final classification is based on a threshold (default = 0.45):

  `e^{log P(Delay | Features)} / (e^{log P(Delay | Features)} + e^{log P(NoDelay | Features)})`

- If the computed probability is greater than 0.45, the flight is predicted as delayed. Otherwise, it is predicted as on time.

---

## Formulation of Gaussian HMM

An HMM consists of:
- **Hidden states:** Latent variables representing different delay patterns.
- **Observations:** Continuous flight features (`Time, Length`).
- **Transition probabilities:** Probability of transitioning from one hidden state to another.
- **Emission probabilities:** Likelihood of observing `Time, Length` given a hidden state.

Mathematically, the **Gaussian HMM** models the probability of an observed sequence \(X = \{X_1, X_2, ..., X_n\}\) as:

\[
P(X | \theta) = \sum_{S} P(S_1) P(X_1 | S_1) \prod_{t=2}^{n} P(S_t | S_{t-1}) P(X_t | S_t)
\]

where:
- \(S_t\) is the hidden state at time \(t\).
- \(P(S_t | S_{t-1})\) is the **transition probability**.
- \(P(X_t | S_t)\) is the **emission probability**, modeled as a Gaussian:

\[
P(X_t | S_t = i) = \frac{1}{\sqrt{2\pi\sigma_i^2}} e^{-\frac{(X_t - \mu_i)^2}{2\sigma_i^2}}
\]

where:
- \(\mu_i\) and \(\sigma_i^2\) are the mean and variance of state \(i\).

---

## How Do We Perform Inference?

### **1. Training (Expectation-Maximization - EM Algorithm)**
The model is trained using **Expectation-Maximization (EM):**
- **E-step:** Compute responsibilities (probability of each state given the data).
- **M-step:** Update state parameters (\(\mu_i, \sigma_i^2\)) based on weighted data points.

### **2. Prediction (Viterbi Algorithm)**
For a given flight, we compute:
- **Naïve Bayes probability of delay** (based on categorical features).
- **Most probable hidden state** using the Gaussian HMM (based on `Time, Length`).
- **Final Hybrid Decision:** Combining NB probability and HMM state to make the final prediction.

---

## Code Files and Project Structure

| File | Description | Link |
|------|------------|------|
| `Airlines.csv` | Original Dataset from Kaggle | [Airlines.csv](./Airlines.csv) |
| `airlines_preprocessing.py` | Cleans and preprocesses dataset | [airlines_preprocessing.py](./airlines_preprocessing.py) |
| `preprocessed_airlines.csv` | Processed dataset used for modeling | [preprocessed_airlines.csv](./preprocessed_airlines.csv) |
| `train_NB.py` | Naïve Bayes training and prediction | [train_NB.py](./train_NB.py) |
| `train_NB_GaussianHMM.py` | Hybrid NB + Gaussian HMM model training | [train_NB_GaussianHMM.py](./train_NB_GaussianHMM.py) |
| `predictions_NB_GaussianHMM.csv` | Model predictions | [predictions_NB_GaussianHMM.csv](./predictions_NB_GaussianHMM.csv) |
| `evaluate_NB_GaussianHMM.py` | Code to evaluate the hybrid model | [evaluate_NB_GaussianHMM.py](./evaluate_NB_GaussianHMM.py) |

---

## Results & Model Performance

| Metric | Naïve Bayes (Threshold = 0.45) | Hybrid NB + Gaussian HMM |
|--------|-----------------|------------------|
| Accuracy | 61.89% | 56.67% |
| Precision | 58.25% | 53.86% |
| Recall | 50.99% | 19.02% |
| F1 Score | 54.38% | 28.11% |
| Specificity | 70.65% | 86.91% |
| False Positive Rate (FPR) | 29.35% | 13.09% |
| False Negative Rate (FNR) | 49.01% | 80.98% |

---

## Key Findings

- **Lower FPR:** The hybrid model is more conservative, making fewer incorrect delay predictions.
- **Lower Recall:** It struggles to catch actual delays, indicating a need for better feature engineering.
- **Higher Specificity:** More accurate at identifying on-time flights.
- **Overall Accuracy Drop:** This suggests that adding HMM did not generalize well for delay prediction.

---

## Confusion Matrix Analysis

| Actual \ Predicted | Delayed | On Time |
|--------------------|---------|---------|
| **Naïve Bayes** | 122,504 | 117,760 |
| **Hybrid NB + Gaussian HMM** | 45,701 | 194,563 |

**Key Observations**
- Hybrid model **reduced false positives** (better FPR).
- However, **missed many actual delays** (higher FNR).
- Suggests that **HMM’s latent states didn’t generalize well for delay prediction.**

---

## Conclusion
- **Gaussian HMM introduced a hidden state model but struggled with recall.**
- **FPR dropped significantly, but the model missed too many actual delays.**
- **Future improvements** could involve:
  - **Feature selection:** Adding airline-specific delay patterns.
  - **Parameter tuning:** Optimizing HMM transition probabilities.
  - **Exploring alternative models:** LSTMs for sequential learning.

---

## Citations
- **pandas**: [https://pandas.pydata.org/](https://pandas.pydata.org/)
- **NumPy**: [https://numpy.org/](https://numpy.org/)
- **matplotlib & seaborn**: [https://matplotlib.org/](https://matplotlib.org/), [https://seaborn.pydata.org/](https://seaborn.pydata.org/)
- **scikit-learn**: [https://scikit-learn.org/](https://scikit-learn.org/)