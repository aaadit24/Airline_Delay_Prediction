# **Airline Delay Prediction using Naïve Bayes**

## **Overview**
In this model, we applied the **Naïve Bayes approach** to predict flight delays based on historical flight data. Using this model, we aim to learn conditional probabilities from features such as airline, airport, day of the week, flight time, and flight length.

We further analyzed the dataset structure, feature interactions, and the assumptions made by Naïve Bayes to improve our model’s performance. We also evaluated alternative decision threshold values to see if lowering it to **0.45** would increase recall.

---

## **PEAS (Performance, Environment, Actuators, Sensors) for Our AI Agent**

| **Component**   | **Explanation** |
|----------------|-----------------------------------------------|
| **Performance** | The model is evaluated based on accuracy, precision, recall, and F1-score. We also compared performance at different threshold values. |
| **Environment** | The world consists of flight departure data, including airline names, departure & arrival airports, day of the week, and flight length. |
| **Actuators** | The model outputs PredictedDelay = {0 (On time), 1 (Delayed)} for each flight. |
| **Sensors** | The model senses input features from the dataset, including Airline, AirportFrom, AirportTo, DayOfWeek, Time, and Length. |

---

## **Type of Agent**
This is a **Goal-Based Agent** as the model has a clear goal of predicting flight delays accurately and chooses actions (**predict delay or no delay**) based on learned probabilities. It also uses **historical data** to make probabilistic decisions.

Alternatively, it can also be classified as a **Utility-Based Agent** since it **maximizes the probability of correct predictions** by optimizing the **decision threshold** and **feature selection** to enhance recall and precision.

---

## **Probabilistic Modeling: Where Does This Fit?**
Our project is a **Supervised Probabilistic Classification model** as it learns from past flight data to predict delays.

### **How It Is Set up:**

### **1. Naïve Bayes Theorem**
 - Through this model, we have estimated the probability of a flight delay based on different features (Airline, Airport, Time, etc.) using this formula:  
     \[
     P(Delay | Features) = \frac{P(Features | Delay) P(Delay)}{P(Features)}
     \]  
   - This helped the model decide whether a flight is likely to be delayed or on time.

### **2. Using Log Probabilities**
- Instead of multiplying probabilities (which can become extremely small), we convert them into **log values**. This **prevents numerical errors** and makes calculations **more stable**.

### **3. Computing Probabilities Dynamically**
- Another thing we did in training our first model was calculate these probabilities on the fly in `train_NB.py` instead of relying on precomputed probability tables (CPTs), which has made our model more flexible and adaptable to different dataset.

---

## **Exploratory Data Analysis & Feature Selection**
To better understand the dataset and feature relationships, we performed an **exploratory data analysis (EDA)**. Key insights include:

### **Feature Importance Analysis**
- **Time and Flight Length** were **critical factors** in predicting delays.
- **Certain airlines** had higher delay rates than others.
- **Some airports were more prone to delays**, particularly in congested hubs.

### **Correlation Heatmap (Feature Interactions)**
To validate **Naïve Bayes’ independence assumption**, we generated a **correlation heatmap** of numerical features.

**Findings:**
- **Flight Length and Time** had a moderate correlation.
- **Day of the Week** showed minor correlations with delays.

**Action Taken:** Since features are mostly independent, **Naïve Bayes remains a reasonable choice** for our model.

---

## **Code files and Project Structure**
| **File** | **Description** | **Link** |
|----------|----------------|----------|
| `Airlines.csv` | Original Dataset taken from kaggle | [Airlines.csv](./Airlines.csv) |
| `airlines_preprocessing.py` | Cleaned and preprocessed the raw dataset | [airlines_preprocessing.py](./airlines_preprocessing.py) |
| `preprocessed_airlines.csv` | Processed dataset which was used for modeling | [preprocessed_airlines.csv](./preprocessed_airlines.csv) |
| `train_NB.py` | Naïve Bayes training and prediction | [train_NB.py](./train_NB.py) |
| `predictions_NB.csv` | Dataset with model predictions (Delayed/Not Delayed) | [predictions_NB.csv](./predictions_NB.csv) |
| `evaluate_NB.py` | Code to evaluate the model and find accuracy, precision, recall, and F1-score | [evaluate_NB.py](./evaluate_NB.py) |

---

## **Results & Model Performance**

| **Metric**        | **Baseline (Threshold = 0.5)** | **Updated (Threshold = 0.45)** |
|-------------------|-------------------------------|-------------------------------|
| **Accuracy**      | 62.73%                         | **63.25%** |
| **Precision**     | 61.63%                         | **59.45%** |
| **Recall**        | 43.24%                         | **55.01%** |
| **F1 Score**      | 50.82%                         | **57.15%** |
| **Specificity**   | —                              | **69.86%** |
| **False Positive Rate (FPR)** | — | **30.14%** |
| **False Negative Rate (FNR)** | — | **44.99%** |

---

## **Confusion Matrix Analysis**
### **Confusion Matrix (Threshold = 0.45)**

| **Actual \ Predicted** | **Delayed** | **On Time** |
|------------------------|------------|------------|
| **Actually Delayed**   | 132,181    | 108,083 (False Negatives) |
| **Actually On Time**   | 90,149 (False Positives) | 208,970 |

### **Interpretation & Insights**
- This udated model successfully classifies 63.25% of all flights correctly.
- Precision is now 59.45%, meaning when the model predicts a flight will be delayed, it is correct 59.45% of the time.
- Recall has improved to 55.01%, meaning the model correctly detects **more than half of actual delays.
- False Negative Rate (FNR) is 44.99%, meaning there are **108,083 delayed flights that the model failed to catch.
- False Positive Rate (FPR) is 30.14%, meaning **90,149 flights were predicted as delayed but were actually on time.
- Specificity (Correct On-Time Predictions) is 69.86% showing that it does well at identifying on-time flights.

A confusion matrix visualization has been saved as [`confusion_matrix_plot.png`](./confusion_matrix_plot.png).

---

## **Threshold Adjustment Analysis**
- The model previously used a threshold of 0.5, which resulted in **lower recall**.
- **With a threshold of 0.45:**
  - More actual delays are caught (higher recall at 55.01%).
  - More false positives occur (lower precision at 59.45%).
  - If the cost of missing a delay is higher than predicting a false delay, this is an acceptable tradeoff.
- We also identify that further tuning could explore thresholds between 0.40 and 0.50 to optimize recall-precision tradeoff.

---

## **Citations**
- **Seaborn Documentation (For Heatmaps):** [https://seaborn.pydata.org/](https://seaborn.pydata.org/)