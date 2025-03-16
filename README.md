# **Airline Delay Prediction using Hybrid Naïve Bayes + Logistic Regression**

## **Overview**
In our Milestone 2, we used a Naïve Bayes (NB) model to predict flight delays. It performed well in estimating probabilities and gave valuable insights into how different features (airline, airport, day, time, flight length) impact delays. However, it assumes feature independence, which may not always be realistic.

For Milestone 3, we improved our model by implementing a Hybrid approach that combines Naïve Bayes with Logistic Regression (NB + LR):
- **Naïve Bayes** calculates initial delay probabilities.
- **Logistic Regression** takes these probabilities as input and adjusts the decision boundary to reduce false positives and increase specificity.

By combining these models, we refine predictions, reduce false alarms, and improve decision-making.

---

## **PEAS (Performance, Environment, Actuators, Sensors) for Our AI Agent**

| **Component**   | **Explanation** |
|----------------|-----------------------------------------------|
| **Performance** | The model is evaluated based on accuracy, precision, recall, and F1-score. We also compared performance at different threshold values. |
| **Environment** | The world consists of flight departure data, including airline names, departure & arrival airports, day of the week, and flight length. |
| **Actuators** | The model outputs PredictedDelay = {0 (On time), 1 (Delayed)} for each flight. |
| **Sensors** | The model senses input features from the dataset, including Airline, AirportFrom, AirportTo, DayOfWeek, Time, and Length. |

---

## **Why a Hybrid Approach?**
### **Milestone 2 (Naïve Bayes Model)**
- **Strengths:** Provided useful probability estimates of delays.
- **Limitations:** Assumes feature independence, leading to potential false positives.

### **Milestone 3 (NB + Logistic Regression)**
- **Why Logistic Regression?** It adjusts probability-based classification by optimizing weights for better decision-making.
- **Key Improvement:** Reduces false positives (incorrectly predicting delays).
- **Trade-off:** Recall decreased slightly, but precision and specificity increased, making delay predictions more reliable.

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

### **Logisitc Regression Model**
- After computing Naïve Bayes probabilities, we feed them into Logistic Regression, which applies the sigmoid function:

  `P(Y = 1 | X) = \frac{1}{1 + e^{-(W X + b)}}`

- \(W\) = Weights learned by the model
- \(b\) = Bias term
- \(X\) = Input feature (Naïve Bayes probability of delay)

- Logistic Regression is helping us adjust the decision boundary and making final predictions based on learned weights.

  $$
  P(Y = 1 | X) = \frac{1}{1 + e^{-(W X + b)}}
  $$
  - \( W \) = Weights learned by the model
  - \( b \) = Bias term
  - \( X \) = Input features (Naïve Bayes probabilities)

---

## **Code files and Project Structure**
| **File** | **Description** | **Link** |
|----------|----------------|----------|
| `Airlines.csv` | Original Dataset taken from Kaggle | [Airlines.csv](./Airlines.csv) |
| `airlines_preprocessing.py` | Cleaned and preprocessed the raw dataset | [airlines_preprocessing.py](./airlines_preprocessing.py) |
| `preprocessed_airlines.csv` | Processed dataset used for modeling | [preprocessed_airlines.csv](./preprocessed_airlines.csv) |
| `train_NB.py` | Naïve Bayes training and prediction | [train_NB.py](./train_NB.py) |
| `train_NB_LR.py` | Hybrid Naïve Bayes + Logistic Regression model training | [train_NB_LR.py](./train_NB_LR.py) |
| `predictions_NB.csv` | Dataset with model predictions (Delayed/Not Delayed) | [predictions_NB.csv](./predictions_NB.csv) |
| `predictions_NB_LR.csv` | Dataset with model predictions for NEW MODEL (Delayed/Not Delayed) | [predictions_NB_LR.csv](./predictions_NB_LR.csv) |
| `evaluate_NB_LR.py` | Code to evaluate the hybrid model | [evaluate_NB_LR.py](./evaluate_NB_LR.py) |

---

## **Results & Model Performance**

| **Metric**        | **Naïve Bayes (Threshold = 0.45)** | **Hybrid NB + Logistic Regression** |
|-------------------|-----------------------------------|------------------------------------|
| **Accuracy**      | 63.26%                            | **63.89%** |
| **Precision**     | 59.47%                            | **66.45%** |
| **Recall**        | 55.05%                            | **38.24%** |
| **F1 Score**      | 57.17%                            | **48.55%** |
| **Specificity**   | 69.86%                            | **84.49%** |
| **False Positive Rate (FPR)** | 30.14% | **15.51%** |
| **False Negative Rate (FNR)** | 44.95% | **61.76%** |

### **Key Findings**
- **Higher precision (66.45%)** → The hybrid model makes fewer incorrect delay predictions.
- **Significantly lower False Positive Rate (FPR: 15.51%)** → Reduces unnecessary delay alerts.
- **Higher specificity (84.49%)** → The model is better at identifying on-time flights.

---

## **Confusion Matrix Analysis**
### **Naïve Bayes Model (Threshold = 0.45)**

| **Actual \ Predicted** | **Delayed** | **On Time** |
|------------------------|------------|------------|
| **Actually Delayed**   | 132,260    | 108,004 (False Negatives) |
| **Actually On Time**   | 90,147 (False Positives) | 208,972 |

### **Hybrid Model (NB + Logistic Regression, Threshold = 0.45)**

| **Actual \ Predicted** | **Delayed** | **On Time** |
|------------------------|------------|------------|
| **Actually Delayed**   | **91,886**  | **148,378** (False Negatives) |
| **Actually On Time**   | **46,402** (False Positives) | **252,717** |

### **Interpretation & Insights**
- The Hybrid model improves accuracy from 63.26% → 63.89%.
- False Positive Rate (FPR) dropped from 30.14% → 15.51%, meaning fewer flights were wrongly predicted as delayed.
- Higher specificity (84.49%) shows that on-time flights are predicted more accurately.
- Recall is lower (38.24%), meaning fewer actual delays were caught.

A confusion matrix visualization has been saved as:
- [`confusion_matrix_nb.png`](./confusion_matrix_nb.png) (Naïve Bayes)
- [`confusion_matrix_hybrid.png`](./confusion_matrix_hybrid.png) (Hybrid NB + LR)

---

## **Threshold Adjustment Analysis**
- The model used a threshold of 0.45, balancing recall and precision.
- Lowering the threshold may catch more delays but increase false positives.
- Further tuning could explore thresholds between 0.40 and 0.50 for optimization.

---

## **Citations**
- **pandas**: Data handling ([https://pandas.pydata.org/](https://pandas.pydata.org/))
- **NumPy**: Mathematical operations ([https://numpy.org/](https://numpy.org/))
- **matplotlib & seaborn**: Data visualization ([https://matplotlib.org/](https://matplotlib.org/), [https://seaborn.pydata.org/](https://seaborn.pydata.org/))
- **scikit-learn**: Confusion matrix ([https://scikit-learn.org/](https://scikit-learn.org/))
---