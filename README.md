# Airline Delay Prediction using Naïve Bayes

## **Overview**
In thhis model, we applied **Naïve Bayes** approach to predict flight delays based on historical flight data.  
The model is designed to learn conditional probabilities from features like airline, airport, day of the week, flight time, and flight length.

---

## **PEAS (Performance, Environment, Actuators, Sensors) for Our AI Agent**
| Component  | Explanation |
|------------|------------|
| **Performance** | The model is evaluated based on accuracy, precision, recall, and F1-score |
| **Environment** | The world consists of flight departure data, including airline names, departure & arrival airports, day of the week, and flight length |
| **Actuators** | The model outputs PredictedDelay = {0 (On time), 1 (Delayed)} for each flight |
| **Sensors** | The model senses input features from the dataset including Airline, AirportFrom, AirportTo, DayOfWeek, Time, and Length |

---

## **Type of Agent**
This is a **Goal-Based Agent** as the model has a clear goal of predicting flight delays accurately and chooses actions (predict delay or no delay) based on learned probabilities.It also uses historical data to make these probabilistic decisions. Alternatively, it can also be seen as a **Utility-Based Agent** as the model maximizes probability of correct predictions and learns from conditional probability distributions over flight delays.

---

## **Probabilistic Modeling: Where Does This Fit?**
Our project is a **Supervised Probabilistic Classification model** as it learns from past flight data to predict delays

### **How It Is Set up:**  
1. **Naïve Bayes Theorem:**  
   - Through this model, we have estimated the probability of a flight delay based on different features (Airline, Airport, Time, etc.) using this formula:  
     \[
     P(Delay | Features) = \frac{P(Features | Delay) P(Delay)}{P(Features)}
     \]  
   - This helped the model decide whether a flight is likely to be delayed or not.

2. **Using Log Probabilities:**  
   - Instead of multiplying probabilities (which can become extremely small), we convert them into log values. We did this to prevent numerical errors and makes calculations more stable.

3. **Computing Probabilities Dynamically:**  
   - Another thing we did in training our first model was calculate these probabilities on the fly in `train_NB.py` instead of relying on precomputed probability tables (CPTs), which has made our model more flexible and adaptable to different dataset.
   
---

## **Code files and Project Structure**
| **File** | **Description** | **Link** |
|----------|----------------|----------|
| `Airlines.csv` | Original Dataset taken from kaggle | [Airlines.csv](./Airlines.csv) |
| `airlines_preprocessing.py` | Cleans and preprocesses the raw dataset | [airlines_preprocessing.py](./airlines_preprocessing.py) |
| `preprocessed_airlines.csv` | Processed dataset ready for modeling | [preprocessed_airlines.csv](./preprocessed_airlines.csv) |
| `train_NB.py` | Naïve Bayes training and prediction | [train_NB.py](./train_NB.py) |
| `predictions_NB.csv` | Model predictions (Delayed/Not Delayed) | [predictions_NB.csv](./predictions_NB.csv) |
| `evaluate_NB.py` | Evaluates accuracy, precision, recall, and F1-score | [evaluate_NB.py](./evaluate_NB.py) |

---

## **Results & Model Performance**
| **Metric** | **Score** |
|------------|-----------------|
| **Accuracy** | 62.73% |
| **Precision** | 61.63% |
| **Recall** | 43.24% |
| **F1 Score** | 50.82% |

**Conclusion:**
- One of the major finding for our group was that adding **Time** and **Flight Length** as features improved recall significantly. The model now identifies more actual delays which reduces False Negatives. Accuracy, Precision, and F1-score all improved, though the changes were moderate.
- We are actively discussing strategies to improve the model further, and one of the strategies we want to try before the next milestone is to lower the decision threshold to 0.45 to catch more delayed flights.
