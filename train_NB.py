import pandas as pd
import numpy as np

# Loading the preprocessed dataset
df = pd.read_csv("preprocessed_airlines.csv")

# Computing P(Delay)
p_delay = df['Delay'].value_counts(normalize=True)

# Computing P(Feature | Delay) dynamically from raw data
def compute_CP(feature):
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
def predict_delay(row):
    log_p_delay_1 = np.log(p_delay[1])
    log_p_delay_0 = np.log(p_delay[0])

    for feature, cpt in zip(['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek', 'Time', 'Length'], 
                            [p_delay_airline, p_delay_airport_from, p_delay_airport_to, 
                             p_delay_day_of_week, p_delay_time, p_delay_length]):
        if row[feature] in cpt.columns:
            log_p_delay_1 += np.log(cpt.loc[1, row[feature]] + 1e-5)
            log_p_delay_0 += np.log(cpt.loc[0, row[feature]] + 1e-5)

    if log_p_delay_1 > log_p_delay_0:
        return 1
    else:
        return 0

# Applying the prediciton function to all the rows
df['PredictedDelay'] = df.apply(predict_delay, axis = 1)

# Evaluate the performance of this NB model
accuracy = np.mean(df['PredictedDelay'] == df['Delay'])
print(f"NB Model Accuracy: {accuracy:.4f}")

# Saving predictions
df.to_csv("predictions_NB.csv", index=False)