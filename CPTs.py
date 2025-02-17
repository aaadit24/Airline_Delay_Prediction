import pandas as pd

# Loading the preprocessed dataset file 
file_path = "preprocessed_airlines.csv"
df = pd.read_csv(file_path)

# Checking for missing values
if df.isnull().values.any():
    print("\n Warning: Dataset contains missing values")

# Calculating P(Delay)
p_delay = df['Delay'].value_counts(normalize=True)
print("\n Probability of Delay:\n", p_delay)

# Saving P(Delay)
p_delay.to_csv("p_delay.csv")

# Calculating P(Delay | Airline)
p_delay_airline = df.groupby('Airline')['Delay'].value_counts(normalize=True).unstack()

# Calculating P(Delay | AirportFrom)
p_delay_airport_from = df.groupby('AirportFrom')['Delay'].value_counts(normalize=True).unstack()

# Calculating P(Delay | AirportTo)
p_delay_airport_to = df.groupby('AirportTo')['Delay'].value_counts(normalize=True).unstack()

# Calculating P(Delay | DayOfWeek)
p_delay_day_of_week = df.groupby('DayOfWeek')['Delay'].value_counts(normalize=True).unstack()

# Classifying time bins (Morning, Afternoon, Evening, Night)
df['Time_Bin'] = pd.cut(pd.to_datetime(df['Time'], format='%H:%M').dt.hour, 
                        bins=[0, 6, 12, 18, 24], 
                        labels=['Night', 'Morning', 'Afternoon', 'Evening'], 
                        right=False)
p_delay_time = df.groupby('Time_Bin', observed=False)['Delay'].value_counts(normalize=True).unstack()

# Classifying flight length bins (Short, Medium, Long flights)
df['Length_Bin'] = pd.cut(df['Length'], bins=[0, 120, 240, df['Length'].max()], labels=['Short', 'Medium', 'Long'])
p_delay_length = df.groupby('Length_Bin', observed=False)['Delay'].value_counts(normalize=True).unstack()

# Saving probability files as CSV files
p_delay_airline.to_csv("p_delay_airline.csv")
p_delay_airport_from.to_csv("p_delay_airport_from.csv")
p_delay_airport_to.to_csv("p_delay_airport_to.csv")
p_delay_day_of_week.to_csv("p_delay_day_of_week.csv")
p_delay_time.to_csv("p_delay_time.csv")
p_delay_length.to_csv("p_delay_length.csv")

# Combining the results from all probability tles into one file for quick and reasy reference
combined_cpts = {
    "P(Delay)": p_delay,
    "P(Delay | Airline)": p_delay_airline,
    "P(Delay | AirportFrom)": p_delay_airport_from,
    "P(Delay | AirportTo)": p_delay_airport_to,
    "P(Delay | DayOfWeek)": p_delay_day_of_week,
    "P(Delay | Time)": p_delay_time,
    "P(Delay | Length)": p_delay_length,
}

# Converting into a dataframe and saving as a summary file
with open("p_delay_summary.csv", "w") as f:
    for key, df in combined_cpts.items():
        f.write(f"=== {key} ===\n")
        df.to_csv(f)
        f.write("\n\n")

print("CPTs have been successfully generated and saved as CSV files")