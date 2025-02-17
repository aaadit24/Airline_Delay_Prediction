import pandas as pd

# Loading the airlines dataset
file_path = "Airlines.csv"
df = pd.read_csv(file_path)

# Checking for any missing values
print("Missing values:\n", df.isnull().sum())

# Renaming 'Flight' to 'FlightNumber'
df.rename(columns={'Flight': 'FlightNumber'}, inplace=True)

# Dropping the 'id' column as it is not useful for modellingtha
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

# Convert 'DayOfWeek' (1-7) into actual weekday names
day_mapping = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Satursday", 7: "Sunday"}
df['DayOfWeek'] = df['DayOfWeek'].map(day_mapping)

# Converting 'Time' (minutes from midnight) into HH:MM format
df['Time'] = df['Time'].apply(lambda x: f"{int(x // 60):02}:{int(x % 60):02}")

df.to_csv("preprocessed_airlines.csv", index=False)
print(df.head())