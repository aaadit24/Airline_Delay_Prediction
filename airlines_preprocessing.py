import pandas as pd

# Loading the airlines dataset
file_path = "Airlines.csv"
df = pd.read_csv(file_path)

# Checking for any missing values
print("Missing values:\n", df.isnull().sum())

# Dropping the 'id' column as it is not useful for modellingtha
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

airline_mapping = {category: idx for idx, category in enumerate(df['Airline'].unique())}
airport_from_mapping = {category: idx for idx, category in enumerate(df['AirportFrom'].unique())}
airport_to_mapping = {category: idx for idx, category in enumerate(df['AirportTo'].unique())}

# Applying mappings
df['Airline'] = df['Airline'].map(airline_mapping)
df['AirportFrom'] = df['AirportFrom'].map(airport_from_mapping)
df['AirportTo'] = df['AirportTo'].map(airport_to_mapping)

time_min, time_max = df['Time'].min(), df['Time'].max()
length_min, length_max = df['Length'].min(), df['Length'].max()

df['Time'] = (df['Time'] - time_min) / (time_max - time_min)
df['Length'] = (df['Length'] - length_min) / (length_max - length_min)

df.to_csv("preprocessed_airlines.csv", index=False)
print(df.head())