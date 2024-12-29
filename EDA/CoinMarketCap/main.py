"""
This dataset contains information about cryptocurrency prices, market capitalization, and other metrics. 
The data is collected from CoinMarketCap (https://coinmarketcap.com/), a popular website that tracks cryptocurrency prices.

This dataset can be used to:
- Analyze the price trends of different cryptocurrencies.
- Compare the market capitalization of different cryptocurrencies.
- Examine the circulating supply of different cryptocurrencies.
- Analyze the trading volume of different cryptocurrencies.
- Study the volatility of different cryptocurrencies.
- Compare the performance of different cryptocurrencies against each other or against a benchmark index.
- Identify correlations between different cryptocurrency prices.
- Use the data to build models to predict future prices or other trends.

+Info: https://www.kaggle.com/datasets/harshalhonde/coinmarketcap-cryptocurrency-dataset-2023
"""

import sys
import os
from utils.processing import DataLoader
from utils.analyzer import DataAnalyzer

# ---------------- LOAD DATASET -------------------#

# Dynamically detect the project's root directory
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
print("Dynamically detected root directory:", project_root)

# Simplified dataset path
df_path = os.path.join(project_root, "dataSet")
print(f"Dataset path: {df_path}")
df = "currencies_data_Kaggle_2023_unique.csv"

# ---------------- LOAD AND ANALYZE DATA -------------------#
try:
    loader = DataLoader(df_path=df_path, df=df)
    df = loader.load_data()
    print("\n--- Dataset successfully loaded ---")
except FileNotFoundError as e:
    print(f"Error loading dataset: {e}")
    df = None
except ValueError as e:
    print(f"Dataset value error: {e}")
    df = None

# Proceed with analysis if data is loaded
if df is not None:
    # Instantiate the analyzer
    analyzer = DataAnalyzer(df)

    # Call analyzer methods to verify functionality
    analyzer.overview()
    analyzer.duplicates_analysis()
    analyzer.missing_values_analysis()  # Takes 7-10 minutes; please be patient...
    analyzer.data_types_analysis()
else:
    print("\n--- Could not load the dataset. Analysis aborted ---")

# ---------------- PROCESS DATA -------------------#

'''
We will handle dates, NaN values, and categorical variables
'''

# For now, drop 'name.1', which is duplicated

if 'name.1' in df.columns:
    df.drop(columns=['name.1'], inplace=True)
    print("Column 'name.1' removed.")
    analyzer.data_types_analysis()

# Check columns with NaN values
nan_by_column = df.isnull().sum()
print(nan_by_column[nan_by_column > 0])
'''The column maxSupply contains all NaN values
and this is because the data is unavailable, so we will fill it with 0.'''

df.fillna(0, inplace=True)
print(f"Remaining NaN values: {df.isnull().sum().sum()}")  # Confirm no NaN values remain
analyzer.missing_values_analysis()

# Convert dates to datetime format and prepare for time series analysis

import pandas as pd

# Convert date columns to datetime format
df['lastUpdated'] = pd.to_datetime(df['lastUpdated'], errors='coerce')
df['dateAdded'] = pd.to_datetime(df['dateAdded'], errors='coerce')
analyzer.overview()

# Create a temporal index without dropping the column dateAdded (in case we want to work with time series later)
df.set_index('dateAdded', inplace=True, drop=False)
print(df.index)

# Sort the DataFrame by the index (dateAdded)
df.sort_index(inplace=True)
print(df.index.is_monotonic_increasing)  # Should return True if sorted

# Create derived columns from 'dateAdded' to study when cryptocurrencies were added
df['year_added'] = df.index.year
df['month_added'] = df.index.month
df['day_added'] = df.index.day
df['weekday_added'] = df.index.weekday  # 0 = Monday, 6 = Sunday
print(df[['year_added', 'month_added', 'day_added', 'weekday_added']].head())
analyzer.overview()
df.head()

# Normalize the data to check for more duplicates
df['name'] = df['name'].str.strip().str.title()  # Title case for names
df['symbol'] = df['symbol'].str.strip().str.upper()  # Uppercase for symbols
print(df[['name', 'symbol']].head())

# Check for duplicates between 'name' and 'symbol'
duplicates = df[df.duplicated(subset=['name', 'symbol'], keep=False)]
print(duplicates)
print(f"Found duplicates: {duplicates.shape[0]}")
'''After normalizing to title case for names and uppercase for symbols,
we found 62 duplicates for Symbol, which corresponds to USD, indicating the value in dollars
for these cryptocurrencies as a pair value. This is not relevant, so we remove them, focusing on their symbol value.'''

# Handle the two remaining categorical variables: name and symbol
'''The strategy is as follows:
Create a dictionary mapping names to their LabelEncoder values.
This allows us to reference this file for future visualizations or mappings.'''

from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder for 'name'
le = LabelEncoder()
df['name_encoded'] = le.fit_transform(df['name'])

# Create a dictionary mapping 'name' -> 'name_encoded'
name_to_encoded = dict(zip(df['name'], df['name_encoded']))

# Verify the result
print("First encoded values:")
print(df[['name', 'name_encoded']].head())

# Save the dictionary to a CSV file
mapping_df = pd.DataFrame(list(name_to_encoded.items()), columns=['name', 'name_encoded'])
mapping_df.to_csv('EDA/CoinMarketCap/dataSet/name_encoded_mapping.csv', index=False)
print("Mapping dictionary created and saved as 'name_encoded_mapping.csv'")

# Remove columns 'name' and 'symbol'
df = df.drop(columns=['name', 'symbol'])
print("Remaining columns after removing 'name' and 'symbol':")
print(df.columns.tolist())

# Save the cleaned dataset ready for further analysis and/or training
df.to_csv('EDA/CoinMarketCap/dataSet/currencies_data_ready.csv', index=False)
print("Dataset ready and saved as 'currencies_data_ready.csv'")
