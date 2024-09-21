import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Task 1: Read local data (diabetes.csv) to data frame
df = pd.read_csv('diabetes.csv')

# Task 2: Print the first 10 rows of the data frame
print("First 10 rows of the data frame:")
print(df.head(10))

# Task 3: Print information about the data types, columns, null value counts, memory consumption
print("\nInformation about the data frame:")
print(df.info())

# Task 4: Print basic statistical details about the data
print("\nBasic statistical details about the data:")
print(df.describe())

# Task 5: Print basic statistical details about the data by reversing the axes
print("\nBasic statistical details about the data with reversed axes:")
print(df.describe().transpose())

# Task 6: Replace zero values with NaN for specific columns
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)

# Task 7: Plot the data distribution for each numeric column
print("\nPlotting data distribution for each numeric column...")
try:
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols].hist(figsize=(10,10))
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error plotting histograms: {e}")

# Task 8: Fill in NaN values for the columns using appropriate strategy
# Here, we can use median for numerical columns and mode for categorical columns
median_fill_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[median_fill_cols] = df[median_fill_cols].fillna(df.median())

# Task 9: Plot the data distribution after filling in the missing data
print("\nPlotting data distribution after filling in missing data...")
try:
    df[numeric_cols].hist(figsize=(10,10))
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error plotting histograms: {e}")



