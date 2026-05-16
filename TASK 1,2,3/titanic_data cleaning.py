import pandas as pd

# Step 1: Load dataset
df = pd.read_csv("titanic.csv")

# Step 2: Preview data
print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Step 3: Handle missing values

# Fill Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked with most frequent value
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column (too many missing values)
if 'Cabin' in df.columns:
    df.drop('Cabin', axis=1, inplace=True)

# Step 4: Remove duplicates
df.drop_duplicates(inplace=True)

# Step 5: Convert categorical data

# Convert Sex to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert Embarked to numeric
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Step 6: Clean column names
df.columns = df.columns.str.lower()

# Step 7: Final check
print("\nAfter Cleaning:")
print(df.isnull().sum())

print("\nCleaned Data Preview:")
print(df.head())

# Step 8: Save cleaned dataset
df.to_csv("cleaned_titanic.csv", index=False)

print("\n✅ Data cleaning completed and saved as 'cleaned_titanic.csv'")