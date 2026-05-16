import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("cleaned_titanic.csv")

# -----------------------------
# 1. Bar Chart: Survival Count
# -----------------------------
plt.figure()
sns.countplot(x='survived', data=df)
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# -----------------------------
# 2. Bar Chart: Survival by Gender
# -----------------------------
plt.figure()
sns.countplot(x='survived', hue='sex', data=df)
plt.title("Survival by Gender")
plt.show()

# -----------------------------
# 3. Bar Chart: Survival by Class
# -----------------------------
plt.figure()
sns.countplot(x='pclass', hue='survived', data=df)
plt.title("Survival by Passenger Class")
plt.show()

# -----------------------------
# 4. Scatter Plot: Age vs Fare
# -----------------------------
plt.figure()
plt.scatter(df['age'], df['fare'])
plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

# -----------------------------
# 5. Line Chart: Age Distribution Trend
# -----------------------------
plt.figure()
df_sorted = df.sort_values(by='age')
plt.plot(df_sorted['age'].values)
plt.title("Age Trend")
plt.xlabel("Index")
plt.ylabel("Age")
plt.show()