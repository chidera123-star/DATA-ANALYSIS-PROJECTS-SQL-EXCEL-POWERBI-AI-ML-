import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv("cleaned_titanic.csv")

print("=== BASIC INFO ===")
print(df.info())

print("\n=== STATISTICS ===")
print(df.describe())

# -----------------------------
# 1. Survival Count
# -----------------------------
sns.countplot(x='survived', data=df)
plt.title("Survival Count")
plt.savefig("survival_count.png")
plt.close()

# -----------------------------
# 2. Survival by Gender
# -----------------------------
sns.countplot(x='survived', hue='sex', data=df)
plt.title("Survival by Gender")
plt.savefig("survival_by_gender.png")
plt.close()

# -----------------------------
# 3. Age Distribution
# -----------------------------
plt.hist(df['age'], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("age_distribution.png")
plt.close()

# -----------------------------
# 4. Correlation Heatmap
# -----------------------------
numeric_df = df.select_dtypes(include=['number'])
plt.figure()
sns.heatmap(numeric_df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

# -----------------------------
# 5. Survival by Passenger Class
# -----------------------------
sns.countplot(x='pclass', hue='survived', data=df)
plt.title("Survival by Class")
plt.savefig("survival_by_class.png")
plt.close()

# -----------------------------
# 6. Fare Distribution
# -----------------------------
plt.hist(df['fare'], bins=20)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.savefig("fare_distribution.png")
plt.close()

# -----------------------------
# 7. Survival by Embarkation Port
# -----------------------------
sns.countplot(x='embarked', hue='survived', data=df)
plt.title("Survival by Embarkation Port")
plt.savefig("survival_by_embarked.png")
plt.close()