import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Cleaned_Sales_Dataset.csv")

# Dataset Info
print("Dataset Shape:")
print(df.shape)

print("\nSummary Statistics:")
print(df.describe())

# Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=10)
plt.title("Age Distribution")
plt.savefig("age_distribution.png")
plt.show()

# Category Distribution
plt.figure(figsize=(8,5))
sns.countplot(x='Category', data=df)
plt.title("Category Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("category_distribution.png")
plt.show()

# Correlation Heatmap
numeric_df = df.select_dtypes(include=['int64','float64'])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# Quantity vs Sales
plt.figure(figsize=(8,5))
sns.scatterplot(x='Quantity', y='Total_Sales', data=df)
plt.title("Quantity vs Total Sales")
plt.savefig("quantity_vs_sales.png")
plt.show()

print("\nEDA Completed Successfully!")

# Age vs Total Sales
plt.figure(figsize=(8,5))
sns.scatterplot(x='Age', y='Total_Sales', data=df)
plt.title("Age vs Total Sales")
plt.savefig("age_vs_sales.png")
plt.show()

# Revenue by Category
plt.figure(figsize=(8,5))
df.groupby('Category')['Total_Sales'].sum().sort_values().plot(kind='bar')
plt.title("Revenue by Category")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("revenue_by_category.png")
plt.show()

# Monthly Revenue Trend
monthly_sales = df.groupby('Month')['Total_Sales'].sum()

plt.figure(figsize=(10,5))
monthly_sales.plot(marker='o')
plt.title("Monthly Revenue Trend")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("monthly_revenue_trend.png")
plt.show()