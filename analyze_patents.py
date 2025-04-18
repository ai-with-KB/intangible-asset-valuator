import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("ai_finance_patents_cleaned.csv")

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Show basic statistics
print("📊 Descriptive Statistics:\n", df.describe())

# --- Plot 1: Citation Count Distribution ---
plt.figure(figsize=(8, 5))
sns.histplot(df['citation_count'], bins=10, kde=True, color='skyblue')
plt.title('Citation Count Distribution')
plt.xlabel('Citation Count')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Number of Patents by Year ---
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='publication_year', palette='viridis')
plt.title('Number of Patents per Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 3: Citation Categories ---
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='citation_category', palette='Set2')
plt.title('Citation Category Distribution')
plt.xlabel('Category')
plt.ylabel('Number of Patents')
plt.grid(True)
plt.tight_layout()
plt.show()
