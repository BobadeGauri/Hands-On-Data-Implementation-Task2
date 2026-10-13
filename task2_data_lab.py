"""Task 2: Hands-On Data Lab Implementation
Run from this folder:
    pip install pandas numpy matplotlib
    python task2_data_lab.py
The practice dataset is synthetic and for educational purposes.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = Path("retail_sales_raw.csv")
OUT = Path("outputs")
PLOTS = Path("plots")
OUT.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

# 1. Explore the raw data
df = pd.read_csv(DATA_FILE)
print("First five rows:\\n", df.head())
print("\\nShape:", df.shape)
print("\\nData types:\\n", df.dtypes)
print("\\nMissing values by column:\\n", df.isna().sum())
print("\\nDuplicate rows:", df.duplicated().sum())
print("\\nDescriptive summary:\\n", df.describe(include="all"))

# 2. Clean types and inconsistent text
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
df["Product"] = df["Product"].astype("string").str.strip().str.title()
df["Region"] = df["Region"].astype("string").str.strip().str.title()
for col in ["Quantity", "Unit_Price", "Sales"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 3. Remove exact duplicates and impute missing values
df = df.drop_duplicates()
for col in ["Quantity", "Unit_Price"]:
    df[col] = df[col].fillna(df[col].median())
for col in ["Product", "Region"]:
    df[col] = df[col].fillna(df[col].mode().iloc[0])
df["Sales"] = (df["Quantity"] * df["Unit_Price"]).round(2)

# 4. Feature engineering
df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)
df["Revenue_Band"] = pd.cut(
    df["Sales"], bins=[-0.01, 100, 500, 1500, 100000],
    labels=["Low", "Medium", "High", "Very High"]
)
df.to_csv(OUT / "retail_sales_cleaned.csv", index=False)

# 5. Statistical summaries and category aggregations
stats = df[["Quantity", "Unit_Price", "Sales"]].agg(
    ["count", "mean", "median", "std", "var", "min", "max"]
).T.round(2)
stats.to_csv(OUT / "descriptive_statistics.csv")
product_summary = df.groupby("Product", observed=False).agg(
    Orders=("Order_ID", "count"),
    Units_Sold=("Quantity", "sum"),
    Total_Sales=("Sales", "sum"),
    Average_Order_Sales=("Sales", "mean")
).sort_values("Total_Sales", ascending=False).round(2)
product_summary.to_csv(OUT / "product_summary.csv")
region_summary = df.groupby("Region", observed=False).agg(
    Orders=("Order_ID", "count"),
    Units_Sold=("Quantity", "sum"),
    Total_Sales=("Sales", "sum"),
    Average_Order_Sales=("Sales", "mean")
).sort_values("Total_Sales", ascending=False).round(2)
region_summary.to_csv(OUT / "region_summary.csv")
corr = df[["Quantity", "Unit_Price", "Sales"]].corr().round(3)
corr.to_csv(OUT / "correlation_matrix.csv")

print("\\nDescriptive statistics:\\n", stats)
print("\\nProduct summary:\\n", product_summary)
print("\\nRegion summary:\\n", region_summary)
print("\\nCorrelation matrix:\\n", corr)

# 6. Visualizations
df.groupby("Product")["Sales"].sum().sort_values().plot(kind="barh", title="Total Sales by Product")
plt.tight_layout(); plt.savefig(PLOTS / "sales_by_product.png", dpi=160); plt.close()
df.groupby("Region")["Sales"].sum().sort_values(ascending=False).plot(kind="bar", title="Total Sales by Region")
plt.tight_layout(); plt.savefig(PLOTS / "sales_by_region.png", dpi=160); plt.close()
df.groupby("Order_Date")["Sales"].sum().sort_index().plot(kind="line", title="Daily Sales Trend")
plt.tight_layout(); plt.savefig(PLOTS / "daily_sales_trend.png", dpi=160); plt.close()
df["Sales"].plot(kind="hist", bins=22, title="Distribution of Order Sales")
plt.tight_layout(); plt.savefig(PLOTS / "sales_distribution.png", dpi=160); plt.close()
matrix = df[["Quantity", "Unit_Price", "Sales"]].corr()
plt.figure(figsize=(5, 4))
plt.imshow(matrix, aspect="auto")
plt.xticks(range(3), matrix.columns, rotation=25, ha="right")
plt.yticks(range(3), matrix.index)
for i in range(3):
    for j in range(3):
        plt.text(j, i, f"{matrix.iloc[i,j]:.2f}", ha="center", va="center")
plt.title("Correlation Matrix"); plt.colorbar(); plt.tight_layout()
plt.savefig(PLOTS / "correlation_matrix.png", dpi=160); plt.close()
df.groupby("Revenue_Band", observed=False)["Sales"].sum().plot(kind="bar", title="Sales by Revenue Band")
plt.tight_layout(); plt.savefig(PLOTS / "sales_by_revenue_band.png", dpi=160); plt.close()

print("\\nDone. Cleaned data, tables and charts are saved in outputs/ and plots/.")
