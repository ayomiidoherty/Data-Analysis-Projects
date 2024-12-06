import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\ayomi\OneDrive\Documents\data analysix\statsfinal.csv")

print(df.head())

print(df.info())

print(df.describe())


#EDA 

#Check for null values
print(df.isnull().sum())  

#Change the data type of the months column
df['Date'] = pd.to_datetime(df['Date'], format='%d%m%Y') 

#Confirm the DATE data type changed

print(df.dtypes)

# Change the column names 

new_columns = ["No","Months","P1_SALES","P2_SALES","P3_SALES","P4_SALES","P1_REVENUE","P2_REVENUE","P3_REVENUE","P4_REVENUE"]
df.columns = new_columns
df.columns
print(df.columns)
print(df.head())

# CHECK FOR NULL VALUES 

missing_values = df.isnull().sum()
print(missing_values)


df['Monthly_Sales']= df['Months'].dt.month
df['Years'] = df['Months'].dt.year

# 1. Analyze Trend Sales 

# Calculate the total amount of sales per month

Monthly_Totals = df.groupby('Monthly_Sales')[["P1_SALES","P2_SALES","P3_SALES","P4_SALES", "P1_REVENUE","P2_REVENUE","P3_REVENUE","P4_REVENUE"]].sum()
print(Monthly_Totals)
Monthly_Totals.to_csv("monthly_totals.csv", index=True)


# 2. Product with the highest sales in the given years 

# Find the month with the highest amount of sales 

Highest_Sales = Monthly_Totals[["P1_SALES","P2_SALES","P3_SALES","P4_SALES"]].idxmax()
print(Highest_Sales)

Lowest_Sales = Monthly_Totals[["P1_SALES","P2_SALES","P3_SALES","P4_SALES"]].idxmin()
print(Lowest_Sales)


df['Total_Sales'] = df[["P1_SALES","P2_SALES","P3_SALES","P4_SALES"]].sum(axis=1)
df['Total_Revenue'] = df[["P1_REVENUE","P2_REVENUE","P3_REVENUE","P4_REVENUE"]].sum(axis=1)
print(df.columns)

revenue_sales = df[["Months_Sales","Total_Sales", "Total_Revenue"]]
revenue_sales.to_csv('revenue_sales.csv', index=False)




#CHECK THE PRICE PER UNIT 

df['P1_Price_Per_Unit'] = df["P1_REVENUE"] / df["P1_SALES"]
df['P2_Price_Per_Unit'] = df["P2_REVENUE"] / df["P2_SALES"]
df['P4_Price_Per_Unit'] = df["P4_REVENUE"] / df["P4_SALES"]
df['P3_Price_Per_Unit'] = df["P3_REVENUE"] / df["P3_SALES"]

df["P4_Price_Per_Unit"]

# ESTIMATE SALES FOR DECEMBER 31ST
# EXPLAIN THIS FORMULA 


df['Day'] = df['Months'].dt.day


avg_sales_30th = df[df['Day'] == 30][["P1_SALES", "P2_SALES", "P3_SALES", "P4_SALES"]].mean()
avg_sales_31st = df[df['Day']== 31][["P1_SALES", "P2_SALES", "P3_SALES", "P4_SALES"]].mean()

scaled_31st = avg_sales_30th * (31/30)
print(scaled_31st)

adjusted_sales_31st = avg_sales_31st.fillna(scaled_31st)
adjusted_sales_31st.to_csv('adjusted_sales.csv', index=True)




# 4. PRODUCT TO DROP 
product_contribution = df[["P1_SALES", "P2_SALES", "P3_SALES", "P4_SALES"]].sum() / df[["P1_SALES", "P2_SALES", "P3_SALES", "P4_SALES"]].sum().sum() *100
print(product_contribution)

product_revenue = df[["P1_REVENUE","P2_REVENUE","P3_REVENUE","P4_REVENUE"]].sum()
product_revenue_contribution = (product_revenue/ product_revenue.sum()) * 100
print(product_revenue_contribution)


""" EXPORT AS A CSV FILE 
product_contribution = {
'Product': ['P1', 'P2', 'P3', 'P4'], 
'Contribution (%)': [39.17, 20.24, 29.90, 10.6]
}


print(product_contribution)


product_revenue_contribution = {
    'Product' : ['P1', 'P2', 'P3', 'P4'], 
    'Revenue Contribution (%)' : [25.31, 26.16, 33.02, 15.51]
}
print(product_revenue_contribution)

product_df = pd.DataFrame(product_contribution)
revenue_df = pd.DataFrame(product_revenue_contribution)

combined_df = pd.merge(product_df, revenue_df, on='Product')

"""



# Q5. Forecast Sales for 2023

Yearly_Sales = df.groupby('Years')[["P1_SALES","P2_SALES","P3_SALES","P4_SALES","P1_REVENUE","P2_REVENUE","P3_REVENUE","P4_REVENUE"]].sum()


print(type(Yearly_Sales))

print(Yearly_Sales.head())

print(Yearly_Sales)
from sklearn.linear_model import LinearRegression
import numpy as np 

print(df['Years'].head(10))


print(df[df['Months'].isnull()])

df = df.dropna(subset=["Months"])
print(df['Months'].isnull().sum())


df["Months"] = pd.to_datetime(df["Months"], errors= "coerce")
print(df["Months"].dtype)
df["Years"] = pd.DatetimeIndex(df["Months"]).year

print(df.head())

X = Yearly_Sales['Years'].values
y_p1 = Yearly_Sales["P1_SALES"].values
y_p2 = Yearly_Sales["P2_SALES"].values
y_p3 = Yearly_Sales["P3_SALES"].values
y_p4 = Yearly_Sales["P4_SALES"].values


model_p1 = LinearRegression().fit(X, y_p1)
model_p2 = LinearRegression().fit(X, y_p2)
model_p3 = LinearRegression().fit(X, y_p3)
model_p4 = LinearRegression().fit(X, y_p4)

year_2023 = np.array([[2023]])


sales_2023_p1 = model_p1.predict(year_2023)
sales_2023_p2 = model_p2.predict(year_2023)
sales_2023_p3 = model_p3.predict(year_2023)
sales_2023_p4 = model_p4.predict(year_2023)








