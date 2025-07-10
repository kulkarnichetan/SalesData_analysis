import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind

df = pd.read_csv('Sample_Data.csv', encoding='ISO-8859-1')


# Function to view records
def view_records():
   while True:
       print("1. Full Sales Record \n")
       print("2. First 5 Sales \n")
       print("3. Last 5 Sales \n")
       print("4. Exit\n")
       choice = int(input("Select Requried Option : "))

       if choice == 4:
           print("Thank You !")
           print("\n")
           break

       elif choice == 1:
           df = pd.read_csv('Sample_Data.csv', encoding='ISO-8859-1')
           print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))
           print("\n")

       elif choice == 2:
           df = pd.read_csv('Sample_Data.csv', encoding='ISO-8859-1')
           print(tabulate(df.head(), headers='keys', tablefmt='pretty', showindex=False))
           print("\n")

       elif choice == 3:
           df = pd.read_csv('Sample_Data.csv', encoding='ISO-8859-1')
           print(tabulate(df.tail(), headers='keys', tablefmt='pretty', showindex=False))
           print("\n")


       else:
           print("Invalid Choice Kindly Select Correct Option")
           print("\n")
       

# Function to add records
def add_records():
    date = input("Enter Data (YYYY-MM-DD) : ")
    product = input("Enter Product Name : ")
    units = int(input("Enter Number of Units Sold : "))
    price = float(input("Enter Unit Price : "))
    revenue = units * price

    new_data = {'Date': date, 'Product': product, 'Units Sold': units, 'Unit Price': price, 'Total Revenue': revenue} 
    df = pd.read_csv('Sample_Data.csv', encoding='ISO-8859-1')
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index = True)
    df.to_csv('Sample_Data.csv', index=False)
    print("Data Added Successfully !")
    print("\n")

# Function to update Record
def update_records():
    df = pd.read_csv('Sample_Data.csv', encoding='ISO-8859-1')
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=True))

    try:
        idx = int(input("Enter row number to update: "))
        if idx < 0 or idx >= len(df.index):
            print("Invalid row index!\n")
            return
    except ValueError:
        print("Invalid input. Please enter a number.\n")
        return

    col = input("Enter column name to update (Product/Quantity/Unit Price): ").strip()
    if col not in ['Product', 'Quantity', 'Unit Price']:
        print("Invalid column name.\n")
        return

    new_val = input("Enter new value: ")
    if col in ['Quantity', 'Unit Price']:
        try:
            new_val = float(new_val)
        except ValueError:
            print("Invalid numeric value.\n")
            return

    df.at[idx, col] = new_val

    # Update Total Revenue if relevant columns changed
    df['Total Revenue'] = df['Units Sold'].astype(float) * df['Unit Price'].astype(float)
    df.to_csv('Sample_Data.csv', index=False)
    print("\nRecord updated successfully.\n")



# Function to delete Record 
def delete_records():
    df = pd.read_csv('Sample_Data.csv', encoding='ISO-8859-1')
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=True))

    try:
        idx = int(input("Enter Row Number To Delete : "))
        if idx < 0 or idx >= len(df.index):
            print("Invalid row index!\n")
            return
    except ValueError:
        print("Invalid input. Please enter a number.\n")
        return

    df.drop(index=idx, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv('Sample_Data.csv', index=False)
    print("\nRecord Deleted Successfully.\n")

    
# Function to View Product-Wise Sales Trend
def product_wise_sales_trend():

     df = pd.read_csv('Sample_Data.csv', encoding='ISO-8859-1')
     df['Date'] = pd.to_datetime(df['Date'])
     df['Month'] = df['Date'].dt.month
     df['Year'] = df['Date'].dt.year

     monthly = df.groupby(['Year', 'Month'])['Total Revenue'].sum().reset_index()

     plt.plot(monthly['Month'], monthly['Total Revenue'], marker='o', linestyle='-', color='hotpink')
     plt.title("Monthly Sales Trend")
     plt.xlabel("Month")
     plt.ylabel("Total Revenue")
     plt.grid(True)
     plt.show()
     print("\n")  

# Function to view Product-wise Sale Summary
def product_wise_sales_summary():
    df = pd.read_csv('Sample_Data.csv', encoding='ISO-8859-1')
    product_sale = df.groupby('Product')['Total Revenue'].sum().reset_index()

    plt.bar(product_sale['Product'], product_sale['Total Revenue'], color='blue')
    plt.title("Product-Wise Sales Summary")
    plt.xlabel("Product")
    plt.ylabel("Total-Revenue")            
    plt.show()
    print("\n")

# Function to view 7-Day Moving Average
def moving_average_trend():

    df = pd.read_csv('Sample_Data.csv', encoding='ISO-8859-1')

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['7_day_avg'] = df['Total Revenue'].rolling(window=7).mean()

    plt.plot(df['Date'], df['Total Revenue'], label="Daily-Sales")
    plt.plot(df['Date'], df['7_day_avg'], label='7-Day Moving Average', color='red')
    plt.xlabel('Date')
    plt.ylabel('Total-Revenue')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("\n")
   
# Sales Summary Report
def generate_sales_report():
    df = pd.read_csv('Sample_Data.csv', encoding='ISO-8859-1')
    print("\nSALES REPORT SUMMARY:\n")
    summary = df.describe(include='all')
    print(tabulate(summary, headers='keys', tablefmt='pretty'))
    print("\n")

# Numpy Stats & Revenue category
def numpy_stats_and_revenue_category():
    df = pd.read_csv('Sample_Data.csv', encoding='ISO-8859-1')
    
    # Create revenue category using NumPy
    conditions = [
        (df['Total Revenue'] >= 10000),
        (df['Total Revenue'] >= 5000) & (df['Total Revenue'] < 10000),
        (df['Total Revenue'] < 5000)
    ]
    choices = ['High', 'Medium', 'Low']
    df['Revenue Category'] = np.select(conditions, choices, default='Unknown')


    print("\nUpdated Data with Revenue Categories:\n")
    print(tabulate(df[['Date', 'Product', 'Total Revenue', 'Revenue Category']], headers='keys', tablefmt='pretty', showindex=False))

    # Basic statistics with NumPy
    revenue = df['Total Revenue'].values
    print("\nRevenue Stats Using NumPy:")
    print(f"Mean Revenue: {np.mean(revenue):.2f}")
    print(f"Standard Deviation: {np.std(revenue):.2f}")
    print(f"Max Revenue: {np.max(revenue):.2f}")
    print(f"Min Revenue: {np.min(revenue):.2f}")
    print("\n")

# Seaborn Visualizations
def seaborn_visualizations():
    df = pd.read_csv('Sample_Data.csv', encoding='ISO-8859-1')
    df['Date'] = pd.to_datetime(df['Date'])

    # Lineplot for sales trend
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_sales = df.groupby('Month')['Total Revenue'].sum().reset_index()
    monthly_sales['Month'] = monthly_sales['Month'].astype(str)

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.lineplot(x='Month', y='Total Revenue', data=monthly_sales, marker='o', color='blue')
    plt.xticks(rotation=45)
    plt.title('Monthly Sales Trend')
    plt.tight_layout()
    plt.show()

    # Barplot for product revenue
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Product', y='Total Revenue', data=df, estimator=sum, ci=None)
    plt.title('Total Revenue by Product')
    plt.tight_layout()
    plt.show()

# Scipy Stats
def scipy_statistical_test():
    df = pd.read_csv('Sample_Data.csv', encoding='ISO-8859-1')
    products = df['Product'].unique()

    print(f"Available Products: {products}")
    p1 = input("Enter first product name: ")
    p2 = input("Enter second product name: ")

    if p1 not in products or p2 not in products:
        print("Invalid product names.\n")
        return

    rev1 = df[df['Product'] == p1]['Total Revenue']
    rev2 = df[df['Product'] == p2]['Total Revenue']

    t_stat, p_val = ttest_ind(rev1, rev2, equal_var=False)
    print(f"\nT-Statistic = {t_stat:.4f}")
    print(f"P-Value = {p_val:.4f}")

    if p_val < 0.05:
        print(f"There IS a significant difference in revenue between {p1} and {p2}.")
    else:
        print(f"There is NO significant difference in revenue between {p1} and {p2}.\n")



# Main loop 
while True:
    print("1.  View Sales Record\n")
    print("2.  Add Sales Record\n")
    print("3.  Update Sales Record\n")
    print("4.  Delete Sales Record\n")
    print("5.  View Monthly Sales Trend \n")
    print("6.  View Product-wise Sales Trend\n")
    print("7.  Show 7-Days Moving Average \n")
    print("8.  Generate Sales Report \n")
    print("9.  View Numpy Stats & Revenue Category\n")
    print("10. Seaborn Visualization\n")
    print("11. Statistical Comparison (SciPy T-test)\n")
    print("12. Exit\n")
    choice = int(input("Enter your Choice: "))

    if choice == 12:
        print("Program Ended Successfully!")
        break

    if choice == 1:
        print("Here's The Sales Record \n")
        view_records()
    
    elif choice == 2:
        print("Add Sales Record functionality goes here\n")
        add_records()
    
    elif choice == 3:
        print("Update Sales Record functionality goes here\n")
        update_records()
    
    elif choice == 4:
        print("Delete Sales Record functionality goes here\n")
        delete_records()
    
    elif choice == 5:
        print("Product-Wise Sales Trend \n")
        product_wise_sales_trend()

    elif choice == 6:
        print("View Product-Wise Sales Trend\n")
        product_wise_sales_summary()

    elif choice == 7:
        print("7-Day Moving Average\n")
        moving_average_trend()

    elif choice == 8:
        print("Generated Sales Report\n")
        generate_sales_report()

    elif choice == 9:
        print("Numpy Stats & Revenue Category\n")
        numpy_stats_and_revenue_category()

    elif choice == 10:
        print("Seaborn Visualization\n")
        seaborn_visualizations()
    
    elif choice == 11:
        print("Statistical Comparison (SciPy T-test)\n")
        scipy_statistical_test()
        
    