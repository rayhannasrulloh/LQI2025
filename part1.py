from main import pd

# ------------------------------------------------------
# PART 1: DATA PREPARATION
# ------------------------------------------------------
print("="*50)
print("PART 1: DATA PREPARATION")
print("="*50)

# Create DataFrame from processed survey data
# Demographic Data
genders = ['Female'] * 14 + ['Male'] * 6 + ['Other'] * 2
study_years = [2024] * 16 + [2023] * 5 + [2022] * 1
duration_stays = ['< 6 Months'] * 4 + ['6 Months - 1 Year'] * 13 + ['1 - 2 Years'] * 5

# Satisfaction Data
package_satisfaction = [3, 2, 1, 2, 5, 5, 3, 3, 4, 2, 4, 3, 3, 4, 3, 3, 5, 3, 2, 3, 4, 3]
internet_satisfaction = [1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 3, 2, 2, 3, 3, 2, 4, 1, 2, 3, 1, 2]
overall_quality = [4, 3, 1, 2, 3, 4, 4, 3, 4, 3, 3, 3, 4, 4, 3, 2, 5, 3, 4, 2, 4, 4]

# Internet Stability and Mobile Data Usage
internet_stability = ['No, frequently unstable'] * 14 + ['Yes, but sometimes slow'] * 8
mobile_data_usage = ['Yes, almost every day'] * 14 + ['Yes, a few times a week'] * 6 + ['Rarely, only sometimes'] * 2

# Monthly Spending and Expense Categories
monthly_spending = ['Rp 1,000,000 - Rp 2,000,000'] * 8 + ['Rp 500,000 - Rp 1,000,000'] * 7 + \
                   ['More than Rp 2,000,000'] * 5 + ['Less than Rp 500,000'] * 2

# Binary indicators for expense categories (1 = major expense, 0 = not a major expense)
food_expense = [1] * 16 + [0] * 6
entertainment_expense = [1] * 8 + [0] * 14
transportation_expense = [1] * 7 + [0] * 15
academic_expense = [1] * 4 + [0] * 18
online_shopping_expense = [1] * 9 + [0] * 13
daily_needs_expense = [1] * 1 + [0] * 21

# Create dataframe
data = {
    'Gender': genders,
    'Study_Year': study_years,
    'Duration_Stay': duration_stays,
    'Package_Satisfaction': package_satisfaction,
    'Internet_Satisfaction': internet_satisfaction,
    'Overall_Quality': overall_quality,
    'Internet_Stability': internet_stability,
    'Mobile_Data_Usage': mobile_data_usage,
    'Monthly_Spending': monthly_spending,
    'Food_Expense': food_expense,
    'Entertainment_Expense': entertainment_expense,
    'Transportation_Expense': transportation_expense,
    'Academic_Expense': academic_expense,
    'Online_Shopping_Expense': online_shopping_expense,
    'Daily_Needs_Expense': daily_needs_expense
}

df = pd.DataFrame(data)

# Display dataframe information
print("Dataset Information:")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print("\nFirst 5 rows of data:")
print(df.head())