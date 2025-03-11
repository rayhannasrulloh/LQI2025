import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import math
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import io

st.set_page_config(
    page_title="Dormitory Survey Analysis",
    page_icon="🏠",
    layout="centered"
)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("pastel")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

np.random.seed(42)

st.title("Comprehensive Statistical Analysis of Dormitory Survey Data")
st.subheader("President University Student Dormitory Survey", divider="red")

tab1, tab2, tab3, tab4 = st.tabs([
    "Data Overview", 
    "Statistical Analysis", 
    "Data Visualization", 
    "Statistical Analysis"
    ])

# ------------------------------------------------------
#           PART 1: DATA PREPARATION
# ------------------------------------------------------
with tab1:
    st.header("Data Overview")
    
    #create DataFrame from processed survey data
    #demographic Data
    genders = ['Female'] * 14 + ['Male'] * 6 + ['Other'] * 2
    study_years = [2024] * 16 + [2023] * 5 + [2022] * 1
    duration_stays = ['< 6 Months'] * 4 + ['6 Months - 1 Year'] * 13 + ['1 - 2 Years'] * 5

    #satisfaction Data
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
    st.subheader("Dataset Information:")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Number of rows: {df.shape[0]}")
    with col2:
        st.write(f"Number of columns: {df.shape[1]}")
    
    # Show the actual data with interactive table
    st.subheader("Raw Survey Data")
    st.dataframe(df, use_container_width=True)
    
    # Add a data download option
    st.download_button(
        label="Download Data as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='dormitory_survey_data.csv',
        mime='text/csv',
    )
    
    # Show demographic overview with charts
    st.subheader("Demographic Overview")
    
    demo_col1, demo_col2, demo_col3 = st.columns(3)
    
    with demo_col1:
        fig, ax = plt.subplots()
        df['Gender'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, startangle=90)
        plt.title('Gender Distribution')
        plt.ylabel('')
        st.pyplot(fig)
    
    with demo_col2:
        fig, ax = plt.subplots()
        df['Study_Year'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, startangle=90)
        plt.title('Study Year Distribution')
        plt.ylabel('')
        st.pyplot(fig)
    
    with demo_col3:
        fig, ax = plt.subplots()
        df['Duration_Stay'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, startangle=90)
        plt.title('Duration of Stay')
        plt.ylabel('')
        st.pyplot(fig)

# ------------------------------------------------------
#           PART 2: DESCRIPTIVE STATISTICS
# ------------------------------------------------------
with tab2:
    st.header("Descriptive Statistics")
    
    # Define numeric and categorical columns
    numeric_cols = ['Package_Satisfaction', 'Internet_Satisfaction', 'Overall_Quality']
    categorical_cols = ['Gender', 'Duration_Stay', 'Internet_Stability', 'Mobile_Data_Usage', 'Monthly_Spending']

    # Calculate descriptive statistics for numeric variables
    st.subheader("Descriptive Statistics for Satisfaction Variables")
    desc_stats = df[numeric_cols].describe()
    st.dataframe(desc_stats, use_container_width=True)
    
    # Visualization of satisfaction scores
    st.subheader("Satisfaction Scores Distribution")
    
    # Allow user to select the variable to view
    selected_var = st.selectbox("Select Satisfaction Variable:", numeric_cols)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df[selected_var], bins=5, kde=True, ax=ax)
        plt.title(f'Distribution of {selected_var}')
        plt.xlabel('Satisfaction Score (1-5)')
        plt.ylabel('Count')
        st.pyplot(fig)
    
    with col2:
        # Calculate and display detailed statistics for the selected variable
        mean = df[selected_var].mean()
        median = df[selected_var].median()
        mode = df[selected_var].mode()[0]
        variance = df[selected_var].var()
        std_dev = df[selected_var].std()
        cv = (std_dev / mean) * 100  # Coefficient of Variation
        skewness = df[selected_var].skew()
        kurtosis = df[selected_var].kurtosis()
        
        st.write(f"Mean (μ) = {mean:.4f}")
        st.write(f"Median = {median}")
        st.write(f"Mode = {mode}")
        st.write(f"Variance (σ²) = {variance:.4f}")
        st.write(f"Standard Deviation (σ) = {std_dev:.4f}")
        st.write(f"Coefficient of Variation (CV) = {cv:.2f}%")
        st.write(f"Skewness = {skewness:.4f}")
        st.write(f"Kurtosis = {kurtosis:.4f}")
        st.write(f"Range = {df[selected_var].max() - df[selected_var].min()}")
        st.write(f"IQR = {df[selected_var].quantile(0.75) - df[selected_var].quantile(0.25)}")
    
    # Display frequency distributions for categorical variables
    st.subheader("Frequency Distributions for Categorical Variables")
    
    # Allow user to select categorical variable
    selected_cat = st.selectbox("Select Categorical Variable:", categorical_cols)
    
    # Calculate and display distribution
    counts = df[selected_cat].value_counts()
    percentages = (counts / len(df) * 100).round(2)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig, ax = plt.subplots()
        sns.barplot(x=counts.index, y=counts.values, ax=ax)
        plt.title(f'Distribution of {selected_cat}')
        plt.xticks(rotation=45 if len(counts) > 3 else 0)
        plt.ylabel('Count')
        st.pyplot(fig)
    
    with col2:
        # Display distribution table
        distribution_df = pd.DataFrame({
            'Count': counts,
            'Percentage (%)': percentages
        })
        st.dataframe(distribution_df)
    
    # Add correlation analysis
    st.subheader("Correlation Between Satisfaction Variables")
    
    # Calculate and display correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Satisfaction Variables')
    st.pyplot(fig)
    
    # Add explanation of the correlations
    st.write("""
    **Interpreting Correlations:**
    - Values close to 1 indicate strong positive correlation
    - Values close to -1 indicate strong negative correlation
    - Values close to 0 indicate weak or no correlation
    """)
    
    # option to explore relationships between variables
    st.subheader("Explore Relationships Between Variables")
    
    # select variables to compare
    x_var = st.selectbox("Select X-axis variable:", df.columns, index=3)
    y_var = st.selectbox("Select Y-axis variable:", df.columns, index=5)
    
    # Create appropriate plot based on variable types
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if df[x_var].dtype in ['int64', 'float64'] and df[y_var].dtype in ['int64', 'float64']:
        # Both numeric - scatterplot
        sns.scatterplot(data=df, x=x_var, y=y_var, ax=ax)
        plt.title(f'Relationship between {x_var} and {y_var}')
    elif df[x_var].dtype in ['int64', 'float64']:
        # X numeric, Y categorical - boxplot
        sns.boxplot(data=df, x=x_var, y=y_var, ax=ax)
        plt.title(f'Distribution of {x_var} across {y_var} categories')
    elif df[y_var].dtype in ['int64', 'float64']:
        # X categorical, Y numeric - boxplot
        sns.boxplot(data=df, x=x_var, y=y_var, ax=ax)
        plt.title(f'Distribution of {y_var} across {x_var} categories')
    else:
        # Both categorical - count plot
        grouped_data = df.groupby([x_var, y_var]).size().reset_index(name='Count')
        sns.barplot(data=grouped_data, x=x_var, y='Count', hue=y_var, ax=ax)
        plt.title(f'Count of {y_var} by {x_var}')
        plt.xticks(rotation=45)
    
    st.pyplot(fig)

# ------------------------------------------------------
#           PART 3: DATA VISUALIZATION
# ------------------------------------------------------
with tab3:
    st.header("Data Visualization")
    st.subheader("Demographic Distribution of Survey Respondents")
    demo_col1, demo_col2, demo_col3 = st.columns(3)

    with demo_col1:
        st.write("**Gender Distribution**")
        fig, ax = plt.subplots(figsize=(4, 4))
        gender_counts = df['Gender'].value_counts()
        gender_percentages = (gender_counts / len(df) * 100).round(1)
        ax.pie(gender_counts, labels=[f"{gender}\n{count} ({percentage}%)" 
                                    for gender, count, percentage in 
                                    zip(gender_counts.index, gender_counts, gender_percentages)], 
            autopct='', startangle=90, colors=sns.color_palette("pastel"))
        ax.set_title('Gender Distribution')
        st.pyplot(fig)

    with demo_col2:
        st.write("**Study Year Distribution**")
        fig, ax = plt.subplots(figsize=(4, 4))
        year_counts = df['Study_Year'].value_counts().sort_index()
        year_percentages = (year_counts / len(df) * 100).round(1)
        bars = ax.bar(year_counts.index.astype(str), year_counts.values, color=sns.color_palette("pastel"))
        ax.set_title('Study Year Distribution')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Students')
        
        # Add count and percentage labels to bars
        for i, (bar, percentage) in enumerate(zip(bars, year_percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height} ({percentage}%)', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)

    with demo_col3:
        st.write("**Duration of Stay Distribution**")
        fig, ax = plt.subplots(figsize=(4, 4))
        duration_order = ['< 6 Months', '6 Months - 1 Year', '1 - 2 Years']
        duration_counts = df['Duration_Stay'].value_counts().reindex(duration_order)
        duration_percentages = (duration_counts / len(df) * 100).round(1)
        bars = ax.bar(duration_counts.index, duration_counts.values, color=sns.color_palette("pastel"))
        ax.set_title('Duration of Stay in Dormitory')
        ax.set_xticks(range(len(duration_counts)))
        ax.set_xticklabels(duration_counts.index, rotation=45, ha='right')
        ax.set_ylabel('Number of Students')
        
        # Add count and percentage labels to bars
        for i, (bar, percentage) in enumerate(zip(bars, duration_percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height} ({percentage}%)', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)

    #add download buttons for generated plots
    st.subheader("Download Visualizations")

    #function to create and save plots
    def get_demographic_plot():
        fig = plt.figure(figsize=(18, 6))
        plt.suptitle("Demographic Distribution of Survey Respondents", fontsize=16)
        
        # gender Distribution
        plt.subplot(1, 3, 1)
        gender_counts = df['Gender'].value_counts()
        gender_percentages = (gender_counts / len(df) * 100).round(1)
        plt.pie(gender_counts, labels=[f"{gender}\n{count} ({percentage}%)" 
                                    for gender, count, percentage in 
                                    zip(gender_counts.index, gender_counts, gender_percentages)], 
                autopct='', startangle=90, colors=sns.color_palette("pastel"))
        plt.title('Gender Distribution')
        
        # Study Year Distribution
        plt.subplot(1, 3, 2)
        year_counts = df['Study_Year'].value_counts().sort_index()
        year_percentages = (year_counts / len(df) * 100).round(1)
        bars = plt.bar(year_counts.index.astype(str), year_counts.values, color=sns.color_palette("pastel"))
        plt.title('Study Year Distribution')
        plt.xlabel('Year')
        plt.ylabel('Number of Students')
        
        for i, (bar, percentage) in enumerate(zip(bars, year_percentages)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height} ({percentage}%)', ha='center', va='bottom')
        
        plt.subplot(1, 3, 3)
        duration_order = ['< 6 Months', '6 Months - 1 Year', '1 - 2 Years']
        duration_counts = df['Duration_Stay'].value_counts().reindex(duration_order)
        duration_percentages = (duration_counts / len(df) * 100).round(1)
        bars = plt.bar(duration_counts.index, duration_counts.values, color=sns.color_palette("pastel"))
        plt.title('Duration of Stay in Dormitory')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Number of Students')
        
        # Add count and percentage labels to bars
        for i, (bar, percentage) in enumerate(zip(bars, duration_percentages)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height} ({percentage}%)', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig

    download_col1, download_col2 = st.columns(2)

    with download_col1:
        # Save demographic plot
        demographic_fig = get_demographic_plot()
        demographic_buf = io.BytesIO()
        demographic_fig.savefig(demographic_buf, format='png', dpi=300)
        demographic_buf.seek(0)
        
        st.download_button(
            label="Download Demographic Plot",
            data=demographic_buf,
            file_name="demographic_distribution.png",
            mime="image/png"
        )

# ------------------------------------------------------
#           PART 4: STATISTICAL ANALYSIS
# ------------------------------------------------------
with tab4:
    st.title("Student Satisfaction Data Analysis")
    st.write("This application analyzes student satisfaction data, focusing on internet satisfaction, package satisfaction, and overall quality of life.")

    ttab1, ttab2, ttab3, ttab4, ttab5, ttab6, summ = st.tabs([
        "Satisfaction Analysis",
        "Expense Analysis", 
        "Internet Usage Analysis", 
        "Probability Distributions",
        "Hypothesis Testing",
        "Advanced Analysis",
        "Summary"
        ])

    # TAB 1: SATISFACTION ANALYSIS
    with ttab1:
        st.header("Satisfaction Analysis")

        # A. Distribution Histograms
        st.subheader("A. Distribution Histograms", divider=True)
        
        categories = ['Package Pickup System', 'Internet Speed', 'Overall Quality of Life']
        colors = ['skyblue', 'salmon', 'lightgreen']
        data_cols = ['Package_Satisfaction', 'Internet_Satisfaction', 'Overall_Quality']
        
        hist_cols = st.columns(3)
        
        for i, (category, col, color) in enumerate(zip(categories, data_cols, colors)):
            with hist_cols[i]:
                st.write(f"**{category}**")
                fig, ax = plt.subplots(figsize=(4, 4))
                
                # Calculate statistics
                mean_val = df[col].mean()
                median_val = df[col].median()
                mode_val = df[col].mode()[0]
                std_dev = df[col].std()
                
                # Create histogram
                sns.histplot(df[col], kde=True, bins=5, color=color, ax=ax)
                
                # Add vertical lines for mean, median, and mode
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean = {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='-.', alpha=0.7, label=f'Median = {median_val}')
                ax.axvline(mode_val, color='purple', linestyle=':', alpha=0.7, label=f'Mode = {mode_val}')
                
                # Calculate and display normal distribution formula
                x = np.linspace(0, 6, 100)
                y = stats.norm.pdf(x, mean_val, std_dev)
                max_height = ax.get_ylim()[1]
                y_scaled = y * (max_height / max(y)) * 0.8  # Scale to fit on histogram
                
                # Add normal distribution curve formula to the plot
                normal_formula = r"$f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$"
                values = f"$\mu={mean_val:.2f}, \sigma={std_dev:.2f}$"
                ax.text(0.95, 0.95, normal_formula + "\n" + values, 
                        transform=ax.transAxes, ha='right', va='top', 
                        bbox=dict(facecolor='white', alpha=0.7))
                
                ax.set_title(f'Distribution of {category} Satisfaction')
                ax.set_xlabel('Satisfaction Score (1-5)')
                ax.set_ylabel('Frequency')
                ax.set_xticks(range(1, 6))
                ax.legend(loc='upper left')
                
                plt.tight_layout()
                st.pyplot(fig)

        # B. Box Plot Comparison
        st.subheader("B. Box Plot Comparison", divider=True)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            satisfaction_data = df[['Package_Satisfaction', 'Internet_Satisfaction', 'Overall_Quality']]
            sns.boxplot(data=satisfaction_data, ax=ax, palette=colors)
            ax.set_title('Comparison of Satisfaction Distributions')
            ax.set_ylabel('Satisfaction Score (1-5)')
            ax.set_xticklabels(['Package\nPickup', 'Internet\nSpeed', 'Overall\nQuality'])
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
                st.markdown("""
                **Box Plot Elements:**
                - Box: IQR (25th to 75th percentile)
                - Line in Box: Median
                - Whiskers: 1.5 x IQR or min/max
                - Points: Outliers
                
                **Statistics Summary:**
                """)
                
                # Display summary statistics
                summary = satisfaction_data.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
                summary.columns = ['Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max']
                summary.index = ['Package Pickup', 'Internet Speed', 'Overall Quality']
                st.dataframe(summary.round(2))

        # C. Correlation Analysis
        st.subheader("Correlation Analysis", divider=True)
        corr_col1, corr_col2 = st.columns(2)
        
        with corr_col1:
            # Scatterplot between Internet Satisfaction and Overall Quality
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.regplot(x='Internet_Satisfaction', y='Overall_Quality', data=df, 
                        scatter_kws={'alpha':0.6}, line_kws={'color':'red'}, ax=ax)
            
            # Calculate correlation
            corr_internet_quality, p_value = stats.pearsonr(df['Internet_Satisfaction'], df['Overall_Quality'])
            
            # Annotate with correlation formula and value
            ax.set_title('Internet Satisfaction vs. Overall Quality')
            ax.set_xlabel('Internet Satisfaction (1-5)')
            ax.set_ylabel('Overall Quality (1-5)')
            ax.set_xticks(range(1, 6))
            ax.set_yticks(range(1, 6))
            
            # Add correlation information
            corr_text = (f"Pearson Correlation: r = {corr_internet_quality:.4f}\n"
                            f"p-value = {p_value:.4f}\n\n"
                            r"$r = \frac{\sum(x-\bar{x})(y-\bar{y})}{\sqrt{\sum(x-\bar{x})^2 \sum(y-\bar{y})^2}}$")
            ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.7))
            
            # Add jitter to scatter points for better visualization of overlapping points
            ax.collections[0].set_offset_transform(ax.transData + 
                                                    plt.matplotlib.transforms.Affine2D().translate(0.1, 0.1))
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with corr_col2:
            # Heatmap for correlation matrix
            correlation_matrix = df[['Package_Satisfaction', 'Internet_Satisfaction', 'Overall_Quality']].corr()
            
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.4f', linewidths=0.5, ax=ax)
            ax.set_title('Correlation Matrix of Satisfaction Variables')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Add correlation interpretation
            st.markdown("""
            **Correlation Interpretation:**
            - Strong positive: 0.7 to 1.0
            - Moderate positive: 0.3 to 0.7
            - Weak positive: 0 to 0.3
            - Weak negative: -0.3 to 0
            - Moderate negative: -0.7 to -0.3
            - Strong negative: -1.0 to -0.7
            """)

    # TAB 2: EXPENSE ANALYSIS
    with ttab2:
        st.header("💰 Expense Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Monthly Spending Distribution")
            
            # Get data in correct order
            spending_order = ['Less than Rp 500,000', 'Rp 500,000 - Rp 1,000,000', 
                            'Rp 1,000,000 - Rp 2,000,000', 'More than Rp 2,000,000']
            spending_counts = df['Monthly_Spending'].value_counts().reindex(spending_order)
            spending_percentages = (spending_counts / len(df) * 100).round(1)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(range(len(spending_counts)), spending_counts.values, color=sns.color_palette("pastel"))
            ax.set_xticks(range(len(spending_counts)))
            ax.set_xticklabels(spending_counts.index, rotation=45, ha='right')
            ax.set_ylabel('Number of Students')
            
            # Add count and percentage labels
            for i, (bar, percentage) in enumerate(zip(bars, spending_percentages)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height} ({percentage}%)', ha='center', va='bottom')
            
            st.pyplot(fig)
            
        with col2:
            st.subheader("Major Expense Categories")
            
            expense_categories = ['Food_Expense', 'Online_Shopping_Expense', 'Entertainment_Expense', 
                                'Transportation_Expense', 'Academic_Expense', 'Daily_Needs_Expense']
            expense_labels = ['Food', 'Online Shopping', 'Entertainment', 'Transportation', 'Academic', 'Daily Needs']
            expense_counts = [df[col].sum() for col in expense_categories]
            expense_percentages = [(count / sum(expense_counts) * 100).round(1) for count in expense_counts]
            
            # Sort by frequency
            sorted_indices = np.argsort(expense_counts)[::-1]  # Descending order
            sorted_labels = [expense_labels[i] for i in sorted_indices]
            sorted_counts = [expense_counts[i] for i in sorted_indices]
            sorted_percentages = [expense_percentages[i] for i in sorted_indices]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(range(len(sorted_counts)), sorted_counts, color=sns.color_palette("pastel"))
            ax.set_xticks(range(len(sorted_counts)))
            ax.set_xticklabels(sorted_labels, rotation=45, ha='right')
            ax.set_ylabel('Total Expense')
            
            # Add count and percentage labels
            for i, (bar, percentage) in enumerate(zip(bars, sorted_percentages)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height} ({percentage}%)', ha='center', va='bottom')
            
            st.pyplot(fig)
        
        # Chi-square analysis
        st.subheader("Chi-Square Analysis of Expense Distribution")
        
        # Calculate Chi-square
        n = len(df)
        k = len(expense_categories)
        observed_freq = expense_counts
        expected_freq = [sum(expense_counts)/k] * k  # Equal distribution hypothesis
        
        chi_sq = sum([(o - e)**2 / e for o, e in zip(observed_freq, expected_freq)])
        dof = k - 1
        p_value = 1 - stats.chi2.cdf(chi_sq, dof)
        critical_value = stats.chi2.ppf(0.95, dof)
        
        # Create bar chart showing observed vs expected
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(expense_labels))
        width = 0.35
        
        ax.bar(x - width/2, observed_freq, width, label='Observed', color='skyblue')
        ax.bar(x + width/2, expected_freq, width, label='Expected (Equal Distribution)', color='lightgreen')
        
        ax.set_xticks(x)
        ax.set_xticklabels(expense_labels, rotation=45, ha='right')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Add Chi-square test results
        chi_text = (f"Chi-square statistic = {chi_sq:.2f}\n"
                    f"Degrees of freedom = {dof}\n"
                    f"p-value = {p_value:.6f}\n"
                    f"Critical value (α=0.05) = {critical_value:.2f}\n\n"
                    r"$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$" + "\n"
                    f"Conclusion: {'Reject' if chi_sq > critical_value else 'Fail to reject'} null hypothesis")
        ax.text(0.02, 0.95, chi_text, transform=ax.transAxes, ha='left', va='top',
                bbox=dict(facecolor='white', alpha=0.7))
        
        st.pyplot(fig)

    # TAB 3: INTERNET USAGE ANALYSIS
    with ttab3:
        st.header("🌐 Internet Usage and Stability Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Internet Stability Distribution")
            
            stability_counts = df['Internet_Stability'].value_counts()
            stability_percentages = (stability_counts / len(df) * 100).round(1)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(stability_counts, labels=[f"{label}\n{count} ({percentage}%)" 
                                            for label, count, percentage in 
                                            zip(stability_counts.index, stability_counts, stability_percentages)], 
                    autopct='', startangle=90, colors=['salmon', 'skyblue'])
            
            st.pyplot(fig)
            
        with col2:
            st.subheader("Mobile Data Usage Distribution")
            
            usage_counts = df['Mobile_Data_Usage'].value_counts()
            usage_percentages = (usage_counts / len(df) * 100).round(1)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(usage_counts, labels=[f"{label}\n{count} ({percentage}%)" 
                                        for label, count, percentage in 
                                        zip(usage_counts.index, usage_counts, usage_percentages)], 
                    autopct='', startangle=90, colors=sns.color_palette("pastel"))
            
            st.pyplot(fig)
        
        # Cross-tabulation analysis
        st.subheader("Relationship Between Internet Stability and Mobile Data Usage")
        
        # Create cross-tabulation
        cross_tab = pd.crosstab(df['Internet_Stability'], df['Mobile_Data_Usage'])
        cross_tab_percentages = pd.crosstab(df['Internet_Stability'], df['Mobile_Data_Usage'], 
                                            normalize='all') * 100
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
            ax.set_xlabel('Mobile Data Usage')
            ax.set_ylabel('Internet Stability')
            st.pyplot(fig)
        
        with col2:
            # Calculate conditional probabilities
            p_daily_mobile = df[df['Mobile_Data_Usage'] == 'Yes, almost every day'].shape[0] / df.shape[0]
            p_unstable = df[df['Internet_Stability'] == 'No, frequently unstable'].shape[0] / df.shape[0]
            
            p_unstable_given_daily_mobile = df[(df['Mobile_Data_Usage'] == 'Yes, almost every day') & 
                                            (df['Internet_Stability'] == 'No, frequently unstable')].shape[0] / \
                                        df[df['Mobile_Data_Usage'] == 'Yes, almost every day'].shape[0]
            
            p_daily_mobile_given_unstable = df[(df['Mobile_Data_Usage'] == 'Yes, almost every day') & 
                                            (df['Internet_Stability'] == 'No, frequently unstable')].shape[0] / \
                                        df[df['Internet_Stability'] == 'No, frequently unstable'].shape[0]
            
            # Display Bayes' theorem information
            st.subheader("Bayes' Theorem Analysis")
            st.latex(r"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}")
            st.write(f"P(Daily Mobile Usage) = {p_daily_mobile:.4f}")
            st.write(f"P(Unstable Internet) = {p_unstable:.4f}")
            st.write(f"P(Unstable Internet | Daily Mobile Usage) = {p_unstable_given_daily_mobile:.4f}")
            st.write(f"P(Daily Mobile Usage | Unstable Internet) = {p_daily_mobile_given_unstable:.4f}")
            
            st.markdown("**Calculation:**")
            st.write(f"P(D|U) = [P(U|D) × P(D)] / P(U)")
            st.write(f"= [{p_unstable_given_daily_mobile:.4f} × {p_daily_mobile:.4f}] / {p_unstable:.4f}")
            st.write(f"= {p_daily_mobile_given_unstable:.4f}")

    # TAB 4: PROBABILITY DISTRIBUTION ANALYSIS
    with ttab4:
        st.header("📊 Probability Distribution Analysis")
        
        # Binomial Distribution Analysis
        st.subheader("Binomial Distribution: Probability of Students with Stable Internet")
        
        # Interactive inputs
        col1, col2 = st.columns(2)
        with col1:
            p_stable_internet = df[df['Internet_Stability'] == 'Yes, but sometimes slow'].shape[0] / df.shape[0]
            st.write(f"Probability of stable internet (p): {p_stable_internet:.4f}")
            
            n_students = st.slider("Number of new students (n):", 5, 20, 10)
            
        with col2:
            k_value = st.slider("Calculate probability for exactly k students:", 0, n_students, 4)
        
        # Calculate binomial probabilities
        x = np.arange(0, n_students + 1)
        binomial_pmf = stats.binom.pmf(x, n_students, p_stable_internet)
        binomial_cdf = stats.binom.cdf(x, n_students, p_stable_internet)
        
        # Plot PMF
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(x, binomial_pmf, alpha=0.7, color='skyblue', label='PMF')
        ax.set_xticks(x)
        ax.set_xlabel("Number of Students with Stable Internet (k)")
        ax.set_ylabel("Probability P(X = k)")
        
        # Highlight selected k value
        if 0 <= k_value <= n_students:
            ax.bar([k_value], [binomial_pmf[k_value]], alpha=1.0, color='red', label=f'P(X = {k_value})')
        
        # Add probability values above bars
        for i, (bar, prob) in enumerate(zip(bars, binomial_pmf)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Add binomial formula
        binomial_formula = (r"$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$" + "\n"
                        f"n = {n_students}, p = {p_stable_internet:.4f}")
        ax.text(0.95, 0.95, binomial_formula, transform=ax.transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.7))
        
        ax.legend()
        st.pyplot(fig)
        
        # Example calculation for selected k
        binom_coef = math.comb(n_students, k_value)
        prob_success = p_stable_internet ** k_value
        prob_failure = (1 - p_stable_internet) ** (n_students - k_value)
        prob_k_example = binom_coef * prob_success * prob_failure
        
        st.markdown("### Example Calculation")
        st.latex(f"P(X = {k_value}) = C({n_students},{k_value}) × {p_stable_internet:.4f}^{k_value} × (1-{p_stable_internet:.4f})^{n_students-k_value}")
        st.latex(f"= {binom_coef} × {prob_success:.6f} × {prob_failure:.6f} = {prob_k_example:.6f}")
        
        # Poisson Distribution Analysis
        st.subheader("Poisson Distribution: Probability of Daily Package Arrivals")
        
        # Interactive lambda input
        lambda_packages = st.slider("Average number of packages per day (λ):", 1, 10, 5)
        k_poisson = st.slider("Calculate probability for exactly k packages:", 0, 15, 8)
        
        # Calculate Poisson probabilities
        x = np.arange(0, 16)
        poisson_pmf = stats.poisson.pmf(x, lambda_packages)
        
        # Plot PMF
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(x, poisson_pmf, alpha=0.7, color='lightgreen', label='PMF')
        
        # Highlight selected k value
        if 0 <= k_poisson <= 15:
            ax.bar([k_poisson], [poisson_pmf[k_poisson]], alpha=1.0, color='red', label=f'P(X = {k_poisson})')
        
        ax.set_xticks(x)
        ax.set_xlabel("Number of Packages per Day (k)")
        ax.set_ylabel("Probability P(X = k)")
        
        # Add probability values above bars
        for i, (bar, prob) in enumerate(zip(bars, poisson_pmf)):
            if i % 2 == 0:  # Show every other value to avoid crowding
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{prob:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Add Poisson formula
        poisson_formula = (r"$P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$" + "\n"
                        f"λ = {lambda_packages}")
        ax.text(0.95, 0.95, poisson_formula, transform=ax.transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.7))
        
        ax.legend()
        st.pyplot(fig)
        
        # Example calculation for k
        numerator = (lambda_packages ** k_poisson) * np.exp(-lambda_packages)
        denominator = math.factorial(k_poisson)
        prob_k_poisson = numerator / denominator
        
        st.markdown("### Example Calculation")
        st.latex(f"P(X = {k_poisson}) = \\frac{{{lambda_packages}^{k_poisson} × e^{{-{lambda_packages}}}}}{{{k_poisson}!}}")
        st.latex(f"= \\frac{{{lambda_packages ** k_poisson:.1f} × {np.exp(-lambda_packages):.6f}}}{{{denominator}}} = {prob_k_poisson:.6f}")
        
        # Normal Distribution Analysis
        st.subheader("Normal Distribution: Overall Quality of Life")
        
        # Calculate mean and std for Overall Quality
        mean_quality = df['Overall_Quality'].mean()
        std_quality = df['Overall_Quality'].std()
        
        # Interactive inputs for normal distribution
        quality_value = st.slider("Select Quality Score:", 1.0, 5.0, 4.0, 0.1)
        
        # Generate x values for plotting
        x = np.linspace(mean_quality - 4*std_quality, mean_quality + 4*std_quality, 1000)
        pdf = stats.norm.pdf(x, mean_quality, std_quality)
        
        # Plot PDF
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, pdf, 'r-', lw=2, label='PDF')
        ax.axvline(mean_quality, color='green', linestyle='--', label=f'Mean = {mean_quality:.2f}')
        ax.axvline(mean_quality + std_quality, color='blue', linestyle=':', label=f'Mean + 1σ = {mean_quality + std_quality:.2f}')
        ax.axvline(mean_quality - std_quality, color='blue', linestyle=':', label=f'Mean - 1σ = {mean_quality - std_quality:.2f}')
        
        # Highlight selected value
        ax.axvline(quality_value, color='purple', linestyle='-', label=f'Selected x = {quality_value}')
        
        # Fill areas for standard deviations
        ax.fill_between(x, 0, pdf, where=(x >= mean_quality - std_quality) & (x <= mean_quality + std_quality),
                        color='lightblue', alpha=0.5, label='68% within 1σ')
        
        # Add histogram of actual data for comparison
        ax.hist(df['Overall_Quality'], bins=5, density=True, alpha=0.3, color='gray', label='Actual Data')
        
        ax.set_xlabel('Overall Quality Score')
        ax.set_ylabel('Probability Density')
        ax.legend(loc='upper left', fontsize=9)
        
        # Add normal distribution formula
        normal_formula = (r"$f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$" + "\n"
                        f"μ = {mean_quality:.4f}, σ = {std_quality:.4f}")
        ax.text(0.95, 0.95, normal_formula, transform=ax.transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.7))
        
        st.pyplot(fig)
        
        # Example calculation for the selected value
        z_score = (quality_value - mean_quality) / std_quality
        prob_density = stats.norm.pdf(quality_value, mean_quality, std_quality)
        
        st.markdown("### Example Calculation")
        st.latex(f"Z\\text{{-score}} = \\frac{{x - \\mu}}{{\\sigma}} = \\frac{{{quality_value} - {mean_quality:.4f}}}{{{std_quality:.4f}}} = {z_score:.4f}")
        st.latex(f"f({quality_value}) = \\frac{{1}}{{({std_quality:.4f}\\sqrt{{2\\pi}})}} × e^{{-0.5 × {z_score:.4f}^2}} = {prob_density:.6f}")
        
        # Confidence Interval for Proportion
        st.subheader("Confidence Interval for Mobile Data Usage Proportion")
        
        # Calculate proportion and CI
        daily_mobile_users = df[df['Mobile_Data_Usage'] == 'Yes, almost every day'].shape[0]
        p_daily_mobile = daily_mobile_users / df.shape[0]
        
        # Interactive confidence level
        confidence_level = st.slider("Confidence Level (%):", 80, 99, 95)
        z_alpha = stats.norm.ppf(1 - (1 - confidence_level/100)/2)
        
        std_error = np.sqrt((p_daily_mobile * (1 - p_daily_mobile)) / df.shape[0])
        margin_error = z_alpha * std_error
        ci_lower = max(0, p_daily_mobile - margin_error)
        ci_upper = min(1, p_daily_mobile + margin_error)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot([0, 1], [1, 1], 'k-', lw=2)  # horizontal line
        ax.plot([p_daily_mobile, p_daily_mobile], [0.9, 1.1], 'r-', lw=2, label=f'Proportion = {p_daily_mobile:.4f}')
        ax.plot([ci_lower, ci_upper], [1, 1], 'b-', lw=4, alpha=0.6, 
            label=f'{confidence_level}% CI: ({ci_lower:.4f}, {ci_upper:.4f})')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0.8, 1.2)
        ax.set_yticks([])
        ax.set_xlabel('Proportion of Students Using Mobile Data Almost Daily')
        ax.legend(loc='upper center')
        
        # Add confidence interval formula
        ci_formula = (r"$p \pm z_{\alpha/2} \sqrt{\frac{p(1-p)}{n}}$" + "\n"
                    f"p = {p_daily_mobile:.4f}, n = {df.shape[0]}, z_{{α/2}} = {z_alpha:.4f}")
        ax.text(0.05, 0.9, ci_formula, transform=ax.transAxes, ha='left', va='top',
                bbox=dict(facecolor='white', alpha=0.7))
        
        st.pyplot(fig)
        
        # Example calculation
        st.markdown("### Example Calculation")
        st.latex(f"\\text{{Standard Error}} = \\sqrt{{\\frac{{p(1-p)}}{{n}}}} = \\sqrt{{\\frac{{{p_daily_mobile:.4f}×{1-p_daily_mobile:.4f}}}{{{df.shape[0]}}}}} = {std_error:.4f}")
        st.latex(f"\\text{{Margin of Error}} = {z_alpha:.4f} × {std_error:.4f} = {margin_error:.4f}")
        st.latex(f"{confidence_level}\\% \\text{{ CI}} = {p_daily_mobile:.4f} \\pm {margin_error:.4f} = ({ci_lower:.4f}, {ci_upper:.4f})")

    # TAB 5: HYPOTHESIS TESTING VISUALIZATION
    with ttab5:
        st.header("Statistical Hypothesis Testing")

        # Independent t-test: Internet Satisfaction by Gender
        st.subheader("A. Independent t-test: Internet Satisfaction by Gender")
        
        # Filter data for female and male
        female_internet = df[df['Gender'] == 'Female']['Internet_Satisfaction']
        male_internet = df[df['Gender'] == 'Male']['Internet_Satisfaction']
        
        # Calculate t-test
        t_stat, p_value = stats.ttest_ind(female_internet, male_internet, equal_var=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create grouped boxplot
        gender_data = df[df['Gender'].isin(['Female', 'Male'])]
        sns.boxplot(x='Gender', y='Internet_Satisfaction', data=gender_data, ax=ax)
        plt.ylabel('Internet Satisfaction Score')
        
        # Add t-test results
        t_test_result = (f"Independent t-test results:\n"
                        f"Female mean (n={len(female_internet)}): {female_internet.mean():.4f}\n"
                        f"Male mean (n={len(male_internet)}): {male_internet.mean():.4f}\n"
                        f"t-statistic: {t_stat:.4f}\n"
                        f"p-value: {p_value:.4f}\n"
                        f"Conclusion: {'Significant' if p_value < 0.05 else 'Not significant'} at α = 0.05")
        plt.text(0.5, 0.05, t_test_result, transform=ax.transAxes, ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7))
        
        # Add t-test formula
        t_formula = (r"$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$")
        plt.text(0.95, 0.95, t_formula, transform=ax.transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.7))
        
        st.pyplot(fig)
        
        # One-way ANOVA: Overall Quality by Duration of Stay
        st.subheader("B. One-way ANOVA: Overall Quality by Duration of Stay")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create boxplot
        sns.boxplot(x='Duration_Stay', y='Overall_Quality', data=df, order=['< 6 Months', '6 Months - 1 Year', '1 - 2 Years'], ax=ax)
        plt.xlabel('Duration of Stay')
        plt.ylabel('Overall Quality Score')
        plt.xticks(rotation=45)
        
        # Perform one-way ANOVA
        groups = [df[df['Duration_Stay'] == d]['Overall_Quality'] for d in ['< 6 Months', '6 Months - 1 Year', '1 - 2 Years']]
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Add ANOVA results
        anova_result = (f"One-way ANOVA results:\n"
                    f"F-statistic: {f_stat:.4f}\n"
                    f"p-value: {p_value:.4f}\n"
                    f"Conclusion: {'Significant' if p_value < 0.05 else 'Not significant'} at α = 0.05")
        plt.text(0.5, 0.05, anova_result, transform=ax.transAxes, ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7))
        
        # Add ANOVA formula
        anova_formula = (r"$F = \frac{MS_{between}}{MS_{within}}$" + "\n"
                        r"$MS_{between} = \frac{SS_{between}}{df_{between}}$" + "\n"
                        r"$MS_{within} = \frac{SS_{within}}{df_{within}}$")
        plt.text(0.95, 0.95, anova_formula, transform=ax.transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.7))
        
        st.pyplot(fig)
        
        # Multiple Regression Analysis
        st.subheader("C. Multiple Regression: Factors Affecting Overall Quality of Life")
        
        # Perform regression
        X = df[['Internet_Satisfaction', 'Package_Satisfaction']]
        X = sm.add_constant(X)
        y = df['Overall_Quality']
        model = sm.OLS(y, X).fit()
        
        # Display regression results
        st.write(f"**Model**: Overall_Quality = {model.params[0]:.4f} + {model.params[1]:.4f}×Internet_Satisfaction + {model.params[2]:.4f}×Package_Satisfaction")
        st.write(f"**R²**: {model.rsquared:.4f}")
        st.write(f"**Adjusted R²**: {model.rsquared_adj:.4f}")
        st.write(f"**F-statistic**: {model.fvalue:.4f}, **p-value**: {model.f_pvalue:.4f}")
        st.write(f"**Internet_Satisfaction**: β = {model.params[1]:.4f}, p = {model.pvalues[1]:.4f}")
        st.write(f"**Package_Satisfaction**: β = {model.params[2]:.4f}, p = {model.pvalues[2]:.4f}")
        
        # create 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Generate grid for prediction surface
        x_pred = np.linspace(df['Internet_Satisfaction'].min(), df['Internet_Satisfaction'].max(), 10)
        y_pred = np.linspace(df['Package_Satisfaction'].min(), df['Package_Satisfaction'].max(), 10)
        xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
        model_pred = np.column_stack([np.ones(xx_pred.flatten().shape), xx_pred.flatten(), yy_pred.flatten()])
        zz_pred = model.predict(model_pred).reshape(xx_pred.shape)
        
        # Plot prediction surface
        surf = ax.plot_surface(xx_pred, yy_pred, zz_pred, rstride=1, cstride=1, alpha=0.4,
                            cmap='viridis', edgecolor='none')
        
        # Plot actual data points
        ax.scatter(df['Internet_Satisfaction'], df['Package_Satisfaction'], df['Overall_Quality'],
                c='red', marker='o', s=50, alpha=0.8)
        
        ax.set_xlabel('Internet Satisfaction')
        ax.set_ylabel('Package Satisfaction')
        ax.set_zlabel('Overall Quality')
        ax.set_xticks(range(1, 6))
        ax.set_yticks(range(1, 6))
        ax.set_zticks(range(1, 6))
        
        # Add regression formula
        regression_formula = (r"$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2$")
        ax.text2D(0.95, 0.05, regression_formula, transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7))
        
        st.pyplot(fig)

    # Tab 6: ADVANCED ANALYSIS: MARKOV CHAIN & CLUSTERING
    with ttab6:
        st.header("Advanced Statistical Analysis", divider="red")

        # A
        st.subheader("A. Markov Chain Analysis: Long-term Internet Stability Prediction")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        transition_matrix = np.array([
            [0.3, 0.7],  # Stable -> Stable, Stable -> Unstable
            [0.2, 0.8]   # Unstable -> Stable, Unstable -> Unstable
        ])
        
        initial_state = np.array([
            df[df['Internet_Stability'] == 'Yes, but sometimes slow'].shape[0] / df.shape[0],
            df[df['Internet_Stability'] == 'No, frequently unstable'].shape[0] / df.shape[0]
        ])
        
        # Calculate states over time
        num_periods = 10
        states = np.zeros((num_periods + 1, 2))
        states[0] = initial_state
        
        for i in range(1, num_periods + 1):
            states[i] = states[i-1].dot(transition_matrix)
        
        # Calculate steady state
        steady_state = np.array([transition_matrix[1,0], transition_matrix[0,1]]) / (transition_matrix[0,1] + transition_matrix[1,0])
        
        # Plot state probabilities over time
        plt.plot(range(num_periods + 1), states[:, 0], 'b-', marker='o', label='Stable Internet')
        plt.plot(range(num_periods + 1), states[:, 1], 'r-', marker='o', label='Unstable Internet')
        plt.axhline(y=steady_state[0], color='b', linestyle='--', alpha=0.7, label=f'Steady State Stable: {steady_state[0]:.4f}')
        plt.axhline(y=steady_state[1], color='r', linestyle='--', alpha=0.7, label=f'Steady State Unstable: {steady_state[1]:.4f}')
        
        plt.xlabel('Time Period')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True)
        
        # Add Markov Chain information
        markov_info = (f"Markov Chain Analysis:\n\n"
                    f"Transition Matrix:\n"
                    f"[{transition_matrix[0,0]:.1f} {transition_matrix[0,1]:.1f}] (Stable -> Stable, Stable -> Unstable)\n"
                    f"[{transition_matrix[1,0]:.1f} {transition_matrix[1,1]:.1f}] (Unstable -> Stable, Unstable -> Unstable)\n\n"
                    f"Initial State: [{initial_state[0]:.4f}, {initial_state[1]:.4f}]\n"
                    f"After 5 periods: [{states[5][0]:.4f}, {states[5][1]:.4f}]\n"
                    f"Steady State: [{steady_state[0]:.4f}, {steady_state[1]:.4f}]")
        plt.text(1.02, 0.5, markov_info, transform=ax.transAxes, ha='left', va='center',
                bbox=dict(facecolor='white', alpha=0.7))
        
        st.pyplot(fig)
        
        # Interactive Markov Chain parameters
        st.write("### Adjust Markov Chain Parameters")
        
        stable_to_stable = st.slider("Probability of Stable → Stable", 0.0, 1.0, 0.3, 0.1)
        unstable_to_stable = st.slider("Probability of Unstable → Stable", 0.0, 1.0, 0.2, 0.1)
        
        if st.button("Calculate Updated Markov Chain"):
            # Create new transition matrix
            new_transition_matrix = np.array([
                [stable_to_stable, 1-stable_to_stable],
                [unstable_to_stable, 1-unstable_to_stable]
            ])
            
            # Calculate new steady state
            new_steady_state = np.array([new_transition_matrix[1,0], new_transition_matrix[0,1]]) / (new_transition_matrix[0,1] + new_transition_matrix[1,0])
            
            st.write(f"**New steady state probability of stable internet**: {new_steady_state[0]:.4f}")
            st.write(f"**New steady state probability of unstable internet**: {new_steady_state[1]:.4f}")
        
        # -----
        #   B
        # -----
        st.subheader("B. K-means Clustering: Student Satisfaction Profiles")
        
        # Prepare data for clustering
        X_cluster = df[['Internet_Satisfaction', 'Package_Satisfaction', 'Overall_Quality']]
        
        # Let user select number of clusters
        n_clusters = st.slider("Number of clusters", 2, 5, 2)
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_cluster)
        
        # Get cluster centers and sizes
        centers = kmeans.cluster_centers_
        cluster_sizes = df['Cluster'].value_counts()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create scatter plot
        sns.scatterplot(x='Internet_Satisfaction', y='Overall_Quality', hue='Cluster', 
                        data=df, palette='viridis', s=100, alpha=0.7, ax=ax)
        plt.scatter(centers[:, 0], centers[:, 2], c='red', marker='X', s=200, 
                    label='Cluster Centers', edgecolors='black')
        
        # Add lines connecting points to their cluster centers
        for i, row in df.iterrows():
            cluster = row['Cluster']
            plt.plot([row['Internet_Satisfaction'], centers[cluster, 0]], 
                    [row['Overall_Quality'], centers[cluster, 2]], 
                    'k-', alpha=0.1)
        
        plt.xlabel('Internet Satisfaction')
        plt.ylabel('Overall Quality')
        plt.legend(title='Cluster')
        plt.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Display cluster information
        st.write("### Cluster Information")
        for i in range(len(centers)):
            st.write(f"**Cluster {i+1}** (n={cluster_sizes.get(i, 0)}):")
            st.write(f"- Internet Satisfaction: {centers[i, 0]:.2f}")
            st.write(f"- Package Satisfaction: {centers[i, 1]:.2f}")
            st.write(f"- Overall Quality: {centers[i, 2]:.2f}")
            
        st.header("Summary of Statistical Measures and Findings")
        findings = [
            "1. Internet satisfaction has the lowest mean (1.95/5) among satisfaction variables",
            "2. 63.64% of students report internet is frequently unstable",
            "3. There is a moderate positive correlation (r = 0.39) between internet satisfaction and overall quality of life",
            "4. Chi-square test confirms uneven distribution of expense categories, with food being the major expense (72.73%)",
            "5. Markov chain analysis predicts internet will be stable only 22.22% of the time in the long run",
            "6. Cluster analysis identified two distinct student groups based on their satisfaction levels",
            "7. The regression model explains 18.4% of variance in overall quality of life"
        ]
        
        for finding in findings:
            st.write(finding)

    # SUMMARY
    with summ:
        st.header("Summary of Statistical Measures and Findings")
        
        findings = [
            "1. Internet satisfaction has the lowest mean (1.95/5) among satisfaction variables",
            "2. 63.64% of students report internet is frequently unstable",
            "3. There is a moderate positive correlation (r = 0.39) between internet satisfaction and overall quality of life",
            "4. Chi-square test confirms uneven distribution of expense categories, with food being the major expense (72.73%)",
            "5. Markov chain analysis predicts internet will be stable only 22.22% of the time in the long run",
            "6. Cluster analysis identified two distinct student groups based on their satisfaction levels",
            "7. The regression model explains 18.4% of variance in overall quality of life"
        ]
        
        for finding in findings:
            st.write(finding)

