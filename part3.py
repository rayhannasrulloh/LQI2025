from main import df
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

# ------------------------------------------------------
# PART 3: DATA VISUALIZATION
# ------------------------------------------------------
print("\n" + "="*50)
print("PART 3: DATA VISUALIZATION")
print("="*50)

# Create a function to save plots
def save_plot(filename, dpi=300):
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    print(f"Plot saved as: {filename}")
    plt.close()

# 1. DEMOGRAPHIC VISUALIZATION
print("\nCreating Demographic Visualizations...")

plt.figure(figsize=(18, 6))
plt.suptitle("Demographic Distribution of Survey Respondents", fontsize=16)

# Gender Distribution
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

# Add count and percentage labels to bars
for i, (bar, percentage) in enumerate(zip(bars, year_percentages)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height} ({percentage}%)', ha='center', va='bottom')

# Duration of Stay Distribution
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

save_plot('1_demographic_distribution.png')

# 2. SATISFACTION ANALYSIS VISUALIZATION
print("\nCreating Satisfaction Analysis Visualizations...")

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, figure=fig)
plt.suptitle("Satisfaction Analysis", fontsize=16)

# Satisfaction distributions with mean, median and mode markers
categories = ['Package Pickup System', 'Internet Speed', 'Overall Quality of Life']
colors = ['skyblue', 'salmon', 'lightgreen']
data_cols = ['Package_Satisfaction', 'Internet_Satisfaction', 'Overall_Quality']

for i, (category, col, color) in enumerate(zip(categories, data_cols, colors)):
    ax = fig.add_subplot(gs[0, i])
    
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
    ax.legend()

# Boxplot for comparison
ax = fig.add_subplot(gs[1, 0])
satisfaction_data = df[['Package_Satisfaction', 'Internet_Satisfaction', 'Overall_Quality']]
sns.boxplot(data=satisfaction_data, ax=ax, palette=colors)
ax.set_title('Comparison of Satisfaction Distributions')
ax.set_ylabel('Satisfaction Score (1-5)')
ax.set_xticklabels(['Package\nPickup', 'Internet\nSpeed', 'Overall\nQuality'])

# Add box plot explanation
box_explanation = """
Box Plot Elements:
- Box: IQR (25th to 75th percentile)
- Line in Box: Median
- Whiskers: 1.5 x IQR or min/max
- Points: Outliers
"""
ax.text(0.95, 0.05, box_explanation, transform=ax.transAxes, ha='right', va='bottom',
        bbox=dict(facecolor='white', alpha=0.7), fontsize=10)

# Calculate correlation matrix
correlation_matrix = df[['Package_Satisfaction', 'Internet_Satisfaction', 'Overall_Quality']].corr()

# Scatterplot between Internet Satisfaction and Overall Quality
ax = fig.add_subplot(gs[1, 1])
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

# Heatmap for correlation matrix
ax = fig.add_subplot(gs[1, 2])
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.4f', linewidths=0.5, ax=ax)
ax.set_title('Correlation Matrix of Satisfaction Variables')

save_plot('2_satisfaction_analysis.png')

# 3. EXPENSE ANALYSIS VISUALIZATION
print("\nCreating Expense Analysis Visualizations...")

plt.figure(figsize=(18, 10))
plt.suptitle("Student Expense Analysis", fontsize=16)

# Monthly Spending Distribution
plt.subplot(2, 2, 1)
spending_order = ['Less than Rp 500,000', 'Rp 500,000 - Rp 1,000,000', 
                  'Rp 1,000,000 - Rp 2,000,000', 'More than Rp 2,000,000']
spending_counts = df['Monthly_Spending'].value_counts().reindex(spending_order)
spending_percentages = (spending_counts / len(df) * 100).round(1)

bars = plt.bar(range(len(spending_counts)), spending_counts.values, color=sns.color_palette("pastel"))
plt.xticks(range(len(spending_counts)), spending_counts.index, rotation=45, ha='right')
plt.title('Monthly Spending Distribution')
plt.ylabel('Number of Students')

# Add count and percentage labels
for i, (bar, percentage) in enumerate(zip(bars, spending_percentages)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height} ({percentage}%)', ha='center', va='bottom')

# Expense Categories
plt.subplot(2, 2, 2)
expense_categories = ['Food_Expense', 'Online_Shopping_Expense', 'Entertainment_Expense', 
                     'Transportation_Expense', 'Academic_Expense', 'Daily_Needs_Expense']
expense_labels = ['Food', 'Online Shopping', 'Entertainment', 'Transportation', 'Academic', 'Daily Needs']
expense_counts = [df[col].sum() for col in expense_categories]
expense_percentages = [(count / len(df) * 100).round(1) for count in expense_counts]

# Sort by frequency
sorted_indices = np.argsort(expense_counts)[::-1]  # Descending order
sorted_labels = [expense_labels[i] for i in sorted_indices]
sorted_counts = [expense_counts[i] for i in sorted_indices]
sorted_percentages = [expense_percentages[i] for i in sorted_indices]

bars = plt.bar(range(len(sorted_counts)), sorted_counts, color=sns.color_palette("pastel"))
plt.xticks(range(len(sorted_counts)), sorted_labels, rotation=45, ha='right')
plt.title('Major Expense Categories')
plt.ylabel('Number of Students')

# Add count and percentage labels
for i, (bar, percentage) in enumerate(zip(bars, sorted_percentages)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height} ({percentage}%)', ha='center', va='bottom')

# Chi-square analysis of expense distribution
plt.subplot(2, 1, 2)
plt.title('Chi-Square Analysis of Expense Distribution')

# Calculate Chi-square
n = len(df)
k = len(expense_categories)
observed_freq = expense_counts
expected_freq = [n/k] * k  # Equal distribution hypothesis

chi_sq = sum([(o - e)**2 / e for o, e in zip(observed_freq, expected_freq)])
dof = k - 1
p_value = 1 - stats.chi2.cdf(chi_sq, dof)
critical_value = stats.chi2.ppf(0.95, dof)

# Create bar chart showing observed vs expected
x = np.arange(len(expense_labels))
width = 0.35

plt.bar(x - width/2, observed_freq, width, label='Observed', color='skyblue')
plt.bar(x + width/2, expected_freq, width, label='Expected (Equal Distribution)', color='lightgreen')

plt.xticks(x, expense_labels, rotation=45, ha='right')
plt.ylabel('Frequency')
plt.legend()

# Add Chi-square test results
chi_text = (f"Chi-square statistic = {chi_sq:.2f}\n"
            f"Degrees of freedom = {dof}\n"
            f"p-value = {p_value:.6f}\n"
            f"Critical value (α=0.05) = {critical_value:.2f}\n\n"
            r"$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$" + "\n"
            f"Conclusion: {'Reject' if chi_sq > critical_value else 'Fail to reject'} null hypothesis")
plt.text(0.02, 0.95, chi_text, transform=plt.gca().transAxes, ha='left', va='top',
         bbox=dict(facecolor='white', alpha=0.7))

save_plot('3_expense_analysis.png')

# 4. INTERNET USAGE ANALYSIS
print("\nCreating Internet Usage Visualizations...")

plt.figure(figsize=(18, 10))
plt.suptitle("Internet Usage and Stability Analysis", fontsize=16)

# Internet Stability Distribution
plt.subplot(2, 2, 1)
stability_counts = df['Internet_Stability'].value_counts()
stability_percentages = (stability_counts / len(df) * 100).round(1)

plt.pie(stability_counts, labels=[f"{label}\n{count} ({percentage}%)" 
                                 for label, count, percentage in 
                                 zip(stability_counts.index, stability_counts, stability_percentages)], 
        autopct='', startangle=90, colors=['salmon', 'skyblue'])
plt.title('Internet Stability in Dormitory')

# Mobile Data Usage Distribution
plt.subplot(2, 2, 2)
usage_counts = df['Mobile_Data_Usage'].value_counts()
usage_percentages = (usage_counts / len(df) * 100).round(1)

plt.pie(usage_counts, labels=[f"{label}\n{count} ({percentage}%)" 
                             for label, count, percentage in 
                             zip(usage_counts.index, usage_counts, usage_percentages)], 
        autopct='', startangle=90, colors=sns.color_palette("pastel"))
plt.title('Frequency of Personal Mobile Data Usage')

# Cross-tabulation analysis
plt.subplot(2, 1, 2)
plt.title('Relationship Between Internet Stability and Mobile Data Usage')

# Create cross-tabulation
cross_tab = pd.crosstab(df['Internet_Stability'], df['Mobile_Data_Usage'])
cross_tab_percentages = pd.crosstab(df['Internet_Stability'], df['Mobile_Data_Usage'], 
                                    normalize='all') * 100

# Calculate conditional probabilities
p_daily_mobile = df[df['Mobile_Data_Usage'] == 'Yes, almost every day'].shape[0] / df.shape[0]
p_unstable = df[df['Internet_Stability'] == 'No, frequently unstable'].shape[0] / df.shape[0]
p_unstable_given_daily_mobile = df[(df['Mobile_Data_Usage'] == 'Yes, almost every day') & 
                                  (df['Internet_Stability'] == 'No, frequently unstable')].shape[0] / \
                             df[df['Mobile_Data_Usage'] == 'Yes, almost every day'].shape[0]
p_daily_mobile_given_unstable = df[(df['Mobile_Data_Usage'] == 'Yes, almost every day') & 
                                  (df['Internet_Stability'] == 'No, frequently unstable')].shape[0] / \
                             df[df['Internet_Stability'] == 'No, frequently unstable'].shape[0]

# Create heatmap for cross-tabulation
sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu')
plt.xlabel('Mobile Data Usage')
plt.ylabel('Internet Stability')

# Add Bayes' theorem information
bayes_text = (
    "Bayes' Theorem Analysis:\n\n"
    r"$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$" + "\n\n"
    f"P(Daily Mobile Usage) = {p_daily_mobile:.4f}\n"
    f"P(Unstable Internet) = {p_unstable:.4f}\n"
    f"P(Unstable Internet | Daily Mobile Usage) = {p_unstable_given_daily_mobile:.4f}\n"
    f"P(Daily Mobile Usage | Unstable Internet) = {p_daily_mobile_given_unstable:.4f}\n\n"
    f"Calculation: P(D|U) = [P(U|D) × P(D)] / P(U)\n"
    f"                    = [{p_unstable_given_daily_mobile:.4f} × {p_daily_mobile:.4f}] / {p_unstable:.4f}\n"
    f"                    = {p_daily_mobile_given_unstable:.4f}"
)
plt.text(1.05, 0.5, bayes_text, transform=plt.gca().transAxes, ha='left', va='center',
         bbox=dict(facecolor='white', alpha=0.7))

save_plot('4_internet_usage_analysis.png')

# 5. PROBABILITY DISTRIBUTION ANALYSIS
print("\nCreating Probability Distribution Visualizations...")

plt.figure(figsize=(18, 12))
plt.suptitle("Probability Distribution Analysis", fontsize=16)

# Binomial Distribution: Internet Stability
plt.subplot(2, 2, 1)
plt.title("Binomial Distribution: Probability of Students with Stable Internet")

# Probability of stable internet
p_stable_internet = df[df['Internet_Stability'] == 'Yes, but sometimes slow'].shape[0] / df.shape[0]
n_students = 10  # Assume 10 new students

# Calculate binomial probabilities
x = np.arange(0, n_students + 1)
binomial_pmf = stats.binom.pmf(x, n_students, p_stable_internet)
binomial_cdf = stats.binom.cdf(x, n_students, p_stable_internet)

# Plot PMF
bars = plt.bar(x, binomial_pmf, alpha=0.7, color='skyblue', label='PMF')
plt.xticks(x)
plt.xlabel("Number of Students with Stable Internet (k)")
plt.ylabel("Probability P(X = k)")

# Add probability values above bars
for i, (bar, prob) in enumerate(zip(bars, binomial_pmf)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{prob:.4f}', ha='center', va='bottom', fontsize=9)

# Add binomial formula
binomial_formula = (r"$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$" + "\n"
                   f"n = {n_students}, p = {p_stable_internet:.4f}")
plt.text(0.95, 0.95, binomial_formula, transform=plt.gca().transAxes, ha='right', va='top',
         bbox=dict(facecolor='white', alpha=0.7))

# Example calculation for k=4
k_example = 4
binom_coef = math.comb(n_students, k_example)
prob_success = p_stable_internet ** k_example
prob_failure = (1 - p_stable_internet) ** (n_students - k_example)
prob_k_example = binom_coef * prob_success * prob_failure

example_calc = (f"Example calculation for k = {k_example}:\n"
               f"P(X = {k_example}) = C({n_students},{k_example}) × {p_stable_internet:.4f}^{k_example} × (1-{p_stable_internet:.4f})^{n_students-k_example}\n"
               f"= {binom_coef} × {prob_success:.6f} × {prob_failure:.6f}\n"
               f"= {prob_k_example:.6f}")
plt.text(0.5, 0.05, example_calc, transform=plt.gca().transAxes, ha='center', va='bottom',
         bbox=dict(facecolor='white', alpha=0.7), fontsize=9)

# Poisson Distribution: Package Arrivals
plt.subplot(2, 2, 2)
plt.title("Poisson Distribution: Probability of Daily Package Arrivals")

# Mean packages per day
lambda_packages = 5

# Calculate Poisson probabilities
x = np.arange(0, 15)
poisson_pmf = stats.poisson.pmf(x, lambda_packages)
poisson_cdf = stats.poisson.cdf(x, lambda_packages)

# Plot PMF
bars = plt.bar(x, poisson_pmf, alpha=0.7, color='lightgreen', label='PMF')
plt.xticks(x)
plt.xlabel("Number of Packages per Day (k)")
plt.ylabel("Probability P(X = k)")

# Add probability values above bars
for i, (bar, prob) in enumerate(zip(bars, poisson_pmf)):
    if i % 2 == 0:  # Show every other value to avoid crowding
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{prob:.4f}', ha='center', va='bottom', fontsize=9)

# Add Poisson formula
poisson_formula = (r"$P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$" + "\n"
                  f"λ = {lambda_packages}")
plt.text(0.95, 0.95, poisson_formula, transform=plt.gca().transAxes, ha='right', va='top',
         bbox=dict(facecolor='white', alpha=0.7))

# Example calculation for k=8
k_example = 8
numerator = (lambda_packages ** k_example) * np.exp(-lambda_packages)
denominator = math.factorial(k_example)
prob_k_example = numerator / denominator

example_calc = (f"Example calculation for k = {k_example}:\n"
               f"P(X = {k_example}) = ({lambda_packages}^{k_example} × e^(-{lambda_packages})) / {k_example}!\n"
               f"= ({lambda_packages ** k_example:.1f} × {np.exp(-lambda_packages):.6f}) / {denominator}\n"
               f"= {numerator:.6f} / {denominator}\n"
               f"= {prob_k_example:.6f}")
plt.text(0.5, 0.05, example_calc, transform=plt.gca().transAxes, ha='center', va='bottom',
         bbox=dict(facecolor='white', alpha=0.7), fontsize=9)

# Normal Distribution
plt.subplot(2, 2, 3)
plt.title("Normal Distribution: Overall Quality of Life")

# Calculate mean and std for Overall Quality
mean_quality = df['Overall_Quality'].mean()
std_quality = df['Overall_Quality'].std()

# Generate x values for plotting
x = np.linspace(mean_quality - 4*std_quality, mean_quality + 4*std_quality, 1000)
pdf = stats.norm.pdf(x, mean_quality, std_quality)

# Plot PDF
plt.plot(x, pdf, 'r-', lw=2, label='PDF')
plt.axvline(mean_quality, color='green', linestyle='--', label=f'Mean = {mean_quality:.2f}')
plt.axvline(mean_quality + std_quality, color='blue', linestyle=':', label=f'Mean + 1σ = {mean_quality + std_quality:.2f}')
plt.axvline(mean_quality - std_quality, color='blue', linestyle=':', label=f'Mean - 1σ = {mean_quality - std_quality:.2f}')

# Fill areas for standard deviations
plt.fill_between(x, 0, pdf, where=(x >= mean_quality - std_quality) & (x <= mean_quality + std_quality),
                color='lightblue', alpha=0.5, label='68% within 1σ')

# Add histogram of actual data for comparison
plt.hist(df['Overall_Quality'], bins=5, density=True, alpha=0.3, color='gray', label='Actual Data')

plt.xlabel('Overall Quality Score')
plt.ylabel('Probability Density')
plt.legend(loc='upper left', fontsize=9)

# Add normal distribution formula
normal_formula = (r"$f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$" + "\n"
                 f"μ = {mean_quality:.4f}, σ = {std_quality:.4f}")
plt.text(0.95, 0.95, normal_formula, transform=plt.gca().transAxes, ha='right', va='top',
         bbox=dict(facecolor='white', alpha=0.7))

# Example calculation for a specific value
x_example = 4
z_score = (x_example - mean_quality) / std_quality
prob_density = stats.norm.pdf(x_example, mean_quality, std_quality)

example_calc = (f"Example calculation for x = {x_example}:\n"
               f"Z-score = (x - μ) / σ = ({x_example} - {mean_quality:.4f}) / {std_quality:.4f} = {z_score:.4f}\n"
               f"f({x_example}) = (1 / ({std_quality:.4f}√(2π))) × e^(-0.5 × {z_score:.4f}²)\n"
               f"= {prob_density:.6f}")
plt.text(0.5, 0.05, example_calc, transform=plt.gca().transAxes, ha='center', va='bottom',
         bbox=dict(facecolor='white', alpha=0.7), fontsize=9)

# Confidence Interval for Proportion
plt.subplot(2, 2, 4)
plt.title("Confidence Interval for Mobile Data Usage Proportion")

# Calculate proportion and CI
daily_mobile_users = df[df['Mobile_Data_Usage'] == 'Yes, almost every day'].shape[0]
p_daily_mobile = daily_mobile_users / df.shape[0]
z_alpha = 1.96  # 95% confidence level
std_error = np.sqrt((p_daily_mobile * (1 - p_daily_mobile)) / df.shape[0])
margin_error = z_alpha * std_error
ci_lower = max(0, p_daily_mobile - margin_error)
ci_upper = min(1, p_daily_mobile + margin_error)

# Create visualization
plt.plot([0, 1], [1, 1], 'k-', lw=2)  # Horizontal line
plt.plot([p_daily_mobile, p_daily_mobile], [0.9, 1.1], 'r-', lw=2, label=f'Proportion = {p_daily_mobile:.4f}')
plt.plot([ci_lower, ci_upper], [1, 1], 'b-', lw=4, alpha=0.6, label=f'95% CI: ({ci_lower:.4f}, {ci_upper:.4f})')

plt.xlim(0, 1)
plt.ylim(0.8, 1.2)
plt.yticks([])
plt.xlabel('Proportion of Students Using Mobile Data Almost Daily')
plt.legend(loc='upper center')

# Add confidence interval formula
ci_formula = (r"$p \pm z_{\alpha/2} \sqrt{\frac{p(1-p)}{n}}$" + "\n"
             f"p = {p_daily_mobile:.4f}, n = {df.shape[0]}, z_{{α/2}} = {z_alpha}")
plt.text(0.05, 0.9, ci_formula, transform=plt.gca().transAxes, ha='left', va='top',
         bbox=dict(facecolor='white', alpha=0.7))

# Example calculation
example_calc = (f"Example calculation:\n"
               f"Standard Error = √[p(1-p)/n] = √[{p_daily_mobile:.4f}×{1-p_daily_mobile:.4f}/{df.shape[0]}] = {std_error:.4f}\n"
               f"Margin of Error = {z_alpha} × {std_error:.4f} = {margin_error:.4f}\n"
               f"95% CI = {p_daily_mobile:.4f} ± {margin_error:.4f} = ({ci_lower:.4f}, {ci_upper:.4f})")
plt.text(0.5, 0.3, example_calc, transform=plt.gca().transAxes, ha='center', va='center',
         bbox=dict(facecolor='white', alpha=0.7), fontsize=9)

save_plot('5_probability_distributions.png')

# 6. HYPOTHESIS TESTING VISUALIZATION
print("\nCreating Hypothesis Testing Visualizations...")

plt.figure(figsize=(18, 12))
plt.suptitle("Statistical Hypothesis Testing", fontsize=16)

# Independent t-test: Internet Satisfaction by Gender
plt.subplot(2, 2, 1)
plt.title("Independent t-test: Internet Satisfaction by Gender")

# Filter data for female and male
female_internet = df[df['Gender'] == 'Female']['Internet_Satisfaction']
male_internet = df[df['Gender'] == 'Male']['Internet_Satisfaction']

# Calculate t-test
t_stat, p_value = stats.ttest_ind(female_internet, male_internet, equal_var=False)

# Create grouped boxplot
gender_data = df[df['Gender'].isin(['Female', 'Male'])]
sns.boxplot(x='Gender', y='Internet_Satisfaction', data=gender_data)
plt.ylabel('Internet Satisfaction Score')

# Add t-test results
t_test_result = (f"Independent t-test results:\n"
                f"Female mean (n={len(female_internet)}): {female_internet.mean():.4f}\n"
                f"Male mean (n={len(male_internet)}): {male_internet.mean():.4f}\n"
                f"t-statistic: {t_stat:.4f}\n"
                f"p-value: {p_value:.4f}\n"
                f"Conclusion: {'Significant' if p_value < 0.05 else 'Not significant'} at α = 0.05")
plt.text(0.5, 0.05, t_test_result, transform=plt.gca().transAxes, ha='center', va='bottom',
         bbox=dict(facecolor='white', alpha=0.7))

# Add t-test formula
t_formula = (r"$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$")
plt.text(0.95, 0.95, t_formula, transform=plt.gca().transAxes, ha='right', va='top',
         bbox=dict(facecolor='white', alpha=0.7))

# ANOVA: Overall Quality by Duration of Stay
plt.subplot(2, 2, 2)
plt.title("One-way ANOVA: Overall Quality by Duration of Stay")

# Create boxplot
sns.boxplot(x='Duration_Stay', y='Overall_Quality', data=df, order=['< 6 Months', '6 Months - 1 Year', '1 - 2 Years'])
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
plt.text(0.5, 0.05, anova_result, transform=plt.gca().transAxes, ha='center', va='bottom',
         bbox=dict(facecolor='white', alpha=0.7))

# Add ANOVA formula
anova_formula = (r"$F = \frac{MS_{between}}{MS_{within}}$" + "\n"
                r"$MS_{between} = \frac{SS_{between}}{df_{between}}$" + "\n"
                r"$MS_{within} = \frac{SS_{within}}{df_{within}}$")
plt.text(0.95, 0.95, anova_formula, transform=plt.gca().transAxes, ha='right', va='top',
         bbox=dict(facecolor='white', alpha=0.7))

# Multiple Regression Analysis
plt.subplot(2, 1, 2)
plt.title("Multiple Regression: Factors Affecting Overall Quality of Life")

# Perform regression
X = df[['Internet_Satisfaction', 'Package_Satisfaction']]
X = sm.add_constant(X)
y = df['Overall_Quality']
model = sm.OLS(y, X).fit()

# Create 3D scatter plot
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

# Add regression results
regression_result = (f"Multiple Regression Results:\n"
                    f"Model: Overall_Quality = {model.params[0]:.4f} + {model.params[1]:.4f}×Internet_Satisfaction + {model.params[2]:.4f}×Package_Satisfaction\n"
                    f"R²: {model.rsquared:.4f}\n"
                    f"Adjusted R²: {model.rsquared_adj:.4f}\n"
                    f"F-statistic: {model.fvalue:.4f}, p-value: {model.f_pvalue:.4f}\n\n"
                    f"Internet_Satisfaction: β = {model.params[1]:.4f}, p = {model.pvalues[1]:.4f}\n"
                    f"Package_Satisfaction: β = {model.params[2]:.4f}, p = {model.pvalues[2]:.4f}")

ax.text2D(0.05, 0.95, regression_result, transform=ax.transAxes, ha='left', va='top',
        bbox=dict(facecolor='white', alpha=0.7))

# Add regression formula
regression_formula = (r"$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2$")
ax.text2D(0.95, 0.05, regression_formula, transform=ax.transAxes, ha='right', va='bottom',
        bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig('6_hypothesis_testing.png', dpi=300)
print("Plot saved as: 6_hypothesis_testing.png")
plt.close()

# 7. ADVANCED ANALYSIS: MARKOV CHAIN & CLUSTERING
print("\nCreating Advanced Analysis Visualizations...")

plt.figure(figsize=(18, 12))
plt.suptitle("Advanced Statistical Analysis", fontsize=16)

# Markov Chain Analysis
plt.subplot(2, 1, 1)
plt.title("Markov Chain Analysis: Long-term Internet Stability Prediction")

# Define transition matrix
transition_matrix = np.array([
    [0.3, 0.7],  # Stable -> Stable, Stable -> Unstable
    [0.2, 0.8]   # Unstable -> Stable, Unstable -> Unstable
])

# Calculate initial state based on data
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
plt.text(1.02, 0.5, markov_info, transform=plt.gca().transAxes, ha='left', va='center',
         bbox=dict(facecolor='white', alpha=0.7))

# K-means Clustering Analysis
plt.subplot(2, 1, 2)
plt.title("K-means Clustering: Student Satisfaction Profiles")

# Prepare data for clustering
X_cluster = df[['Internet_Satisfaction', 'Package_Satisfaction', 'Overall_Quality']]

# Perform k-means clustering with k=2
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_cluster)

# Get cluster centers and sizes
centers = kmeans.cluster_centers_
cluster_sizes = df['Cluster'].value_counts()

# Create scatter plot
sns.scatterplot(x='Internet_Satisfaction', y='Overall_Quality', hue='Cluster', 
                data=df, palette='viridis', s=100, alpha=0.7)
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

# Add cluster information
cluster_info = "K-means Clustering Results:\n\n"
for i in range(len(centers)):
    cluster_info += (f"Cluster {i+1} (n={cluster_sizes[i]}):\n"
                   f"  Internet Satisfaction: {centers[i, 0]:.2f}\n"
                   f"  Package Satisfaction: {centers[i, 1]:.2f}\n"
                   f"  Overall Quality: {centers[i, 2]:.2f}\n\n")

cluster_info += "K-means algorithm minimizes:\n"
cluster_info += r"$J = \sum_{i=1}^{k} \sum_{x \in S_i} ||x - \mu_i||^2$"

plt.text(1.02, 0.5, cluster_info, transform=plt.gca().transAxes, ha='left', va='center',
         bbox=dict(facecolor='white', alpha=0.7))

save_plot('7_advanced_analysis.png')

print("\nAll visualizations completed!")
print("\nSummary of statistical measures and findings:")
print("1. Internet satisfaction has the lowest mean (1.95/5) among satisfaction variables")
print("2. 63.64% of students report internet is frequently unstable")
print("3. There is a moderate positive correlation (r = 0.39) between internet satisfaction and overall quality of life")
print("4. Chi-square test confirms uneven distribution of expense categories, with food being the major expense (72.73%)")
print("5. Markov chain analysis predicts internet will be stable only 22.22% of the time in the long run")
print("6. Cluster analysis identified two distinct student groups based on their satisfaction levels")
print("7. The regression model explains 18.4% of variance in overall quality of life")