# A Complete Beginner's Guide to Inferential Statistics with Python

## Introduction to Statistical Analysis

Statistics is broadly divided into two categories:
1. **Descriptive Statistics**: Summarizes and describes data
2. **Inferential Statistics**: Makes predictions and draws conclusions about a population based on a sample

This guide focuses on inferential statistics, which helps us make educated guesses about large groups (populations) by studying smaller groups (samples).

## 1. Population & Sample

### Understanding Population
A **population** is the complete set of all items/individuals that we want to study. For example:
- All students in a university
- All customers of a company
- All products manufactured by a factory

However, studying an entire population is often:
- Too expensive
- Too time-consuming
- Sometimes impossible (like studying all potential customers)

### Understanding Sample
A **sample** is a subset of the population that we actually study. Think of it like taste-testing a spoonful of soup instead of the whole pot.

Let's demonstrate this concept with Python:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Create a population (e.g., heights of all people in a city)
population = np.random.normal(loc=170, scale=10, size=10000)

# Take different sample sizes
small_sample = np.random.choice(population, size=30)
medium_sample = np.random.choice(population, size=100)
large_sample = np.random.choice(population, size=500)

# Visualize the differences
plt.figure(figsize=(15, 5))

# Population distribution
plt.subplot(141)
plt.hist(population, bins=50, alpha=0.7, color='blue')
plt.title('Population\n(n=10000)')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')

# Small sample
plt.subplot(142)
plt.hist(small_sample, bins=15, alpha=0.7, color='red')
plt.title('Small Sample\n(n=30)')
plt.xlabel('Height (cm)')

# Medium sample
plt.subplot(143)
plt.hist(medium_sample, bins=20, alpha=0.7, color='green')
plt.title('Medium Sample\n(n=100)')
plt.xlabel('Height (cm)')

# Large sample
plt.subplot(144)
plt.hist(large_sample, bins=30, alpha=0.7, color='purple')
plt.title('Large Sample\n(n=500)')
plt.xlabel('Height (cm)')

plt.tight_layout()

# Compare statistics
def print_statistics(data, name):
    print(f"\n{name} Statistics:")
    print(f"Mean: {np.mean(data):.2f}")
    print(f"Median: {np.median(data):.2f}")
    print(f"Standard Deviation: {np.std(data):.2f}")
    print(f"Minimum: {np.min(data):.2f}")
    print(f"Maximum: {np.max(data):.2f}")

print_statistics(population, "Population")
print_statistics(small_sample, "Small Sample")
print_statistics(medium_sample, "Medium Sample")
print_statistics(large_sample, "Large Sample")
```

### Key Points About Sampling:
1. **Representativeness**: A good sample should represent the population well
2. **Random Sampling**: Each member of the population should have an equal chance of being selected
3. **Sample Size**: Larger samples generally provide more accurate estimates
4. **Sampling Error**: The difference between sample statistics and population parameters

## 2. Point Estimation & Interval Estimation

### Point Estimation
Point estimation is like trying to guess the exact value of a population parameter using sample data.

#### Common Point Estimators:

1. **Sample Mean (μ̂)**: Estimates population mean
   ```python
   def sample_mean_example():
       # Generate population
       population = np.random.normal(loc=100, scale=15, size=10000)
       
       # Take sample
       sample = np.random.choice(population, size=100)
       
       # Calculate point estimate
       point_estimate = np.mean(sample)
       
       # Compare with true population parameter
       true_value = np.mean(population)
       
       print(f"True Population Mean: {true_value:.2f}")
       print(f"Sample Point Estimate: {point_estimate:.2f}")
       print(f"Estimation Error: {abs(true_value - point_estimate):.2f}")
   
   sample_mean_example()
   ```

2. **Sample Proportion (p̂)**: Estimates population proportion
   ```python
   def sample_proportion_example():
       # Generate binary population (e.g., success/failure)
       population = np.random.binomial(n=1, p=0.7, size=10000)
       
       # Take sample
       sample = np.random.choice(population, size=100)
       
       # Calculate point estimates
       true_prop = np.mean(population)
       sample_prop = np.mean(sample)
       
       print(f"True Population Proportion: {true_prop:.2f}")
       print(f"Sample Proportion Estimate: {sample_prop:.2f}")
       print(f"Estimation Error: {abs(true_prop - sample_prop):.2f}")
   
   sample_proportion_example()
   ```

3. **Sample Variance (σ̂²)**: Estimates population variance
   ```python
   def sample_variance_example():
       # Generate population
       population = np.random.normal(loc=100, scale=15, size=10000)
       
       # Take sample
       sample = np.random.choice(population, size=100)
       
       # Calculate point estimates
       true_var = np.var(population)
       sample_var = np.var(sample, ddof=1)  # ddof=1 for sample variance
       
       print(f"True Population Variance: {true_var:.2f}")
       print(f"Sample Variance Estimate: {sample_var:.2f}")
       print(f"Estimation Error: {abs(true_var - sample_var):.2f}")
   
   sample_variance_example()
   ```

### Interval Estimation
Instead of guessing a single value, interval estimation provides a range where we believe the true population parameter lies.

```python
def confidence_interval_example():
    # Generate sample data
    sample = np.random.normal(loc=100, scale=15, size=30)
    
    # Calculate confidence intervals
    confidence_levels = [0.90, 0.95, 0.99]
    
    for conf_level in confidence_levels:
        ci = stats.t.interval(confidence=conf_level,
                            df=len(sample)-1,
                            loc=np.mean(sample),
                            scale=stats.sem(sample))
        
        print(f"\n{conf_level*100}% Confidence Interval:")
        print(f"Lower Bound: {ci[0]:.2f}")
        print(f"Upper Bound: {ci[1]:.2f}")
        print(f"Interval Width: {ci[1]-ci[0]:.2f}")

confidence_interval_example()
```

## 3. Confidence Intervals

A confidence interval tells us the range where we can reasonably expect the true population parameter to fall.

### Understanding Confidence Intervals

The interpretation of a 95% confidence interval is often misunderstood:
- ✓ Correct: "If we repeated this sampling process many times, about 95% of the calculated intervals would contain the true population parameter."
- ✗ Incorrect: "There's a 95% chance that the true parameter is in this specific interval."

```python
def demonstrate_confidence_intervals():
    # True population parameter
    true_mean = 100
    
    # Generate multiple samples and calculate CIs
    n_simulations = 100
    sample_size = 30
    confidence_level = 0.95
    
    # Store results
    contains_true_value = 0
    intervals = []
    
    for _ in range(n_simulations):
        # Generate sample
        sample = np.random.normal(loc=true_mean, scale=15, size=sample_size)
        
        # Calculate CI
        ci = stats.t.interval(confidence=confidence_level,
                            df=len(sample)-1,
                            loc=np.mean(sample),
                            scale=stats.sem(sample))
        
        # Check if CI contains true value
        if ci[0] <= true_mean <= ci[1]:
            contains_true_value += 1
        
        intervals.append(ci)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    y_positions = range(n_simulations)
    
    for i, (lower, upper) in enumerate(intervals):
        color = 'green' if lower <= true_mean <= upper else 'red'
        plt.plot([lower, upper], [i, i], color=color, alpha=0.5)
    
    plt.axvline(x=true_mean, color='blue', linestyle='--', label='True Mean')
    plt.ylabel('Simulation Number')
    plt.xlabel('Value')
    plt.title(f'95% Confidence Intervals from {n_simulations} Simulations\n'
              f'Percentage containing true value: {contains_true_value/n_simulations*100:.1f}%')
    plt.legend()
    
demonstrate_confidence_intervals()
```

### Factors Affecting Confidence Interval Width:
1. **Sample Size**: Larger samples = narrower intervals
2. **Confidence Level**: Higher confidence = wider intervals
3. **Population Variability**: More variable populations = wider intervals

```python
def compare_ci_factors():
    # Compare sample sizes
    sample_sizes = [10, 30, 100, 300]
    population = np.random.normal(loc=100, scale=15, size=10000)
    
    plt.figure(figsize=(12, 6))
    
    for i, n in enumerate(sample_sizes):
        samples = [np.random.choice(population, size=n) for _ in range(100)]
        cis = [stats.t.interval(0.95, df=len(s)-1, loc=np.mean(s), scale=stats.sem(s))
               for s in samples]
        
        widths = [upper - lower for (lower, upper) in cis]
        
        plt.subplot(2, 2, i+1)
        plt.hist(widths, bins=20)
        plt.title(f'Sample Size = {n}\nMean Width: {np.mean(widths):.2f}')
        plt.xlabel('Confidence Interval Width')
    
    plt.tight_layout()

compare_ci_factors()
```

## 4. Student's T-Distribution

### Understanding the T-Distribution
The t-distribution is similar to the normal distribution but has heavier tails, making it more conservative for small samples.

Key characteristics:
1. Bell-shaped and symmetric
2. Shape depends on degrees of freedom (df = n-1)
3. Approaches normal distribution as df increases

```python
def visualize_t_distribution():
    x = np.linspace(-4, 4, 1000)
    
    # Plot different t-distributions
    plt.figure(figsize=(12, 6))
    
    # Add normal distribution
    plt.plot(x, stats.norm.pdf(x), 'k-', lw=2, label='Normal')
    
    # Add t-distributions with different df
    dfs = [1, 5, 10, 30]
    colors = ['red', 'blue', 'green', 'purple']
    
    for df, color in zip(dfs, colors):
        plt.plot(x, stats.t.pdf(x, df), color=color, 
                label=f't (df={df})')
    
    plt.title('Comparison of T-distributions with Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)

visualize_t_distribution()
```

### When to Use T-Distribution:
1. Small sample size (n < 30)
2. Population standard deviation unknown
3. Data approximately normally distributed

```python
def demonstrate_t_vs_z():
    # Generate sample
    sample_size = 20
    population_mean = 100
    population_std = 15
    
    sample = np.random.normal(loc=population_mean, 
                             scale=population_std, 
                             size=sample_size)
    
    # Calculate confidence intervals using both methods
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    
    # Z-interval (if we knew population std)
    z_ci = stats.norm.interval(0.95,
                              loc=sample_mean,
                              scale=population_std/np.sqrt(sample_size))
    
    # T-interval (when we don't know population std)
    t_ci = stats.t.interval(0.95,
                           df=sample_size-1,
                           loc=sample_mean,
                           scale=sample_std/np.sqrt(sample_size))
    
    print("Comparing Z and T intervals:")
    print(f"Z-interval: ({z_ci[0]:.2f}, {z_ci[1]:.2f})")
    print(f"T-interval: ({t_ci[0]:.2f}, {t_ci[1]:.2f})")
    print(f"Z-interval width: {z_ci[1]-z_ci[0]:.2f}")
    print(f"T-interval width: {t_ci[1]-t_ci[0]:.2f}")

demonstrate_t_vs_z()
```

## 5. Hypothesis Testing

### 5.1 Understanding Hypothesis Testing

Hypothesis testing is like a scientific trial where we:
1. Start with a default position (null hypothesis)
2. Collect evidence (data)
3. Decide if there's enough evidence to reject the default position

#### Null Hypothesis (H₀):
- The "status quo" or "no effect" hypothesis
- Usually contains "=" in the statement
- Example: "The mean height is equal to 170 cm"

#### Alternative Hypothesis (H₁ or Hₐ):
- The claim we're trying to support
- Usually contains "≠", "<", or ">" in the statement
- Example: "The mean height is different from 170 cm"

```python
def hypothesis_testing_example():
    # Example: Testing if a new teaching method improves test scores
    
    # Null Hypothesis (H₀): μ = 70 (old average)
    # Alternative Hypothesis (H₁): μ > 70 (new average is higher)
    
    # Generate sample data (new test scores)
    np.random.seed(42)  # for reproducibility
    sample_size = 25
    sample = np.random.normal(loc=75, scale=10, size=sample_size)
    
    # Perform one-sample t-test
    t_stat, p_value = stats.ttest_1samp(sample, popmean=70)
    
    # For one-tailed test, divide p-value by 2 (if t-stat is in the expected direction)
    p_value_one_tailed = p_value / 2 if t_stat > 0 else 1 - p_value / 2
    
    print("Hypothesis Test Results:")
    print(f"Sample Mean: {np.mean(sample):.2f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value (two-tailed): {p_value:.4f}")
    print(f"p-value (one-tailed): {p_value_one_tailed:.4f}")
    
    # Visualize the results
    plt.figure(figsize=(12, 8))
    
    # Create x values for t-distribution
    x = np.linspace(-4, 4, 1000)
    
    # Get degrees of freedom
    df = sample_size - 1
    
    # Plot t-distribution
    t_dist = stats.t.pdf(x, df)
    plt.plot(x, t_dist, 'b-', label='t-distribution')
    
    # Calculate critical values for two-tailed test (alpha = 0.05)
    critical_value = stats.t.ppf(0.975, df)  # 97.5th percentile for two-tailed test
    
    # Shade rejection regions for two-tailed test
    x_left = x[x <= -critical_value]
    x_right = x[x >= critical_value]
    plt.fill_between(x_left, stats.t.pdf(x_left, df), color='red', alpha=0.3, label='Rejection Region (α=0.05)')
    plt.fill_between(x_right, stats.t.pdf(x_right, df), color='red', alpha=0.3)
    
    # Add vertical line for observed t-statistic
    plt.axvline(x=t_stat, color='green', linestyle='--', label=f'Observed t-stat: {t_stat:.2f}')
    
    # Add vertical lines for critical values
    plt.axvline(x=-critical_value, color='black', linestyle=':', label=f'Critical Values: ±{critical_value:.2f}')
    plt.axvline(x=critical_value, color='black', linestyle=':')
    
    # Add labels and title
    plt.title("Hypothesis Test Visualization\n" + 
              f"H₀: μ = 70 vs H₁: μ > 70 (n={sample_size})")
    plt.xlabel('t-statistic')
    plt.ylabel('Density')
    
    # Add a legend
    plt.legend()
    
    # Add annotations for key information
    text_info = (f"Sample Mean: {np.mean(sample):.2f}\n"
                 f"Sample Size: {sample_size}\n"
                 f"Degrees of Freedom: {df}\n"
                 f"p-value (two-tailed): {p_value:.4f}\n"
                 f"p-value (one-tailed): {p_value_one_tailed:.4f}")
    
    plt.text(0.98, 0.98, text_info, 
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Show plot
    plt.show()
    
    # Add interpretation
    alpha = 0.05
    print("\nTest Interpretation:")
    print("-" * 50)
    if p_value_one_tailed < alpha:
        print(f"At α = {alpha}, we reject the null hypothesis.")
        print("There is sufficient evidence to conclude that the new teaching method")
        print("significantly improves test scores above 70.")
    else:
        print(f"At α = {alpha}, we fail to reject the null hypothesis.")
        print("There is insufficient evidence to conclude that the new teaching method")
        print("significantly improves test scores above 70.")
    
    # Effect size calculation (Cohen's d)
    cohens_d = (np.mean(sample) - 70) / np.std(sample, ddof=1)
    print(f"\nEffect Size (Cohen's d): {cohens_d:.4f}")
    
    # Interpret effect size
    if abs(cohens_d) < 0.2:
        effect = "negligible"
    elif abs(cohens_d) < 0.5:
        effect = "small"
    elif abs(cohens_d) < 0.8:
        effect = "medium"
    else:
        effect = "large"
    
    print(f"Effect size interpretation: {effect}")

# Run the example
hypothesis_testing_example()
```


### 5.2 P-value and Statistical Significance

The p-value answers the question: "If the null hypothesis is true, what's the probability of getting results as extreme as what we observed?"

```python
def demonstrate_p_value():
    # Set up the visualization
    np.random.seed(42)
    
    # Generate null distribution
    null_distribution = np.random.normal(loc=0, scale=1, size=10000)
    
    # Calculate observed test statistic (example)
    observed_value = 2.5
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot null distribution
    plt.hist(null_distribution, bins=50, density=True, alpha=0.7,
             label='Null Distribution')
    
    # Add normal curve
    x = np.linspace(-4, 4, 100)
    plt.plot(x, stats.norm.pdf(x), 'r-', lw=2, label='Normal Distribution')
    
    # Shade p-value region
    x_fill = np.linspace(observed_value, 4, 100)
    plt.fill_between(x_fill, stats.norm.pdf(x_fill), color='red', alpha=0.3,
                    label='p-value region')
    
    # Add observed value line
    plt.axvline(x=observed_value, color='green', linestyle='--',
                label='Observed Value')
    
    plt.title('Visualizing P-value')
    plt.xlabel('Test Statistic')
    plt.ylabel('Density')
    plt.legend()
    
    # Calculate actual p-value
    p_value = 1 - stats.norm.cdf(observed_value)
    print(f"P-value: {p_value:.4f}")

demonstrate_p_value()
```

### 5.3 Types of Errors in Hypothesis Testing

#### Type I Error (α - Alpha Error)
- Rejecting H₀ when it's actually true
- "False Positive"
- Probability = significance level (α)

#### Type II Error (β - Beta Error)
- Failing to reject H₀ when it's actually false
- "False Negative"
- Probability = β
- Power = 1 - β

```python
def demonstrate_type_errors():
    # Create a visualization of Type I and Type II errors
    
    # Parameters
    mu0 = 0  # null hypothesis mean
    mu1 = 2  # alternative hypothesis mean
    sigma = 1  # standard deviation
    alpha = 0.05  # significance level
    
    # Create x values for plotting
    x = np.linspace(-4, 6, 1000)
    
    # Calculate critical value
    critical_value = stats.norm.ppf(1 - alpha)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot null distribution
    plt.plot(x, stats.norm.pdf(x, mu0, sigma), 'b-', 
            label='Null Distribution (H₀)')
    
    # Plot alternative distribution
    plt.plot(x, stats.norm.pdf(x, mu1, sigma), 'r-', 
            label='Alternative Distribution (H₁)')
    
    # Shade Type I error
    x_type1 = np.linspace(critical_value, 4, 100)
    plt.fill_between(x_type1, stats.norm.pdf(x_type1, mu0, sigma),
                    color='blue', alpha=0.3, label='Type I Error')
    
    # Shade Type II error
    x_type2 = np.linspace(-4, critical_value, 100)
    plt.fill_between(x_type2, stats.norm.pdf(x_type2, mu1, sigma),
                    color='red', alpha=0.3, label='Type II Error')
    
    plt.axvline(x=critical_value, color='black', linestyle='--',
                label='Critical Value')
    
    plt.title('Type I and Type II Errors')
    plt.xlabel('Test Statistic')
    plt.ylabel('Density')
    plt.legend()

demonstrate_type_errors()
```

### 5.4 Types of Statistical Tests

#### One-Tailed vs Two-Tailed Tests

```python
def demonstrate_tail_types():
    # Create visualization of one-tailed and two-tailed tests
    x = np.linspace(-4, 4, 1000)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Two-tailed test
    ax1.plot(x, stats.norm.pdf(x), 'b-')
    ax1.fill_between(x[x <= -1.96], stats.norm.pdf(x[x <= -1.96]), 
                    color='red', alpha=0.3)
    ax1.fill_between(x[x >= 1.96], stats.norm.pdf(x[x >= 1.96]), 
                    color='red', alpha=0.3)
    ax1.set_title('Two-Tailed Test (α = 0.05)')
    
    # One-tailed test
    ax2.plot(x, stats.norm.pdf(x), 'b-')
    ax2.fill_between(x[x >= 1.645], stats.norm.pdf(x[x >= 1.645]), 
                    color='red', alpha=0.3)
    ax2.set_title('One-Tailed Test (α = 0.05)')
    
    plt.tight_layout()

demonstrate_tail_types()
```

### 5.5 Common Statistical Tests

#### 1. Z-Test (One Sample)
Used when:
- Population standard deviation is known
- Sample size is large (n ≥ 30)
- Data is normally distributed

```python
def one_sample_z_test_example():
    # Generate sample data
    np.random.seed(42)
    population_mean = 100
    population_std = 15
    sample_size = 50
    
    # Generate sample
    sample = np.random.normal(loc=105, scale=population_std, size=sample_size)
    
    # Calculate z-statistic
    sample_mean = np.mean(sample)
    z_stat = (sample_mean - population_mean) / (population_std / np.sqrt(sample_size))
    
    # Calculate p-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    print("One-Sample Z-Test Results:")
    print(f"Sample Mean: {sample_mean:.2f}")
    print(f"Z-statistic: {z_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    x = np.linspace(-4, 4, 1000)
    plt.plot(x, stats.norm.pdf(x), 'b-', label='Standard Normal')
    plt.axvline(x=z_stat, color='red', linestyle='--', label='Observed Z-stat')
    plt.title('Z-Test Visualization')
    plt.legend()

one_sample_z_test_example()
```

#### 2. T-Test

##### Independent T-Test (Two Sample)
Used when:
- Comparing means of two independent groups
- Population standard deviation unknown
- Data approximately normally distributed

```python
def independent_t_test_example():
    # Generate sample data for two groups
    np.random.seed(42)
    
    # Group 1: Control group
    group1 = np.random.normal(loc=100, scale=15, size=30)
    
    # Group 2: Treatment group
    group2 = np.random.normal(loc=110, scale=15, size=30)
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    # Calculate effect size (Cohen's d)
    cohens_d = (np.mean(group2) - np.mean(group1)) / np.sqrt(
        ((np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2))
    
    print("\nIndependent T-Test Results:")
    print(f"Group 1 Mean: {np.mean(group1):.2f}")
    print(f"Group 2 Mean: {np.mean(group2):.2f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Cohen's d: {cohens_d:.4f}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.boxplot([group1, group2], labels=['Control', 'Treatment'])
    plt.title('Comparing Two Independent Groups')
    plt.ylabel('Values')

independent_t_test_example()
```

##### Paired T-Test
Used when:
- Comparing two related measurements
- Same subjects measured twice
- Data approximately normally distributed

```python
def paired_t_test_example():
    # Generate paired data
    np.random.seed(42)
    n = 30
    
    # Before treatment
    before = np.random.normal(loc=100, scale=15, size=n)
    
    # After treatment (correlated with before, with some improvement)
    improvement = np.random.normal(loc=10, scale=5, size=n)
    after = before + improvement
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(before, after)
    
    print("\nPaired T-Test Results:")
    print(f"Mean Difference: {np.mean(after - before):.2f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    # Before-After plot
    plt.subplot(121)
    plt.plot([1, 2], [before, after], 'b-', alpha=0.3)
    plt.plot([1, 2], [np.mean(before), np.mean(after)], 'r-', linewidth=2)
    plt.xticks([1, 2], ['Before', 'After'])
    plt.title('Before-After Measurements')
    
    # Difference histogram
    plt.subplot(122)
    plt.hist(after - before, bins=15)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Histogram of Differences')
    plt.xlabel('Difference (After - Before)')
    
    plt.tight_layout()

paired_t_test_example()
```

### 5.6 Choosing the Right Statistical Test

Here's a decision flowchart to help choose the appropriate test:

```python
def create_test_selection_diagram():
    mermaid_code = '''
    graph TD
        A[Start] --> B{How many groups?}
        B -->|One group| C{Known population SD?}
        B -->|Two groups| D{Independent or Paired?}
        C -->|Yes| E[Z-test]
        C -->|No| F[One-sample t-test]
        D -->|Independent| G[Independent t-test]
        D -->|Paired| H[Paired t-test]
    '''
    return mermaid_code

# Note: This would be rendered as a Mermaid diagram in a proper environment
```

### Summary of Key Points

1. **Hypothesis Testing Process**:
   - State hypotheses (H₀ and H₁)
   - Choose significance level (α)
   - Collect data and calculate test statistic
   - Calculate p-value
   - Make decision and interpret results

2. **Common Mistakes to Avoid**:
   - Using the wrong test for your data
   - Ignoring assumptions of the test
   - Misinterpreting p-values
   - Confusing statistical significance with practical significance

3. **Best Practices**:
   - Always check test assumptions
   - Report effect sizes along with p-values
   - Consider practical significance
   - Use appropriate visualizations to support conclusions