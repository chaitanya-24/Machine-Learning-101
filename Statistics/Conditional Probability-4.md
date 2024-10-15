## Conditional Probability

1. Conditional Probability

Conditional probability is the probability of an event A occurring, given that another event B has already occurred.

Formula: P(A|B) = P(A ∩ B) / P(B)

2. Bayes' Theorem

Bayes' Theorem provides a way to revise predictions or theories (update probabilities) given new or additional evidence.

Formula: P(A|B) = [P(B|A) * P(A)] / P(B)

Where:
- P(A|B) is the posterior probability
- P(B|A) is the likelihood
- P(A) is the prior probability
- P(B) is the marginal likelihood

Real-world use cases:
1. Medical diagnosis
2. Spam filtering
3. Criminal investigations
4. Machine learning algorithms (e.g., Naive Bayes classifier)

Let's implement Bayes' Theorem with a medical diagnosis example:



```python
def bayes_theorem(prior, likelihood, marginal):
    return (likelihood * prior) / marginal

# Medical diagnosis example
# Probability of having the disease (prior)
P_disease = 0.01

# Probability of positive test result given the disease (likelihood)
P_positive_given_disease = 0.95

# Probability of positive test result given no disease (false positive rate)
P_positive_given_no_disease = 0.05

# Calculate marginal likelihood
P_positive = P_positive_given_disease * P_disease + P_positive_given_no_disease * (1 - P_disease)

# Calculate posterior probability
P_disease_given_positive = bayes_theorem(P_disease, P_positive_given_disease, P_positive)

print(f"Probability of having the disease given a positive test result: {P_disease_given_positive:.4f}")

```

3. Probability Distributions

A probability distribution describes the probability of each possible outcome in a random experiment.

4. Bernoulli Distribution

The Bernoulli distribution models a single binary outcome (success/failure).

Parameters: p (probability of success)
Mean: p
Variance: p(1-p)

5. Binomial Distribution

The Binomial distribution models the number of successes in a fixed number of independent Bernoulli trials.

Parameters: n (number of trials), p (probability of success)
Mean: np
Variance: np(1-p)

6. Uniform Distribution

The Uniform distribution models a constant probability over a specified range.

Parameters: a (minimum), b (maximum)
Mean: (a + b) / 2
Variance: (b - a)^2 / 12

7. Normal Distribution

The Normal (or Gaussian) distribution is a symmetric, bell-shaped distribution defined by its mean and standard deviation.

Parameters: μ (mean), σ (standard deviation)
Mean: μ
Variance: σ^2

8. Standard Normal Distribution (z-score)

The Standard Normal Distribution is a Normal Distribution with μ = 0 and σ = 1.

Z-score formula: z = (x - μ) / σ

Let's implement these distributions and compare them:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Sample sizes
n_samples = 10000

# Bernoulli Distribution
p_bernoulli = 0.3
bernoulli_samples = np.random.binomial(n=1, p=p_bernoulli, size=n_samples)

# Binomial Distribution
n_binomial, p_binomial = 10, 0.3
binomial_samples = np.random.binomial(n=n_binomial, p=p_binomial, size=n_samples)

# Uniform Distribution
a, b = 0, 10
uniform_samples = np.random.uniform(a, b, n_samples)

# Normal Distribution
mu, sigma = 0, 1
normal_samples = np.random.normal(mu, sigma, n_samples)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Probability Distributions")

axs[0, 0].hist(bernoulli_samples, bins=[-0.5, 0.5, 1.5], rwidth=0.8)
axs[0, 0].set_title("Bernoulli Distribution")

axs[0, 1].hist(binomial_samples, bins=range(n_binomial+2), rwidth=0.8)
axs[0, 1].set_title("Binomial Distribution")

axs[1, 0].hist(uniform_samples, bins=20)
axs[1, 0].set_title("Uniform Distribution")

axs[1, 1].hist(normal_samples, bins=30)
axs[1, 1].set_title("Normal Distribution")

plt.tight_layout()
plt.show()

# Calculate and print z-scores
x = 2
z_score = (x - mu) / sigma
print(f"Z-score for x={x}: {z_score}")

# Standard Scaler vs Normalization
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = np.random.normal(loc=10, scale=5, size=(100, 1))

# Standard Scaler (Z-score normalization)
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)

# Min-Max Normalization
normalizer = MinMaxScaler()
normalized_data = normalizer.fit_transform(data)

print("\nOriginal data statistics:")
print(f"Mean: {np.mean(data):.2f}, Std: {np.std(data):.2f}")
print("\nStandardized data statistics:")
print(f"Mean: {np.mean(standardized_data):.2f}, Std: {np.std(standardized_data):.2f}")
print("\nNormalized data statistics:")
print(f"Mean: {np.mean(normalized_data):.2f}, Std: {np.std(normalized_data):.2f}")

```

This code demonstrates the various probability distributions and compares StandardScaler (Z-score normalization) with Min-Max normalization.

9. Standard Scaler vs Normalization

Standard Scaler (Z-score normalization):
- Transforms data to have mean=0 and standard deviation=1
- Formula: z = (x - μ) / σ
- Use when: The distribution is approximately normal, or when you want to preserve the shape of the distribution while centering and scaling it.

Min-Max Normalization:
- Scales data to a fixed range, typically [0, 1]
- Formula: x_norm = (x - x_min) / (x_max - x_min)
- Use when: You want to bound your values within a specific range, or when the distribution is not Gaussian or unknown.

When to use which:

1. Use Standard Scaler:
   - For algorithms assuming normally distributed data (e.g., linear regression, logistic regression)
   - When features have different units or scales
   - When outliers are few or you want to preserve their effects

2. Use Min-Max Normalization:
   - For neural networks or algorithms that need data on a fixed scale
   - When you want to preserve zero values in sparse data
   - When you want to retain the exact bounds of your data

Key points from the code:

1. The Bayes' Theorem example shows how to calculate the probability of having a disease given a positive test result, demonstrating the power of Bayes' Theorem in medical diagnosis.

2. The probability distributions example visualizes Bernoulli, Binomial, Uniform, and Normal distributions, helping to understand their shapes and characteristics.

3. The z-score calculation shows how to standardize a value from a normal distribution.

4. The comparison between Standard Scaler and Min-Max Normalization demonstrates how these techniques affect the mean and standard deviation of the data.

Remember, the choice between standardization and normalization depends on your specific data and the requirements of your machine learning algorithm. Always consider the nature of your data and the assumptions of your model when choosing a scaling method.

