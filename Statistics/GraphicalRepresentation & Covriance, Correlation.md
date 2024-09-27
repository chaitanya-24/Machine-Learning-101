## Graphical Representation

1. Histogram

A histogram is a graphical representation of the distribution of a dataset. It consists of bars where each bar represents a range of values (bin) and the height of the bar represents the frequency of data points falling within that range.

2. Types of Skewed Histograms

a) Symmetric (Normal Distribution):
   - Mean = Median = Mode
   - The distribution is balanced on both sides of the center.

b) Right Skewed (Positively Skewed):
   - Mode < Median < Mean
   - The tail of the distribution extends towards higher values.

c) Left Skewed (Negatively Skewed):
   - Mean < Median < Mode
   - The tail of the distribution extends towards lower values.

Dealing with skewed data:
- Apply transformations (e.g., log, square root) to make the distribution more symmetric.
- Use robust statistics (e.g., median instead of mean).
- Consider using non-parametric statistical methods.

3. Types of Histograms based on Number of Modes

a) Unimodal: One peak (most common)
b) Bimodal: Two peaks
c) Multimodal: Multiple peaks
d) Uniform: No clear peak, all bars approximately equal height

4. Boxplot

A boxplot (box-and-whisker plot) provides a summary of the distribution of a dataset:
- The box represents the interquartile range (IQR) from Q1 to Q3.
- The line inside the box is the median.
- Whiskers extend to the minimum and maximum values within 1.5 * IQR.
- Points beyond the whiskers are considered outliers.

5. Scatter Plot

A scatter plot shows the relationship between two continuous variables. Each point represents an observation with its x and y coordinates corresponding to the values of the two variables.

6. Covariance and Correlation

Covariance measures how two variables change together but is scale-dependent.

Equation: Cov(X,Y) = Σ((X - μX)(Y - μY)) / (n - 1)

Correlation is a standardized measure of the strength and direction of the linear relationship between two variables.

Equation: Corr(X,Y) = Cov(X,Y) / (σX * σY)

Where:
- μX and μY are the means of X and Y
- σX and σY are the standard deviations of X and Y

7. Pearson's Correlation Coefficient

Pearson's correlation coefficient is the most common measure of correlation:
- Ranges from -1 to 1
- -1 indicates a perfect negative linear relationship
- 0 indicates no linear relationship
- 1 indicates a perfect positive linear relationship

Now, let's implement these concepts in Python:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Generate sample data
np.random.seed(42)
x = np.random.normal(0, 1, 1000)
y = 2*x + np.random.normal(0, 1, 1000)

# 1. Histogram
plt.figure(figsize=(10, 6))
plt.hist(x, bins=30, edgecolor='black')
plt.title('Histogram of X')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# 2. Skewed Histograms
right_skewed = np.random.exponential(2, 1000)
left_skewed = 10 - np.random.exponential(2, 1000)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.hist(x, bins=30, edgecolor='black')
ax1.set_title('Symmetric')
ax2.hist(right_skewed, bins=30, edgecolor='black')
ax2.set_title('Right Skewed')
ax3.hist(left_skewed, bins=30, edgecolor='black')
ax3.set_title('Left Skewed')
plt.tight_layout()
plt.show()

# 3. Histograms with different modes
unimodal = np.random.normal(0, 1, 1000)
bimodal = np.concatenate([np.random.normal(-2, 0.5, 500), np.random.normal(2, 0.5, 500)])
multimodal = np.concatenate([np.random.normal(-4, 0.5, 300), np.random.normal(0, 0.5, 400), np.random.normal(4, 0.5, 300)])
uniform = np.random.uniform(-4, 4, 1000)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
ax1.hist(unimodal, bins=30, edgecolor='black')
ax1.set_title('Unimodal')
ax2.hist(bimodal, bins=30, edgecolor='black')
ax2.set_title('Bimodal')
ax3.hist(multimodal, bins=30, edgecolor='black')
ax3.set_title('Multimodal')
ax4.hist(uniform, bins=30, edgecolor='black')
ax4.set_title('Uniform')
plt.tight_layout()
plt.show()

# 4. Boxplot
plt.figure(figsize=(10, 6))
plt.boxplot([x, y])
plt.title('Boxplot of X and Y')
plt.xticks([1, 2], ['X', 'Y'])
plt.show()

# 5. Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5)
plt.title('Scatter Plot of Y vs X')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# 6. Covariance and Correlation
covariance = np.cov(x, y)[0, 1]
correlation = np.corrcoef(x, y)[0, 1]

print(f"Covariance between X and Y: {covariance}")
print(f"Correlation between X and Y: {correlation}")

# 7. Pearson's Correlation Coefficient
pearson_corr, _ = stats.pearsonr(x, y)
print(f"Pearson's Correlation Coefficient: {pearson_corr}")

# Correlation Heatmap
data = pd.DataFrame({'X': x, 'Y': y})
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap')
plt.show()

```

This code demonstrates how to create various statistical visualizations and calculate correlation measures using Python libraries like NumPy, Pandas, Matplotlib, Seaborn, and SciPy. Let's break down some key points:

1. Histogram: We use `plt.hist()` to create histograms. The `bins` parameter controls the number of bars.

2. Skewed Histograms: We generate right-skewed data using an exponential distribution and left-skewed data by subtracting exponential data from a constant.

3. Different Modes: We create various distributions (unimodal, bimodal, multimodal, and uniform) using different combinations of normal and uniform distributions.

4. Boxplot: `plt.boxplot()` is used to create boxplots. It clearly shows the median, quartiles, and potential outliers.

5. Scatter Plot: `plt.scatter()` is used to create a scatter plot, showing the relationship between X and Y.

6. Covariance and Correlation: We use `np.cov()` and `np.corrcoef()` to calculate these measures.

7. Pearson's Correlation Coefficient: We use `stats.pearsonr()` from SciPy to calculate this specific correlation measure.

Additionally, we create a correlation heatmap using Seaborn's `sns.heatmap()` function, which provides a visual representation of the correlation matrix.

When dealing with skewed data:
- For right-skewed data, you might apply a log transformation: `np.log(right_skewed)`
- For left-skewed data, you might apply a power transformation: `np.power(left_skewed, 2)`

These transformations can help make the data more symmetric, which is often desirable for many statistical analyses.
