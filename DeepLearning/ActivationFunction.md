# Activation Functions in Deep Learning

Activation functions are mathematical functions used in neural networks to introduce non-linearity into the model. This non-linearity allows the neural network to learn complex patterns in the data. Without activation functions, the neural network would essentially be performing linear transformations, regardless of the number of layers. Different activation functions have distinct characteristics, advantages, and disadvantages.

### 1. **Sigmoid Activation Function**

- **Formula:** \( \sigma(x) = \frac{1}{1 + e^{-x}} \)
- **Range:** (0, 1)

**Pros:**
- Smooth gradient, preventing abrupt changes in gradients, which helps in the gradient-based optimization.
- Output values bound between 0 and 1, useful for models where output is a probability.

**Cons:**
- **Vanishing Gradient Problem:** Gradients become very small during backpropagation, making training deep networks difficult.
- **Output Not Zero-Centered:** Can slow down the convergence of gradient descent.
- Computationally expensive due to the exponential function.

### 2. **Tanh (Hyperbolic Tangent) Activation Function**

- **Formula:** \( \text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \)
- **Range:** (-1, 1)

**Pros:**
- Smooth gradient.
- Zero-centered output, which can lead to faster convergence compared to the sigmoid function.

**Cons:**
- **Vanishing Gradient Problem:** Like the sigmoid function, it can suffer from the vanishing gradient issue.
- Computationally expensive due to the exponential functions.

### 3. **ReLU (Rectified Linear Unit) Activation Function**

- **Formula:** \( f(x) = \max(0, x) \)
- **Range:** [0, ∞)

**Pros:**
- Simple and computationally efficient.
- Helps to alleviate the vanishing gradient problem.
- Introduces sparsity in the network, making the model more efficient.

**Cons:**
- **Dying ReLU Problem:** Neurons can sometimes get stuck during training and always output zero, effectively "dying."
- Can lead to exploding gradients if not handled properly.

### 4. **Leaky ReLU Activation Function**

- **Formula:** \( f(x) = \begin{cases} 
x & \text{if } x \ge 0 \\
\alpha x & \text{if } x < 0
\end{cases} \)
- **Range:** (-∞, ∞)

**Pros:**
- Addresses the dying ReLU problem by allowing a small, non-zero gradient when the unit is not active (negative part).

**Cons:**
- The value of \(\alpha\) (typically a small number like 0.01) needs to be predefined or learned, adding complexity.

### 5. **Parametric ReLU (PReLU) Activation Function**

- **Formula:** \( f(x) = \begin{cases} 
x & \text{if } x \ge 0 \\
\alpha x & \text{if } x < 0
\end{cases} \)
- **Range:** (-∞, ∞)

**Pros:**
- Similar to Leaky ReLU but \(\alpha\) is learned during training, which can lead to better performance.
- Helps in mitigating the dying ReLU problem.

**Cons:**
- Increased complexity due to the need to learn additional parameters (\(\alpha\)).

### 6. **ELU (Exponential Linear Unit) Activation Function**

- **Formula:** \( f(x) = \begin{cases} 
x & \text{if } x \ge 0 \\
\alpha (e^x - 1) & \text{if } x < 0
\end{cases} \)
- **Range:** (-\(\alpha\), ∞)

**Pros:**
- Smooth gradient.
- Helps mitigate the vanishing gradient problem.
- Zero-centered outputs.

**Cons:**
- Computationally more expensive due to the exponential function.
- Can lead to exploding gradients if not properly managed.

### 7. **Swish Activation Function**

- **Formula:** \( f(x) = x \cdot \sigma(x) \) where \(\sigma(x)\) is the sigmoid function.
- **Range:** (-∞, ∞)

**Pros:**
- Smooth and non-monotonic, providing better gradient flow.
- Often leads to better performance in deeper networks compared to ReLU.

**Cons:**
- Computationally more expensive due to the combination of multiplication and sigmoid function.

### 8. **Softmax Activation Function**

- **Formula:** \( \sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}} \)
- **Range:** (0, 1), with the outputs summing to 1.

**Pros:**
- Converts raw scores into probabilities, useful for multi-class classification problems.
- Ensures that the sum of the output probabilities is 1.

**Cons:**
- Computationally expensive due to the exponentiation and normalization.
- Susceptible to the vanishing gradient problem in deep networks.

### Summary Table

| **Activation Function** | **Pros** | **Cons** |
|-------------------------|----------|----------|
| **Sigmoid** | - Smooth gradient <br> - Output between 0 and 1 | - Vanishing gradient problem <br> - Output not zero-centered <br> - Computationally expensive |
| **Tanh** | - Smooth gradient <br> - Zero-centered output | - Vanishing gradient problem <br> - Computationally expensive |
| **ReLU** | - Simple and efficient <br> - Helps alleviate vanishing gradient | - Dying ReLU problem <br> - Can lead to exploding gradients |
| **Leaky ReLU** | - Mitigates dying ReLU problem | - Predefined or learned \(\alpha\) |
| **PReLU** | - Mitigates dying ReLU problem <br> - \(\alpha\) is learned | - Increased complexity |
| **ELU** | - Smooth gradient <br> - Zero-centered output | - Computationally expensive <br> - Can lead to exploding gradients |
| **Swish** | - Smooth and non-monotonic <br> - Better gradient flow | - Computationally expensive |
| **Softmax** | - Converts scores to probabilities <br> - Sum of probabilities is 1 | - Computationally expensive <br> - Susceptible to vanishing gradient |

### Conclusion

Choosing the right activation function is crucial for the performance of a neural network. The decision depends on the specific problem, the architecture of the network, and empirical performance. Each activation function has its own strengths and weaknesses, and understanding these helps in selecting the most appropriate one for your neural network model.