# Week 7

## Large Margin Classification

### Optimization Objective

In logistic regression: `h(x) = 1 / (1 + e^(-theta' * x))`.
If y = 1, we want h(x) ~= 1, ie theta' * x >> 0.
If y = 0, we want h(x) ~= 0, ie theta' * x << 0.

- Logistic regrassion's cost function: `min (1 / m) * (sum(i=1..m, yi * -log(h(xi)) + (1 - yi) * (-log(1 - h(xi)))) + (lambda / 2) * sum(j=1..n, theta_j²))`
- Support Vector Machine: `min (1 / m) * (sum(i=1..m, yi * cost_1(theta' * xi) + (1 - yi) * cost_0(theta' * xi)) + (lambda / 2) * sum(j=1..n, theta_j²))`
- Support Vector Machine, conventionnaly: `min (1 / lambda) * sum(i=1..m, yi * cost_1(theta' * xi) + (1 - yi) * cost_0(theta' * xi)) + 2 * sum(j=1..n, theta_j²)`

```
cost_1: \_ (at 1)
cost_0: _/ (at -1)
```

SVM do not output a probability (like logistic regression), just a prediction (0 or 1).

### Large Margin Intuition

SVM are also called "large margin classifiers".

They try to separate the data using large margins.

### Mathematics Behind Large Margin Classification

Think projections :) https://www.coursera.org/learn/machine-learning/lecture/3eNnh/mathematics-behind-large-margin-classification




