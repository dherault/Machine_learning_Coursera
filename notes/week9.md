# Week 9

## Anomaly Detection

### Density Estimation

#### Problem Motivation

Is x_test anomalous? (Given many ok examples)

The model is p(x) a probability of anomaly. We check that p(x_test) >= epsilon (means ok).

Applications:
- Fraud Detection
- Manufacturing (find defects)
- Monitoring computers in a data center

#### Gaussian distribution

Also called normal distribution.
If x is a distributed Gaussian with mean mu and variance sigma².
i.e. x~N(mu, sigma²).

In a Gaussian distribution:
`p(x; mu, sigma²) = (1 / (sqrt(2Pi) * sigma)) * exp(-(x - mu)² / (2 * sigma²))`
(formula for the bell-shaped curve)

Parameter estimation: we have examples of x. We suspect that they are distributed according to a Gaussian distribution.
How to estimate mu and sigma²?
```
mu = (1 / m) * sum(i=1..m, x-i)
sigma² = (1 / m) * sum(i=1..m, (x-i - mu)²)
```
This is actualy the maximum likelyhood estimation of those 2 parameters.
In ML, we tend to use (1 / m), not (1 / (m - 1)).

#### Algorithm

```
x € R^n
x = x_1 ... x_n
x1~N(mu_1, sigma²_1)
x2~N(mu_2, sigma²_2)
p(x) = p(x_1: mu_1, sigma²_1) * p(x_2; ...) * ... * p(x_n; ...)
p(x) = product(j=1..n, p(x_j; mu_j, sigma²_j))
```

Depends on an independence assumption on x_1, x_2, etc... But works fine if they are dependent too.

Algo:
```
1. Choose features x_j that you think might be indicative of anomalous examples.
2. Fit parameters mu_1, ..., mu_n, sigma²_1, ..., sigma²_n
mu_j = (1 / m) * sum(i=1..m, x_j-i)
sigma²_j = (1 / m) * sum(i=1..m, (x_j-i - mu_j)²)
3. Given new examples x, compute p(x):
p(x) = product(j=1..n, p(x_j; mu_j, sigma²_j)) = product(j=1..n, (1 / (sqrt(2Pi) * sigma)) * exp(-(x_j - mu_j)² / (2 * sigma²_j)))
4. Anomaly if p(x) < epsilon
```

### Building an Anomaly Detection System

#### Developing and Evaluating an Anomaly Detection System

The importance of real number evaluation.
When developing a learning algorithm (choosing features, etc...) making decisions is much easier if we have a way of evaluating our learning Algorithm
--> Have some labeled data (normal/anomalous) to evaluate! On the test set
- Traininng set: 6000 good engines (no anomaly)
- CV set: 2000 good, 10 bad
- Test set: 2000 good, 10 bad

Don't use the same data in the CV and test set!

Algo evaluation:
- Fit model `p(x)` on the training set
- On the CV/Test set, predict  `y = 1 if p(x) < e` (anomality), `y = 0 if p(x) >= e` (normal). Possible evaluation metrics:
  - True Positive, false positive, false negative, true negative
  - Precision/Recall
  - F1-Score
- Can also use CV to choose parameter e (the one that maximises the F1-score)

Using the classification accuracy as an evaluation metric is not a good way of measuring the algo's performance because of skewed classes (much more 0s than 1s). An algo that always predict 0 will have a high accuracy

#### Anomaly Detection vs. Supervised Learning

Why don't we directly use a supervised learning algo to predict 0 or 1 ?

- Anomaly detection:
  - very small number of positive examples (~ 0 - 20 - 50 examples)
  - Many different types of anomaly, future ones may not look like anything before
- Supervised learning:
  - Large number of positive and negative examples
  - Enough positive examples to get a sense of what future positive examples will look like

#### Choosing What Features to Use

Plot the data to make sure it's Gaussian. If it's not Gaussian the algo usualy works fine too.
If not gaussian, use transformations on the data (like a log or ^a (a < 1) transformation)

Error analysis:
- Want p(x) large for normal examples x
- Want p(x) small for anomalous examples x

Most common problems:
- p(x) is comparable (say, both large) for normal and anomalous examples --> Find a new feature!

Combine features like x1² / x2
