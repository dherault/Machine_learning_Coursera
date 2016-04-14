# Week 6

## Evaluating a Learning Algorithm

### Deciding What to Try Next

- Get more training examples
- Try smaller set of features (to prevent overfitting)
- Try getting additionnal features
- Try adding polynomial features (x1², x2², x1x2, ...)
- Try decreasing lambda
- Try increasing lambda

Don't do it by going with your guts! THINK!!!
--> Machine Learning diagnostic (! can take time, but are worth it).

### Evaluating a Hypothesis

How to evaluate an hypothesis (h) that have been learned by our algo ?
Might be overfitting :(

--> Split our examples data in two: a training set and a test set (70% / 30%). 
We can then train with our training set and test our hypothesis with the test set.
How ? 
- Learn theta with the training set, 
- then compute J_test with the test set
- then Compute the misclassification error 
example: 0/1 misclassification error:
```
err(h(x), y) = { 1 if (h(x) >= 0.5 and y = 0) or if (h(x) < 0.5 and y = 1) } 
               { 0 otherwise }
Test_error = (1 / m_test) * sum(i=1..m_test, err(h(x_test-i), y_test-i))
```
### Model Selection and Train/Validation/Test Sets

The training set error is not a good indication of how good an hypothesis is. 
For example, you could overfit and have a very low error on the training set, but a high one on new examples.

How to select a model ?

`h(x) = theta_0 + theta_1 * x + theta_2 * x² + ... + theta_d * x ^ d`
How to pick d, the degree of our polynomial function?

- 1 - `h(x) = theta_0 + theta_1 * x ` --> theta-1 --> J_test(theta-1)
- ... - ... 
- d - `h(x) = theta_0 + theta_1 * x + theta_2 * x² + ... + theta_d * x ^ d` --> theta-d --> J_test(theta-d)

If we choose d based on J_test results, it is likely to be an optimistic estimate of generalization error.
d could be fit to the test set, not the training set, but then te test set would be a 'configuration of d' set, not a 'test' set.

Cross validation (CV) set. For example: 60% training set, 20% cross validation set, 20% test set.
So we can have J_training, J_cv, J_test

- 1 - `h(x) = theta_0 + theta_1 * x ` --> min J --> theta_1 --> J_cv(theta-1)
- ... - ... 
- d - `h(x) = theta_0 + theta_1 * x + theta_2 * x² + ... + theta_d * x ^ d` --> min J --> theta_d --> J_cv(theta-d)

allow us to pick d by minimizing J_cv, so the test set can be independant of our choosing of d and remain a 'test' set.

## Bias vs. Variance

### Diagnosing Bias vs. Variance

If a learning algo does not do as well as hoped, it is (almost always) because of a high bias (underfitting) or a high variance (overfitting);

You can plot `y: error (J_train and J_cv), x: d` to see how to choose d.
If d is small and J_cv is high (compared to J_train): High bias (undefitting)
If d is big and J_cv is high: High variance (overfitting)

### Regularization and Bias/Variance

How does regularization affects bias and variance ?

- Large lambda: high bias (underfitting, theta_i ~= 0)
- Small lambda: high variance (overfitting)


First, compute the 3 J (train, cv and test) without the lambda parameter (lambda = 0)

then try (choose the step wisely)
- lambda = 0 --> min J --> theta-1 (not really theta-1) --> J_cv(theta-1)
- lambda = 0.01  --> min J --> theta-2 --> J_cv(theta-2)
- lambda = 0.02 ...
- lambda = 0.04
- lambda = 0.08
- ...
- lambda = 10.24  --> min J --> theta-12 --> J_cv(theta-12)

Pick the lambda that gives the lowest J_cv, and compute J_test

### Learning Curves

A good thing to plot (to sanity check our algo, to diagnose overfitting/underfitting)

Plot errors:
- `J_train(theta) = (1 / 2m) * sum(i=1..m, (h_theta(x-i) - y-i)²)`
- `J_cv(theta) = (1 / 2m_cv) * sum(i=1..m, (h_theta(x_cv-i) - y_cv-i)²)`
as a function of m (not m_cv).

In high bias cases, the training error will be close to the cv error.
In high variance cases, the training error will be low (indeed), and there will be a large gap between the training error and the cv error.

In high variance cases, getting more training data is likely to help.

### Deciding What to Do Next Revisited

- Get more training examples --> fixes high variance
- Try smaller set of features --> fixes high variance
- Try getting additionnal features --> fixes high bias
- Try adding polynomial features (x1², x2², x1x2, ...) --> fixes high bias
- Try decreasing lambda --> fixes high bias
- Try increasing lambda --> fixes high variance

So,

Fixing high variance (overfitting):
- Get more training examples
- Try smaller set of features
- Try increasing lambda

Fixing high bias (underfitting):
- Try getting additionnal features
- Try adding polynomial features (x1², x2², x1x2, ...)
- Try decreasing lambda

Neural networks and overfitting:

- Small nn: fewer parameters, more prone to underfitting, computationally cheaper
- Large nn: more parameters, more prone to overfitting, computationally more expensive --> use regularization
- Use a different number of hidden layers, and see what works best on our CV set

## Building a Spam Classifier

### Prioritizing What to Work On

Feature x: choose 100 words indicative of spam/not spam (eg: discount, buy --> spam. David --> not spam).
In pratice, take the most frequently occuring n words (10,000 to 50,000).

Also: email header information! (obscure origin information, unusual servers).

You might need a sophisticated "misspelling" checking algo to detect hidden words. (H3ll0 vvorld).

So many options! Don't go for what your guts tells you. THINK!! --> Error analysis.

### Error Analysis

Recommended approach:
- Start with a simple algo that you can implement quickly. Implement it and test it on your cross-validation data.
- Plot learning curves to decide if more data, more features, etc.. are likely to help.
- Error analysis: Manually examine the examples (in cross validation set) that your algorithm made errors on. See if you can spot any systematic tred in what type of examples it is making errors on.

NLP: stemming software (Porter stemmer)

## Handling Skewed Data

### Error Metrics for Skewed Classes

Precision / Recall.
y = 1 in presence of rare class that we want to detect.
```
                        Actual class
             |        1       |        0       |
          ---|----------------|----------------|
Predicted  1 | True positive  | False positive |
  class   ---|----------------|----------------|
           0 | False negative | True negative  |
          ---|----------------|----------------|
```
Compute 2 numbers:
- Precision: of all y = 1 predictions, what fraction is actually 1 ?
- Precision = **(number of)** true positives / predicted positives = true positives / (true positives + false positives)
- Recall: of all actual y = 1, what fraction did we predict 1 ?
- Recall = true positives / actual positives = true positives / (true positives / false negatives)

If a classifier has a recall = 0 all the time, it is not a good one. (?)

- Accuracy = (true positives + true negatives) / (total examples)
- Precision = (true positives) / (true positives + false positives)
- Recall = (true positives) / (true positives + false negatives)
- F1 score = (2 * precision * recall) / (precision + recall)

### Trading Off Precision and Recall

Higher precision leads to lower recall.
Higher recall leads to lower precision.

More generally, predict 1 if h(x) >= threshold.
High threshold --> high precision, low recal.
Low threshold --> high recall, low precision.

Compute 
- average = (P + R) / 2 (useless)
- F1 score = 2 * P * R / (P + R) (useful, compute it on the CV set and maximize it to get the threshold!!!)

F1 should be used on skewed dataset instead of accuracy. A dataset is skewed when too many outputs are the same (too many 0s or too many 1s)

## Using Large Data Sets

### Data For Machine Learning

Using a lot a data to train our larning algo is useful under certain conditions.

"It's not who has the best algorithm that wins, it's who has the most data".

Useful test: given the (unique) input x, can a human expert donfidenly predict y?

Use a learning algo with many parameters (many features, nn with many hidden units) --> low bias algorithms.
Use a very large training set (unlikely to overfit) --> low variance

- given the (unique) input x, can a human expert donfidenly predict y?
- can we get a large training set and train a large number of parameters?
- 