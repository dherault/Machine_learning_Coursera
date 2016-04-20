# Week 7

## Large Margin Classification

### Optimization Objective

In logistic regression: `h(x) = 1 / (1 + e^(-theta' * x))`.
If y = 1, we want h(x) ~= 1, ie theta' * x >> 0.
If y = 0, we want h(x) ~= 0, ie theta' * x << 0.

- Logistic regrassion's cost function: `min (1 / m) * (sum(i=1..m, yi * -log(h(xi)) + (1 - yi) * (-log(1 - h(xi)))) + (lambda / 2) * sum(j=1..n, theta_j²))`
- Support Vector Machine: `min (1 / m) * (sum(i=1..m, yi * cost_1(theta' * xi) + (1 - yi) * cost_0(theta' * xi)) + (lambda / 2) * sum(j=1..n, theta_j²))`
- Support Vector Machine, conventionnaly: `min (1 / lambda) * sum(i=1..m, yi * cost_1(theta' * xi) + (1 - yi) * cost_0(theta' * xi)) + 0.5 * sum(j=1..n, theta_j²)`

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

## Kernels

Is there a better choice of features that high order polynomials? (x1², x2², x1x2, etc...)

A kernel is a similarity function (for now). Here is a gaussian kernel for a landmark l and an example x: `similarity(x, l) = k(x, l) = e^(-norm(x - l) / (2 * sigma²))`

Instead of using polynomial features, we're gonna use distances to landmarks! 
Eg: Predict 1 when `t0 + f1t1 + f2t2 + f3t3 >= 0`, with fi the distance to the landmark li.
So for points close to landmarks with positive t, we predict 1, else 0.

<!--How do we choose landmarks and what other similarity functions can we use? Stay tuned.-->

Choosing landmarks: Use the training examples as lamdmarks!

Training: `min C * sum(i=1..m, yi * cost_1(theta' * fi) + (1 - yi) * cost_0(theta' * fi)) + (1 / 2) * sum(j=1..n, theta_j²)` where fi is the distance between xi and li. 
Also m (number of training examples) is equal to n (number of features).
Also, `sum(j=1..n, theta_j²) === theta' * theta` where theta is the teta matrix without theta_0.
Also, most of the time you'll see `theta' * M * theta` where M is a matrix based on the kernel used, for efficiency and scalling to bigger sets.

About SVM parameters:

C (= 1 / lambda):
- Large C: Lower bias, high variance, overfitting (small lambda)
- Small C: Higher bias, low variance, underfitting (large lambda)

sigma²:
- Large sigma²: Features fi vary more smoothly. Higher bias, lower variance
- Small sigma²: Features fi vary less smoothly. Lower bias, higher variance

## SVMs in Practice

- Use libraries (to solve theta) like liblenear or libsvm
- You need to choose C and a kernel. No kernel is good too: predict y=1 if theta'*x >= 0. If n is large and m is small, this is a good idea. (edge factory?)
- If using a non linear kernel like the gaussian kernel, you'll need to choose it's parameters (sigma for the gaussian kernel)
- If n is small and m is large, a non linear kernel will do fine.
- Do perform feature scalling before using a gaussian kernel
- Note all similarity function make valid kernel. They need to pass a condition explained in the "Mercer's theorem". The gaussian is very useful, other famous ones are polynomial kernel, string kernel, chi-square kernel, histogram intersection kernel

About multi-class classification: Use one-vs-all (train a SVM for each class).

Logistic regression vs. SVMs:
- If n is large relative to m: Use logistic regression, or SVM without a kernel (linear kernel).
- If n is small, m is intermediate: Use SVM with a gaussian kernel.
- If n is small, m is large: create/add more features, then use logistic regression or SVM without a kernel.
- A (well designed) neural network is likely to work well for all of these settings, but may be slower to train.

