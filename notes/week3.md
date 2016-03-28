# Week 3

## Classification

- Email: spam / Not spam
- Transaction : Fraudulent / not

--> y € {0, 1}
- 0 is the negative class (absence of something)
- 1 is the positive class (presence)

Latter: multiple classes (more than 2)

We could  threshold with a linear regression, but it poses problems
--> logistic regression (really: classification)

## Logistic regression model

`h(x) = g(theta' * x)`
where
`g(z) = 1 / (1 + e^-z)` (Sigmoid function / Logistic function)
i.e.
`h(x) = 1 / (1 + e^-(theta' * x))`

`h(x)` is the estimated probability that `y = 1` on input `x`
i.e.
`h(x) = p(y=1 | x; theta)`

we have:
`p(y=1 | x; theta) + p(y=0 | x; theta) = 1`

## Decision boundary

`y = 1 if h(x) >= 0.5`

given our sigmoid function, we have
`g(z) >= 0.5 when z >= 0`
i.e.
`g(theta' * x) >= 0.5 when theta' * x >= 0`
i.e.
`y = 1 if theta' * x >= 0`

What about non-linear decision boundaries?
consider `h(x) = g(t0 + t1*x1 + t2*x2 + t3*x1² + t4*x2²)`
or more complex functions

## Cost function

How do we coose/fit our parameters theta?

we define Cost, with `Cost(h(x), y) = 0.5 * (h(x) - y)²`
pb: with g(z), this is non-convex, so it cannot be minized easily

so we are going to use:
`Cost(h(x), y) = `
`-log(h(x)) if y = 1`
`-log(1 - h(x)) if y = 0`

## Logistic regression

`J(theta) = (1 / m) * sum(i=1..m, Cost(h(x), y))`

we can also write Cost as follow:
`Cost(h(x), y) = -y * log(h(x)) - (1 - y) * log(1 - h(x))`

## Gradient descent

`theta_j = theta_j - alpha * (d J(theta)) / (d theta_j)`
i.e.
`theta_j = theta_j - alpha * sum(i=1..m, (h(x(i)) - y(i)) * x_j)`
'looks' identical to linear regression gradient descent (h is in fact different)

## Advanced optimization

To calculate theta, they are more sophisticated algorithms like:
- Gradient descent (you know it!)
- Conjugate gradient
- BFGS
- L-BFGS

## Multi-class classification: 

Using an algo named 'one vs all' (also: 'one vs rest')

## Regularization

### Overfitting

Low quality models (such as `h(x) = t0 + t1*x`) can cause 'underfitting' or 'high bias'
The opposite is 'overfitting' or 'high variance'

How to address overfitting ?
- Reduce the number features (manually or with an algo)
- Regularization (works well with a lot of features)

### Cost function

One could penalize some high degree parameters theta
Having small values for theta leads to
- Simpler hypothesis
- Less prone to overfitting

By convention, we do not regularize theta_0
Our regularized cost function would look like:
`J(theta) = (1 / (2 * m)) * (sum(i=1..m, (h(X) - y)²) + lambda * sum(i=1..m, theta_i²))`
Where lambda is our regularization parameter.

### Gradient descent

Do not forget to not penalize theta_0
Gradient descent:
`theta = theta - (alpha / m) * (sum(i=1..m, (h(x) - y) * x) + lambda * theta)`
`theta = theta * (1 - alpha * lambda / m) - (alpha / m) * sum(i=1..m, (h(x) - y) * x)`

We must have `1 - alpha * lambda / m < 1`

### Normal equation

We can use regularization too:
`theta = inv(X' * X + lambda * xidentity) + X' * y`
Where xidentity is the identity matrix but with a 0 at value 1, 1

If the number of examples is inferior to te number of features, X' * X is non invertible!!! (must look into that)
Adding `lambda * xidentity` makes `X' * X + lambda * xidentity` invertible (proove it! :p)

### Regularized logistic regression

`J(theta) = (-1 / m) * sum(i=1..m, y * log(h(X)) + (1 - y) * log(1 - h(X))) + (lambda / 2m) * sum(i=1..m, theta²)`
