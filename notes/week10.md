# Week 10

## Large Scale Machine Learning

### Gradient Descent with Large Datasets

#### Learning With Large Datasets

Pb: With a million examples, computing one step of the gradient descent would mean summing a million examples
Waht if we train only on a thousand examples?

To sanity check:
Plot J_train and J_cv as a function of m. If the curves are close of each other (high bias) at m=1000, then m=1000000 would not help,
one should try adding more features

#### Stochastic Gradient Descent
!
Our classical Gradient descent (summing over all examples) is called Batch Gradient Descent.

Stochastic Gradient Descent algo:
```
cost(theta, (x-i, y-i)) = 0.5 * (h_theta(x-i) - y-i)²

J_train(theta) = (1 / m) * sum(cost(theta, (x-i, y-i)))

1. Randomly shuffle dataset (speeds up the convergence)
2. Repeat { (onceis enough, 10 times is typical)
  for i=1..m {
    theta_j = theta_j - alpha * (h_theta(x-i) - y-i) * x_j-i (for j=0..n)
    ie. theta_j = theta_j - alpha * d cost(theta, (x-i, y-i)) / d theta_j
  }
}
```

SDG doesnt converge to the local minimum like BGD does, but wanders around which is good enought.
After going throught the entire dataset, BGD has advanced of only one step, whereas SGD can already be close to the local minimum, having done m steps

#### Mini-Batch Gradient Descent

- BGD: use m examples in each iteration
- SGB: use 1 example in each iteration
- mini-BGD: use b examples in each iteration

b is called the mini-batch size.

mini-BGD algo:
```
Say b = 10, m = 1000

Repeat {
  for i = 1, 11, 21, ..., 991 {
    theta_j = theta_j - (alpha / 10) * sum(k=i..i+9, (h_theta(x-k) - y-k) * x_j-k) (for j = 0..n)
  }
}

ie.

Repeat {
  for i = 1..m step b {
    theta_j = theta_j - (alpha / 10) * sum(k=i..i + b - 1, (h_theta(x-k) - y-k) * x_j-k) (for j = 0..n)
  }
}
```

mini-BGD can perform better than SGD if we have a good vectorization implementation.

#### Stochastic Gradient Descent Convergence

How to know if SGD is doing ok, and how to tune the learning rate alpha.

Idea:
compute the cost before upgrading theta, using x-i, y-i (and the not updated theta)
Every (say) 1000 iterations, plot cost(theta, (x-i, y-i)) averaged over the last 1000 examples.

Checking for convergence:
Plot cost(theta, (x-i, y-i)) in function of the number of iterations, averaged over the last (say) 1000 examples.
Do the same plot for many learning rates
Don't use too many examples! The average over too many examples might hinder some information.
If the curve is growing, then use a smaller alpha.

To have greater chance to find the (local...) minimum of J, one could use a decreasing alpha.
`alpha = const1 / (iterationNumber + const2)`;
But you'll need to spend time playing with const1 and const2...

### Advanced Topics

#### Online Learning

Learn from a continious stream of data. Online learning algos can adapt to changing user preferences.
```
Repeat everytime a new user flows in {
  Get (x, y) corresponding to the user
  Update Theta using (x, y) {
    theta_j = theta_j - alpha * (h(x) - y) * x_j (for j = 0..n)
  }
}
```

#### Map Reduce and Data Parallelism

Some pb are too large to be solved on only one machine. Too much data.

BGD: (m == 400)
```
theta_j = theta_j - (alpha / 400) * sum(i=1..400, (h(x-i) - y-i) * x_j-i)
```
Distributed:
```
Machine 1: temp_j-1 = sum(i=1..100, (h(x-i) - y-i) * x_j-i)
Machine 2: temp_j-2 = sum(i=101..200, (h(x-i) - y-i) * x_j-i)
Machine 3: temp_j-3 = sum(i=201..300, (h(x-i) - y-i) * x_j-i)
Machine 4: temp_j-4 = sum(i=301..400, (h(x-i) - y-i) * x_j-i)
```
Combined:
```
theta_j = theta_j + (alpha / 400) * (temp_j-1 + temp_j-2 + temp_j-3 + temp_j-4)
```
For j = 0..n

Machine Learning algos can be expressed as computing sums of functions over the training set.

Or do the same on many computer cores.
  
