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
Our classical Gradient descent (summing over all examples) is called Batch Gradient Descent
