# Week 1

### Supervised vs unsupervised machine learning

Supervised: “Right answers” given 
- Regression problem: predict *continuous* valued output
- Classification problem: predict *discrete* valued output

Unsupervised: 
- Clustering algorithms
- Unlabeled data
- Used to make sense out of raw data, to organize it
- Cocktail party problem: [W, s, v] = svd((repmap(sum(x.\*x, 1), size(x, 1), 1).\*x)\*x');

--> Use Octave for prototyping

### What is macine learning ?

[Full text](https://www.coursera.org/learn/machine-learning/supplement/X64SM/introduction)

Two definitions of Machine Learning are offered. Arthur Samuel described it as: "the field of study that gives computers the ability to learn without being explicitly programmed." This is an older, informal definition.

Tom Mitchell provides a more modern definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

Example: playing checkers.

- E = the experience of playing many games of checkers
- T = the task of playing checkers.
- P = the probability that the program will win the next game.

--> See Introduction.md

### Linear regression with one variables

[Full text](https://www.coursera.org/learn/machine-learning/supplement/Mc0tF/linear-regression-with-one-variable)

For regression problems.
*m* is the size of the training set.

Some better functions can be found [here](http://math.stackexchange.com/questions/70728/partial-derivative-in-gradient-descent-for-two-variables/189792#189792)

**Hypothesis function**: `h(x) = t0 + t1 * x` (aka predictor function)

**Cost function**: `J(t0, t1) = (1 / 2m) * sum(i=1..m, (h(x\_i) - y\_i)²)`

The cost function measure the accuracy of the predictor function.

**Gradient descent**: repeat until convergence: `t\_j := t\_j - a * (d J(t0, t1) / d t\_j)`

**Gradient descent for linear regression**: repeat until convergence:
- `t0 := t0 - (a / m) * sum(i=1..m, (h(x\_i) - y\_i))` notice no square, no half !== cost function
- `t1 := t1 - (a / m) * sum(i=1..m, (h(x\_i) - y\_i) * x\_i)`

!silmultaneous update!

-temp0 := fun0
-temp1 := fun1
-t0 := temp0
-t1 := temp1

### Linear algebra review

Just syk: A\_ij: ith row and jth column

### Model representation

How to represent our hypothesis function h ?

easy answer: `t0 + t1 * x` (lin reg with 1 var === univariate linear regression)

### Cost function

J is called a squared error function

### Gradient descent

Used to minimize cost functions.

repeat until convergence: `t\_j := t\_j - a * (d J(t0, t1) / d t\_j)`
`a` is call the "learning rate". If `a` is too small then the descent can be too slow. If it's too large, then the descent can overshoot the minimum, may fail to converge, or even diverge.
