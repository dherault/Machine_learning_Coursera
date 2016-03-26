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

