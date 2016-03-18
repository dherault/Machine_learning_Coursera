# Week 2

Octave docs: [link](http://www.gnu.org/software/octave/doc/interpreter/)

### Multiple features

Multiple variables

n = number of features

`h(x) = t0 + t1 * x1 + t2 * x2 + ... + tn * xn`

So in vector shape: `h(x) = transpose([t0, t1, ..., tn]) * [x0 , x1 , ..., xn]`

### Gradient feature scaling

idea: make sure features are on a similar scale (to make gradient converge more quickly)

also, mean normalization: replace `x_i` with `x_i - mu_i` where mu\_i is the mean of x\_i

### Gradient learning rate

"Debugging", make sure it's working correctly: J(t) should decrease after every iteration (if a is small enough)

### Polynomial regression

`h(x) = t0 + t1 * x1 + ... + tn * xn` where xi = (feature)^i

### Normal equation

t is the matrix of t parameters (not sure about that)

`t = (transpose(X) * X)^(-1) * transpose(X) * y`

On Octave: `pinv(X'*X)*X'*Y`

Gradient descent:
- Need to choose a
- Needs many iterations
- Works well with large number of features

Normal Equation:
- No need to choose a
- Don't need to iterate
- Need to compute (transpose(X) * X)^(-1)
- Slow if large number of features
