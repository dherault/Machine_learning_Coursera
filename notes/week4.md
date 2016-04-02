# Week 4

## Neural network representation

If n (number of features) is too large, classification (and other) problems become too hard too compute!

## Neurons and the brain

Algorithms that try to mimic the brain. The brain may have a "one learning algorithm" that can learn/adapt to basically anything (quantum?)

## Model representation

In the brain, Dendrite are "input wires" and Axan are "output wires"
(Think X and h(X)) X0 is called the bias unit (or bias neuron)

In neural networks, we call the Sigmoid function the "activation function"

Notation:
- a_i-j = "activation" of unit i in layer j
- theta-j = matrix of weights controlling function mapping from layer j to layer j + 1

theta-j will be of dimension s_j+1 * (s_j + 1)
where s_i is te number of units in layer i

Vectorized implementation
a_3-2 = g(z_3-2)
<-->
a-2 = g(z-2)
a-2 = g(theta-1 * a-1)
h(X) = a-3 = g(z-3) = g(theta-2 * a-2)

(don't forget bias units! : add a_0-2 = 1)

This is called forward propagation
Anything that is not an input layer or an output layer is called an hidden layer

# Examples and intuition

- Xor-XNor
- Or
- Not x_1

Yann LeCun

# Multiclass classification

