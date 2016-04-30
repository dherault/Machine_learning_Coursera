# Week 8

## Clustering

### Unsupervised Learning: Introduction

Unlabelled data sets.

We'll start with clustering.

## K-Means Algorithm

Cluster centroids, moved to their "new means". A step is called "cluster assignment step".

Algo:
Input:
- K: number of clusters
- Training set xi

Output: K positions (of the K cluster centroids)

```
Randomly initialize K cluster centroids mu_i

repeat: {

  // cluster assignment step
  for i = 1 to m {
    ci = index (from 1 to K) of cluster centroid closest to xi
  }

  // move centroid step
  for k = 1 to K {
    mu_k = average (mean) of points assigned to cluster k
  }
}
```

Eliminate cluster centroids without points assigned to it.

### Optimization Objective

All the algos that we've seen have an optimization objective. Untill now it was some Cost function.

mu_c-i is the position of the cluster centroid assigned to xi.

objective: `min J = min (1 / m) * sum(i=1..m, norm(x_i - mu_c-i)²)`.

J is called the distortion.

The cluster assignment step minimizes J wrt ci.
The move centroid step minimizes J wrt mu_k.

### Random Initialization

Lots of ways to randomize mu_k, but one is king:

- Should have K < m.
- Randomly pick K training examples.
- Set the mu_ks equal to these examples.

K-mean can get stuck at some bad local optima (!). To avoid that, we can try multiple random initilizations/

```
for i = 1 to 100 { // say 50 to 1000
  Randomly initialize K-mean
  Run K-means
  Compute cost function J
}

Pick clustering that gave the lowest distortion
```

### Choosing the Number of Clusters

There is no great way of doing that.
Best: manually by looking at vizualizations.

Elbow method: plot J as a function of K. And look for an "elbow"

Or based on business requirements.

## PCA

### Motivations

Dimensionality reduction: reduce number of features, to compress the data.
We project data on a linear subspace of lower dimension basically.

### Principal Component Analysis

#### Principal Component Analysis Problem Formulation

PCA. Tries to find a surface that minimizes the projection error.
Before applying PCA, perform mean normalization and feature scaling.
In pratice: find a vector u-1 onto which to project the data as to minimize the projection error.
Reduce from n-dim to k-dim: Find k vectors.
french: Ces k vecteurs definissent un SEV de projection. (linear subspace).

PCA is not linear projection. Linear projection tries to predict y given x, by minimizing VERTICAL error.
PCA tries to find a vector that minimizes ORTHOGONAL (projected) error. No prediction.

#### Principal Component Analysis Algorithm

Data preprocessing: (feature scaling and mean normalization (ensures every feature has 0 mean))
```
mu_j = (1 / m) * sum(i=1..m, x_j-i)
replace each x_j-i with x_j - mu_j
scale features. (divide by max-min or more commonly the standard deviation)
```

PCA Algo:
```
compute the covariance matrix:
Sigma = (1 / m) * sum(i=1..n, (x-i) * (x-i)')

compute its eigenvectors:
[U, S, V] = svd(Sigma);
```

SVD stands for Singular Value Decomposition. In octave, svd is more numerically stabl that the eig function.
A covariance matrix is always symetric positive semi-definite. (?)
We need the U matrix (n by n). Its columns are our vectors u-i. All we have to do is take its first k vectors.

```
U_reduce is a subset of U, its first k columns (n by k)
U_reduce = U(:,1:k);
z = U_reduce' * x;
```

Z is k by 1.

also, vectorized Sigma:
```
Sigma = (1 / m) * X' * X
```

### Applying PCA

#### Reconstruction from compressed representation

`X_approx = U_reduce * z`

#### Choosing the number of principal components

k is the number of prinsipal components.

PCA tries to minimize the average squared projection error: `(1 / m) * sum(i=1..m, norm(x-i - x_approx-i)²)`.

Let's define the total variation of the data as: `(1 / m) * sum(i=1..m, norm(x-i)²)`.

Typically choose k to be the smallest value so that `average squared projection error / total variation of data <= 0.01 (1%)`
--> 99% of variance is retained. (or 95%, or whatever)

How to choose k:
```
i = 0
do {
  i++
  try PCA with k = i
  Compute U_reduce, z-1...z-m, x_approx-1...x_approx-m
} while (stuff / stuff <= 0.01)
```

calling SVD also gives S, a diagonal matrix (values: S_i_i)
stuff / stuff can be computed by: `1 - sum(i=1..k, S_i_i) / sum(i=1..n, S_i_i)`

--> Check that `sum(i=1..k, S_i_i) / sum(i=1..n, S_i_i) >= 0.99` by incresing k until the condition is met.

#### Advice for applying PCA

Supervised learning speedup: if n is large (like in pictures pixels: 100x100 picture --> dim 10000), apply PCA to go down to dimension 1000.
The x to z mapping should be computed using the training set, not the cv or test set.

Application of PCA:
- Compression (reduce memory/disk needed to store data, speed up learning algorithm)
- Visualization (k=2, k=3)

Bad use: to prevent overfitting by reducing the number of features. BAD! Not a good way to address overfitting.
Use regularization instead.
PCA does not use Y (the label) and deletes information.

A good question: why about not using PCA ?
Use the original raw data. If that does not do what you want, then implement PCA.
