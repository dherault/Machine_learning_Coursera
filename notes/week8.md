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

