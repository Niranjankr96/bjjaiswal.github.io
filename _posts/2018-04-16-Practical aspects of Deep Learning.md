---
layout: post
title: Practical aspects of Deep Learning
date: 2018-04-16
categories:  Deep Learning Notes
cover: 'https://i2.wp.com/philosophyofbrains.com/wp-content/uploads/2016/05/Problem-Mental-Causation-Pic.jpg?resize=1280%2C585'
tags: Deep-Learning
---



After implementation of a neural network, we have gained enough vim to better understand the black box of Deep Learning and to learn all the practical aspects of making our neural network work even better. We are going to learn various tricks or technique like:
1. Hyperparameter tuning
2. Setting the datasets
3. Optimization algorithm- makes learning algorithm to learn in a reasonable time

## Practical aspects of Deep Learning
When we start to solve a new problem in using deep learning, we can never decide earlier what to choose like what learning rate? how many hidden layers? which activation functions to choose and for which layers? To find the answer we have to follow iterative process of ``Idea ---> code ---> Experiment`` Even season deep learning expert find it almost impossible to correctly guess the best choice of hyperparameters the very first time.  


### Setting up your Machine Learning Application
##### 1. Train / Dev / Test sets
Your data will be split into three parts:
  * Training set. (Has to be the largest set)
  * Development or "dev" set.
  * Testing set.

<figure>
  <div style="text-align:center">
  <img src="/assets/img/ml-strategies/new-dist.png" alt="my alt text"/>
      <figcaption> Strategy of distribution in Deep Learning </figcaption>
    </div>
  </figure>

> Generally we have give just Training set and Test set in real life and also in Kaggle competitions. To construct the development set use following technique:
* Test set size = 10,000,000, then split like below
*  Test set size = index from 1 - 9,900,000 i.e. 98% of original test set
* Development set = index from 9,900,001 - 10,000,000 i.e. 1% of original test sets.

**You will try to build a model upon training set then try to optimize hyperparameters on dev set as much as possible. Then after your model is ready you try and evaluate the testing set.**

`Make sure the dev and test set are coming from the same distribution`. suppose training pictures is from the web and the dev/test pictures are from users cell phone they will mismatch. It is better to make sure that dev and test set are from the same distribution.

For more knowledge on Train/dev/test set distributions topic follow [Machine Learning Strategy In Deep Learning](https://bikash-jaiswal.github.io/2018/04/20/Machine-Learning-Strategy-in-Deep-Learning.html) 3rd course in Deep learning specialization.  

##### 2. Bias - Variance

One of the quick question to evaluate someone technical skill about Machine Learning or Deep Learning is to ask him/her about bias/ Variance and Overfitting and Underfitting.

It is simple and vital topic but difficult to master.

More information on the problem of overfitting and underfitting can be found on [The problem of Underfitting](https://github.com/bikash-jaiswal/Machine-Learning-Notes/blob/master/3.a.The-problem-of-overfitting.md) course for Machine learning taught by Andrew Ng in [coursera](https://www.coursera.org/learn/machine-learning).

<figure>
  <div style="text-align:center">
    <img src="/assets/img/practical-aspect/bias-variance.png" alt="Bias-variance"/>
    <figcaption> Bias-Variance </figcaption>
  </div>
</figure>

**Overfitting** : models with overfitting problem has good performance on the training data, poor generliazation to other data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

**Underfitting**: models that can neither model the training data nor generalize to new data. It is usually caused by a function that is too simple or uses too few features.

Analysis Bias and Variance with training error rate and validation error value. For Example: assuming human error = 0%
* High variance (overfitting):
  * Training error: 1%
  * Dev error: 11%
* High Bias (underfitting):
  * Training error: 15%
  * Dev error: 14%
* High Bias (underfitting) && High variance (overfitting) :
  * Training error: 15%
  * Test error: 30%
* Best:
  * Training error: 0.5%
  * Test error: 1%

To know more about human level performance see [Machine Learning Strategy In Deep Learning](https://bikash-jaiswal.github.io/2018/04/20/Machine-Learning-Strategy-in-Deep-Learning.html) 3rd course in Deep learning specialization.


##### 3. Basic Recipe for Machine Learning
* If your algorithm has a high bias:

  * Try to make your NN bigger (size of hidden units, number of layers)

 * Try a different model that is suitable for your data.
    Try to run it longer.
    Different (advanced) optimization algorithms.

* If your algorithm has a high variance:
 * More data.
 * Try regularization.
 * Try a different model that is suitable for your data.

* You should try the previous two points until you have a low bias and low variance.

* In the older days before deep learning, there was a "Bias/variance trade-off". But because now you have more options/tools for solving the bias and variance problem its really helpful to use deep learning.

* Training a bigger neural network never hurts.

### Regularizing your neural network
As of solution listed in case of high variance, getting more data is not always feasible in every case of application nevertheless, trying regularization is always possible.

##### Applying regularization in logistic regression
Here, `||W|| = sum of absolute values of all w` and the cost function of logistic regression is ` J(w,b) = (1/m) * Sum(L(y(i),y'(i)))` and `lambda` is  regularization parameter (another hyperparameter).
<figure>
  <div style="text-align:center">
    <img src="/assets/img/practical-aspect/regularization-LR.png" alt="regularization in Logistic regression"/>
    <figcaption> Regularization in Logistic Regression </figcaption>
  </div>
</figure>

Though `L2 regularization is used much` despite `L1 regularization`, people thinks, can help with compressing the model as it makes `W` sparse by setting some of W's value zero and also uses less memory to store the model.  

##### Regularization in Neural Network.

<figure>
  <div style="text-align:center">
    <img src="/assets/img/practical-aspect/regu-nn.png" alt="regularization in Neural network"/>
    <figcaption> Regularization in Neural Network </figcaption>
  </div>
</figure>

* Update the gradient of `W` by:
  * `dW[l] = (from backpropagation) +  lambda/m * W[l] `
  * `W[l] = W[l] -lr*dW[l]`

```
w[l] = w[l] - lr * dw[l]
     = w[l] - lr * ((from back propagation) + lambda/m * w[l])
     = w[l] - (lr*lambda/m) * w[l] - lr * (from back propagation)
     = (1 - (lr*lambda)/m) * w[l] - lr * (from back propagation)
```   
* The new term `(1 - (learning_rate*lambda)/m) * w[l]` causes the weight to decay in proportion to its size.

### Setting up your optimization problem

### Different Optimization algorithms

### Hyperparameter tuning

### Batch Normalization

### Multiclass classification
