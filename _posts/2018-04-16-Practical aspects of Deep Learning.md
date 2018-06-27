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
When we start to solve a new problem in using deep learning, we can never decide earlier what to choose like - what learning rate? how many hidden layers? which activation functions to choose and for which layers? To find the answer we have to follow iterative process of ``Idea ---> code ---> Experiment``. Even seasoned deep learning expert find it almost impossible to correctly guess the best choice of hyperparameters the very first time.  


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

> Generally, we have give just Training set and Test set in real life and also in Kaggle competitions. To construct the development set use following technique:
* Test set size = 10,000,000, then split like below
*  Test set size = index from 1 - 9,900,000 i.e. 98% of original test set
* Development set = index from 9,900,001 - 10,000,000 i.e. 1% of original test sets.

**You will try to build a model upon training set then try to optimize hyperparameters on dev set as much as possible. Then after your model is ready you try and evaluate the testing set.**

`Make sure the dev and test set are coming from the same distribution`. Suppose training pictures is from the web and the dev/test pictures are from users cell phone they will mismatch. It is better to make sure that dev and test set are from the same distribution.

For more knowledge on Train/dev/test set distributions topic follow [Machine Learning Strategy In Deep Learning](https://bikash-jaiswal.github.io/2018/04/20/Machine-Learning-Strategy-in-Deep-Learning.html) 3rd course in Deep learning specialization.  

##### 2. Bias - Variance

One of the quick question to evaluate someone's technical skill on Machine Learning or Deep Learning is to ask him/her about bias/ Variance and Overfitting and Underfitting.

`It is simple and vital topic but difficult to master.`

More information on the problem of overfitting and underfitting can be found on [The problem of Underfitting](https://github.com/bikash-jaiswal/Machine-Learning-Notes/blob/master/3.a.The-problem-of-overfitting.md) course for Machine learning taught by Andrew Ng in [coursera](https://www.coursera.org/learn/machine-learning).

<figure>
  <div style="text-align:center">
    <img src="/assets/img/practical-aspect/bias-variance.png" alt="Bias-variance"/>
    <figcaption> Bias-Variance </figcaption>
  </div>
</figure>

**Overfitting** : models with overfitting problem has good performance on the training data, poor generliazation to other data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

**Underfitting**: models that can neither model the training data nor generalize to new data. It is usually caused by a function that is too simple or uses too few features.

Analysis Bias and Variance with training error rate and validation error value. Assumption made for below comparison: human error = 0%
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
As of solution listed in case of high variance, getting more data is not always feasible in every case of application, nevertheless, trying regularization is always possible.

##### Applying regularization in logistic regression
Here, `||W|| = sum of absolute values of all weight` and the cost function of logistic regression is `J(w,b) = (1/m) * Sum(L(y(i),y'(i)))` and `lambda` is  regularization parameter (another hyperparameter).
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

* `(lambda/2m) * Sum((||W[l]||^2)`  penalizes the weight matrices from being too large.

##### How does regularization prevent overfitting
Some Intuition that can help us in finding why and how does regularization help with overfitting and reducing variance problems?

`Intuition I`: If regularization parameter `lambda` is too big.
  * a lot of w's will be close to zeros which will make the Neural network simpler (you can think of it as it would behave closer to logistic regression with more hidden layers).
  <figure>
    <div style="text-align:center">
      <img src="/assets/img/practical-aspect/simpler-nn.png" alt="simpler-nn"/>
      <figcaption> NN when lambda is too large </figcaption>
    </div>
  </figure>
  * If lambda is good enough: It will just reduce some weights that makes the neural network overfit.

`Intuition II with tanh function`:
  * If lambda is too large, w's will be small (close to zero) - will use the linear part of the tanh activation function, so we will go from non linear activation to roughly linear which would make the NN a roughly linear classifier.
  * If lambda good enough it will just make some of tanh activations roughly linear which will prevent overfitting.

**`Implementation tip`**: If you implement gradient descent, one of the steps to debug gradient descent is to plot the cost function (J) as a function of the number of iterations of gradient descent and you want to see that the cost function J **decreases monotonically** after every elevation of gradient descent with regularization. If you plot the old definition of J (no regularization) then you might not see it decrease monotonically.
monotonically decreassing
<figure>
  <div style="text-align:center">
    <img src="/assets/img/practical-aspect/monotonically decreassing.png" alt="monotonically decreassing"/>
    <figcaption>monotonically decreassing
 </figcaption>
  </div>
</figure>

##### Dropout technique
Dropout	is	a	technique	used	to	improve	over-fit	on	neural	networks,	we	should	use	Dropout	along with	other	techniques	like	L2	Regularization.
<figure>
  <div style="text-align:center">
    <img src="/assets/img/practical-aspect/dropout.png" alt="Dropout"/>
    <figcaption>Dropout
 </figcaption>
  </div>
</figure>

`Dropout Intuition:` Go through each of the layers of the network and set some probability for eliminating a node in neural network.  So, we end up with a much smaller, really much diminished network.

* Implementation of Dropout aka Inverted dropout

```python
keep_prob = 0.8   # 0 <= keep_prob <=1
# the generated number that are less than 0.8 will be dropped.
#80% stay, 20% dropped
l = 3  # this code is only for layer 3

d3 = np.random.randn(a[l].shape[0], a[l].shape[1]) < keep_prob

a3 = np.multiply(a3,d3)   # keep only the values in d3

# increase a3 to not reduce the expected value of output
# (ensures that the expected value of a3 remains the same)
# to solve the scaling problem
a3 = a3 / keep_prob      
```
* If there are 50 neuron than 10 neurons will be shut off

* Vector d[l] is used for forward and back propagation and is the same for them, but it is different for each iteration (pass) or training example.

* At test time we don't use dropout. If you implement dropout at test time - it would add noise to predictions.

##### Understanding Dropout
`Intuition:` Dropout randomly knocks out units in your network. So it's as if on every iteration you're working with a smaller NN, and so using a smaller NN seems like it should have a regularizing effect.

`Another Intution:` can't rely on any one feature, so have to spread out weights.

* It's possible to show that dropout has a similar effect to L2 regularization.
* Dropout can have different keep_prob per layer.
* The input layer dropout has to be near 1 (or 1 - no dropout) because you don't want to eliminate a lot of features.
* If you're more worried about some layers overfitting than others, you can set a lower keep_prob for some layers than others. The downside is, this gives you even more hyperparameters to search for using cross-validation. One other alternative might be to have some layers where you apply dropout and some layers where you don't apply dropout and then just have one hyperparameter, which is a keep_prob for the layers for which you do apply dropouts.
* A lot of researchers are using dropout with Computer Vision (CV) because they have a very big input size and almost never have enough data, so overfitting is the usual problem. And dropout is a regularization technique to prevent overfitting.
* A downside of dropout is that the cost function J is not well defined and it will be hard to debug (plot J by iteration).

* To solve that you'll need to turn off dropout, set all the `keep_probs` to 1, and then run the code and check that it monotonically decreases J and then turn on the dropouts again.

##### Where	to	use	Dropout	layers
Normally	some	deep	learning	models use Dropout on	the	fully	connected	layers,	but	is	also	possible to	use	dropout	after	the	max-pooling	layers,	creating	some	kind	of	image	noise	augmentation.

##### Other regularization methods
* Data augmentation:
  * For example in a computer vision data:
    * You can flip all your pictures horizontally this will give you m more data instances.
    * You could also apply a random position and rotation to an image to get more data.
  * For example in OCR, you can impose random rotations and distortions to digits/letters.
  * New data obtained using this technique isn't as good as the real independent data, but still can be used as a regularization technique.
  <figure>
    <div style="text-align:center">
      <img src="/assets/img/practical-aspect/data-augment.png" alt="data augmentation"/>
      <figcaption>Data Augmentation
   </figcaption>
    </div>
  </figure>

* Early stopping:
  * In this technique we plot the training set and the dev set cost together for each iteration. At some iteration the dev set cost will stop decreasing and will start increasing.
  * We will pick the point at which the training set error and dev set error are best (lowest training cost with lowest dev cost).
  * We will take these parameters as the best parameters.
  <figure>
    <div style="text-align:center">
      <img src="/assets/img/practical-aspect/early-stopping.png" alt="Early stopping"/>
      <figcaption>Early stopping
   </figcaption>
    </div>
  </figure>
  * `Advantages:` you don't need to search a hyperparameter like in other regularization approaches (like lambda in L2 regularization).
  * `Disadvantages:` Early stopping simultaneously tries to minimize the cost function and not to overfit which contradicts the orthogonalization approach (will be discussed further). So, use L2 regularization.

* Model Ensembles:
  * Algorithm:
    * Train multiple independent models.
    * At test time average their results.
  * It can get you extra 2% performance.
  * It reduces the generalization error.
  * You can use some snapshots of your NN at the training ensembles them and take the results.

### Setting up your optimization problem

##### Normalizing input
When training a neural network, one of the techniques that will speed up training is if we have normalized our inputs.

<figure>
  <div style="text-align:center">
    <img src="/assets/img/practical-aspect/normalizing.png" alt="Early stopping"/>
    <figcaption> Normalization of data
 </figcaption>
  </div>
</figure>

* Normalization are going on these steps:
  * First get the mean of the training set: `mean = (1/m) * sum(x(i))`
  * Then, subtract the mean from each input: `X = X - mean` to make your inputs centered around 0.
  * Atlast, get the variance of the training set: `variance = (1/m) * sum(x(i)^2)`and normalize the variance. `X /= variance`.

`Note:` These steps should be applied to all training, dev, and testing sets (but using mean and variance of the train set).  

* Why normalize?
  * If we don't normalize the inputs our cost function will be deep and its shape will be inconsistent (elongated) then optimizing it will take a long time.
  * But if we normalize it the opposite will occur. The shape of the cost function will be consistent (look more symmetric like circle in 2D example) and we can use a larger learning rate alpha - the optimization will be faster.

##### vanishing/exploding gradients
> Vanishing and Exploding gradients ----> During training when gradient becomes either too small or too big respectively.


Consider this 9-layer neural network just after it was initialized.
<figure>
  <div style="text-align:center">
    <img src="/assets/img/practical-aspect/9-layer.png" alt="vanishing"/>
  </div>
</figure>

To understand the problem, suppose that we have a deep neural network with number of layers L, and all the activation functions are linear(identity matrix)  i.e. `g(z)=z` and each`b = 0`.

Then the output activation is: Y = W<sup>L</sup>W<sup>L-1</sup>W[1].....W<sup>2</sup>W<sup>1</sup>X where L=10 and W<sup>1</sup>,W<sup>2</sup>,…,W<sup>L-1</sup> are all matrices of size (2,2) because layers [1] to [L-1] have 2 neurons and receive 2 inputs.

Consider the case where every weight is initialized slightly larger than the identity matrix.

###### Exploding Gradients:
``` python
 l = L
if W[l] = [1.5   0]
          [0   1.5]
```
This simplifies to y'=W<sup>L</sup>1.5<sup>L−1</sup>X and the values of the activation a<sup>l</sup> increase exponentially with l. When these activations are used in backward propagation, this leads to the exploding gradient problem.

###### Vanishing gradients
Similarly, consider the case where every weight is initialized slightly smaller than the identity matrix

``` python
 l = L
if W[l] = [0.5   0]
          [0   0.5]
```
This simplifies to y'=W<sup>L</sup>0.5<sup>L−1</sup>X and the values of the activation a<sup>l</sup> decreases exponentially with l. When these activations are used in backward propagation, this leads to the vanishing gradient problem.


* Recently Microsoft trained 152 layers (ResNet)! which is a really big number. With such a deep neural network, if your activations or gradients increase or decrease exponentially as a function of L, then these values could get really big or really small. And this makes training difficult, especially if your gradients are exponentially smaller than L, then gradient descent will take tiny little steps. It will take a long time for gradient descent to learn anything.

##### Weight initialization

* A partial solution to the Vanishing / Exploding gradients in NN is better or more careful choice of the random initialization of weights
* In a single neuron (Perceptron model): Z = w<sup>1</sup>x<sup>1</sup> + w<sup>2</sup>x<sup>2</sup> + ... + w<sup>n</sup>x<sup>n</sup>
  * So if `n_x` is large we want W's to be smaller to not explode the cost.

* So it turns out that we need the variance which equals `1/n_x` to be the range of W's
So lets say when we initialize W's like this (better to use with tanh activation):

Maintaining the value of the variance of the input and the output of every layer guarantees no exploding/vanishing gradient(`ReLU + Weight Initialization with variance`). The recommended initialization is Xavier initialization (or one of its derived methods), for every layer `l`:

``` python
np.random.rand(shape) * np.sqrt(1/n[l-1])
```
or variation of this (Bengio et al.):
``` python
np.random.rand(shape) * np.sqrt(1/n[l-1] + n[l])
```
or Setting initialization part inside sqrt to 2/n[l-1] for ReLU is better
``` python
np.random.rand(shape) * np.sqrt(2/n[l-1])
```
##### Numerical approximation in gradients
$W^{[l]}

##### Gradient checking


### Different Optimization algorithms

### Hyperparameter tuning

### Batch Normalization

### Multiclass classification
